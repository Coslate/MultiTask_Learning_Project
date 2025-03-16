#!wget https://hydranets-data.s3.eu-west-3.amazonaws.com/hydranets-data-2.zip && unzip -q hydranets-data-2.zip && mv hydranets-data-2/* . && rm hydranets-data-2.zip && rm -rf hydranets-data-2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import Normalise, RandomCrop, ToTensor, RandomMirror
import torchvision.transforms as transforms
import os
from utils import InvHuberLoss
from model_helpers import Saver, load_state_dict
import operator
import json
import logging
from utils import AverageMeter
from tqdm import tqdm
from utils import MeanIoU, RMSE
from merge_shuff import shuffle_and_split_data
import gc
from hydranet import *
from simclr import *
import torch.optim as optim

#clears Python-level unreferenced objects.
gc.collect()  # Collects unused Python objects
torch.cuda.empty_cache()  # Releases GPU memory

#Argument
def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--lr_enc', default=9e-3, type=float)
    parser.add_argument('--max_iter', default=5000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_freq', default=25, type=int)    
    parser.add_argument('--early_stop_iter', default=None, type=int)           
    parser.add_argument('--cas_warmup_steps_enc', default=20, type=int)
    parser.add_argument('--cas_min_lr_enc', default=7e-4, type=float)
    parser.add_argument('--cas_final_lr_enc', default=7e-5, type=float)
    parser.add_argument('--cas_T_0_enc', default=5000, type=int)
    parser.add_argument('--cas_T_mult_enc', default=1, type=int)
    parser.add_argument('--show_lr_freq_epoch', default=50, type=int)
    parser.add_argument('--load_init', default=1, type=int)
    parser.add_argument('--load_resume', default=0, type=int)
    parser.add_argument('--out_chkpt_file', default='./pretrained_hydranet_encoder.pth.tar', type=str)
    return parser    


parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
args = parser.parse_args()

# Dataset for Self-Supervised Learning (No Labels)
class NYUDv2SSL(Dataset):
    def __init__(self, data_path, transform):
        self.image_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_1 = self.transform(image)
        image_2 = self.transform(image)
        return (image_1, image_2)  # Two augmented views        


#DataLoader
img_scale = 1.0 / 255
depth_scale = 5000.0

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
normalise_params = [img_scale, img_mean.reshape((1, 1, 3)), img_std.reshape((1, 1, 3)), depth_scale,]
transform_common = [Normalise(*normalise_params), ToTensor()]

data_file = "./train_list_depth.txt"
with open(data_file, "rb") as f:
    datalist = f.readlines()
datalist = [x.decode("utf-8").strip("\n").split("\t") for x in datalist]
abs_paths = [os.path.join("nyud", rpath) for rpath in datalist[0]]
img_arr = np.array(Image.open(abs_paths[0]))
img_h, img_w, _ = np.shape(img_arr)
crop_size = min(img_w, img_h)

#transform_train = transforms.Compose([RandomMirror(), RandomCrop(crop_size)] + transform_common)
transform_pretrain = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std)
])

#Saver
init_vals = (0.0, 10000.0)
comp_fns = [operator.gt, operator.lt]

saver = Saver(
    args=locals(),
    ckpt_file=args.out_chkpt_file,
    best_val=init_vals,
    condition=comp_fns,
    save_several_mode=all,
)

def get_params(model):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "layer" in name:  # Encoder layers
            encoder_params.append(param)
        else:  # Decoder layers
            decoder_params.append(param)
    return encoder_params, decoder_params

if args.load_init == 1:
    ckpt_path = ''
    train_losses = []
    start_epoch = 0
elif args.load_resume == 1:
    ckpt_path = args.out_chkpt_file
    start_epoch, _, state_dict = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"],)

    # Load the .npz file
    loaded_data = np.load("pretrain_loss_data.npz")

    # Extract arrays
    train_losses = list(loaded_data["train_losses"])

class CustomScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr, max_lr, final_lr, T_0, T_mult):
        """
        Custom Learning Rate Scheduler.
        
        - Warmup (Linear): Increases from min_lr to max_lr over `warmup_steps`
        - Cosine Annealing: Decays from max_lr to mid-range (5e-4) until 8000 steps
        - Final Linear Decay: Reduces from 5e-4 to final_lr over last 2000 steps

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Starting learning rate
            max_lr: Peak learning rate
            final_lr: Final learning rate at the end of training
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.current_step = 0
        self.T_0 = T_0
        self.T_mult = T_mult
        self.cos_anneal_stage = self.warmup_steps+self.T_0
        self.fin_mid_lr = (self.max_lr + self.min_lr)/2

        # Cosine Annealing Phase (Mid-Phase: 5e-4 as transition point)
        #self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(8000 - warmup_steps), eta_min=5e-4)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warm-up
            progress = self.current_step / self.warmup_steps
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        elif self.current_step < self.cos_anneal_stage:
            # Cosine annealing decay
            self.cosine_scheduler.step()
            new_lr = self.cosine_scheduler.get_last_lr()[0]
        else:
            # Final decay to stabilize learning
            progress = (self.current_step - self.cos_anneal_stage) / (self.total_steps - self.cos_anneal_stage)
            new_lr = self.fin_mid_lr + (self.final_lr - self.fin_mid_lr) * progress

        # Apply new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]   

# Load Dataset
pretrain_dataset = NYUDv2SSL("./nyud/rgb", transform_pretrain)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

# Initialize Model and Optimizer
num_classes, num_tasks = 40, 2
model = SimCLR(num_classes=num_classes, num_tasks=num_tasks).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr_enc, betas=(0.9, 0.999), weight_decay=1e-6)

# ============ Cosine Annealing Scheduler ================#
learning_rate_enc = args.lr_enc  # Max LR, 9e-4
warmup_steps_enc = args.cas_warmup_steps_enc   # Steps for warm-up
total_steps = args.early_stop_iter if args.early_stop_iter is not None else args.max_iter  # Total training steps
min_lr_enc = args.cas_min_lr_enc         # Minimum LR after decay
final_lr_enc = args.cas_final_lr_enc         # Final minimum LR for convergence
T_0_enc = args.cas_T_0_enc
T_mult_enc = args.cas_T_mult_enc

cas_scheduler_enc = CustomScheduler(optimizer, warmup_steps_enc, total_steps, min_lr_enc, learning_rate_enc, final_lr_enc, T_0_enc, T_mult_enc)    

# Pretraining Loop
n_epochs = args.max_iter if args.early_stop_iter is None else args.early_stop_iter
for epoch in range(n_epochs):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_meter = AverageMeter()

    avg_losses = []
    pbar = tqdm(pretrain_loader)
    cas_scheduler_enc.step(epoch)
    if epoch % args.show_lr_freq_epoch == 0:
        current_lr_enc = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Encoder Learning Rate: {current_lr_enc:.6f}")  

    for (x_i, x_j) in pbar:
        x_i, x_j = x_i.cuda(), x_j.cuda()
        z_i, z_j = model(x_i), model(x_j)
        loss = info_nce_loss(z_i, z_j, temperature=0.07)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_losses.append(loss.item())
        loss_meter.update(loss.item())
        pbar.set_description(
                "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )
 
    train_losses.append(np.mean(avg_losses))
    if epoch%args.save_freq == 0 and epoch > 0:
        checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        }
        torch.save(checkpoint, args.out_chkpt_file)
        np.savez("pretrain_loss_data.npz",
                train_losses=train_losses)       

# Save the Pretrained Encoder
torch.save(model.encoder.state_dict(), args.out_chkpt_file)
            
# === Save Train Loss & Validation Loss Plot ===
epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig("training_loss.png", dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory