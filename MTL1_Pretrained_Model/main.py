#!wget https://hydranets-data.s3.eu-west-3.amazonaws.com/hydranets-data-2.zip && unzip -q hydranets-data-2.zip && mv hydranets-data-2/* . && rm hydranets-data-2.zip && rm -rf hydranets-data-2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from utils import Normalise, RandomCrop, ToTensor, RandomMirror
import torchvision.transforms as transforms
import os
from model_helpers import Saver, load_state_dict
import operator
from utils import AverageMeter
from tqdm import tqdm
from utils import MeanIoU, RMSE, spatial_jitter_loss, variance_loss
from merge_shuff import shuffle_and_split_data
import gc
from hydranet import *
from simclr import *
import torch.optim as optim
from scheduler import CustomScheduler
from dataset import NYUDv2SSL


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
    parser.add_argument('--cas_warmup_steps_enc', default=150, type=int) #3%-5%
    parser.add_argument('--cas_min_lr_enc', default=7e-5, type=float)
    parser.add_argument('--cas_final_lr_enc', default=7e-6, type=float)
    parser.add_argument('--cas_T_0_enc', default=5000, type=int)
    parser.add_argument('--cas_T_mult_enc', default=1, type=int)
    parser.add_argument('--show_lr_freq_epoch', default=50, type=int)
    parser.add_argument('--show_std_freq_epoch', default=50, type=int)
    parser.add_argument('--load_init', default=1, type=int)
    parser.add_argument('--load_resume', default=0, type=int)
    parser.add_argument('--out_chkpt_file', default='./pretrained_all.pth.tar', type=str)
    parser.add_argument('--out_encoder_chkpt_file', default='./pretrained_hydranet_encoder.pth.tar', type=str)
    parser.add_argument('--out_pretrain_loss_file', default='./pretrained_loss_file.npz', type=str)
    parser.add_argument('--out_pretrain_loss_figfile_name', default='./pretrained_loss_fig.png', type=str)
    return parser    

parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
args = parser.parse_args()

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
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), ratio=(0.5, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),

    transforms.RandomGrayscale(p=0.2),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    ], p=0.5),

    transforms.RandomApply([
        transforms.RandomSolarize(threshold=128)
    ], p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std),
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

if args.load_init == 1:
    ckpt_path = ''
    train_losses = []
    start_epoch = 0
elif args.load_resume == 1:
    ckpt_path = args.out_chkpt_file

    # Load Checkpoint
    checkpoint = torch.load(args.out_chkpt_file)
    #start_epoch, state_dict, start_opt_dict, start_lr = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "state_dict", "optimizer_state_dict", "learning_ragt"],)

    # Load the .npz file
    loaded_data = np.load(f"{args.out_pretrain_loss_file}")

    # Extract arrays
    train_losses = list(loaded_data["train_losses"])

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

# Reload and resume the training
if args.load_resume == 1:
    load_state_dict(model, checkpoint['state_dict'])
    print("Model has {} parameters".format(sum([p.numel() for p in model.parameters()])))

    # Load Optimizer State
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore Learning Rate (If Needed)
    for param_group in optimizer.param_groups:
        param_group['lr'] = checkpoint.get('learning_rate', args.lr_enc)  # Restore LR    

    # Restore Epoch (If Continuing Training)
    start_epoch = checkpoint.get('epoch', 0)        

    current_lr_enc = optimizer.param_groups[0]['lr']
    print(f"Epoch {start_epoch} | Encoder Learning Rate: {current_lr_enc:.6f}")  

# Training Loop
for epoch in range(start_epoch, n_epochs+1):
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

        # Random masking (MAE-style): Forces robust representation learning/Simulates occlusion and noise/Promotes context encoding
        if torch.rand(1) < 0.5:
            mask = (torch.rand_like(x_i) > 0.2).float()
            x_i *= mask
        if torch.rand(1) < 0.5:
            mask = (torch.rand_like(x_j) > 0.2).float()
            x_j *= mask        

        z_i_list = model(x_i)  # list of feature maps
        z_j_list = model(x_j)        

        if epoch % args.show_std_freq_epoch == 0 and epoch > 0:
            with torch.no_grad():
                stds = []
                for scale_idx, z in enumerate(z_i_list):
                    std = z.std(dim=0).mean().item()  # std over channel dim
                    stds.append(std)
                    print(f"Epoch {epoch} | Scale-{scale_idx+1} Feature Std: {std:.6f}")                
                avg_std = sum(stds) / len(stds)
                if avg_std < 0.05:
                    print("Warning: Feature collapse suspected (avg std < 0.05).")                    

        #loss = info_nce_loss(z_i, z_j, temperature=0.07)
        #loss = dense_contrastive_loss(z_i, z_j, temperature=0.07)
        jitter_loss = sum(spatial_jitter_loss(z) for z in z_i_list) * 0.1
        weights = [0.5, 0.3, 0.2]  # e.g., higher weight for full-res
        loss = multi_scale_dense_contrastive_loss(z_i_list, z_j_list, temperature=0.07, weights=weights) + jitter_loss

        lamb = 1.0
        if epoch > args.cas_warmup_steps_enc:  # Apply after warmup
            var_reg = sum(weights[i] * variance_loss(z) for i, z in enumerate(z_i_list))
            var_reg += sum(weights[i] * variance_loss(z) for i, z in enumerate(z_j_list))
        else:
            var_reg = 0.0

        loss += lamb*var_reg

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
        'state_dict': model.state_dict(),  # Save model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
        'epoch': epoch,  # Save current epoch
        'learning_rate': optimizer.param_groups[0]['lr'],  # Save current LR
        }
        torch.save(checkpoint, args.out_chkpt_file)
        np.savez(f"{args.out_pretrain_loss_file}",
                train_losses=train_losses)       

# Save the Pretrained Encoder
checkpoint = {
    'state_dict': model.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, args.out_chkpt_file)
torch.save(model.encoder.state_dict(), args.out_encoder_chkpt_file)
#torch.save(model.encoder.state_dict(), args.out_encoder_chkpt_file)
            
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
plt.savefig(f"{args.out_pretrain_loss_figfile_name}", dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory