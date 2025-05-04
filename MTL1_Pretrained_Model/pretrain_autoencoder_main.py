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
from utils import AverageMeter
from tqdm import tqdm
import gc
from hydranet import *
from simclr import *
import torch.optim as optim
from autoencoder import HydraAutoencoder
from dataset import NYUDv2Autoencoder
from scheduler import CustomScheduler

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
    parser.add_argument('--out_full_autoencoder_chkpt_file', default='./pretrained_full.autoencoder.pth.tar', type=str)
    parser.add_argument('--out_encoder_chkpt_file', default='./pretrained_hydranet_encoder.autoencoder.pth.tar', type=str)
    parser.add_argument('--out_decoder_chkpt_file', default='./pretrained_hydranet_decoder.autoencoder.pth.tar', type=str)
    parser.add_argument('--out_pretrain_loss_file', default='./pretrained_loss_file.autoencoder.npz', type=str)
    parser.add_argument('--out_pretrain_loss_figfile_name', default='./pretrained_loss_fig.autoencoder.png', type=str)
    parser.add_argument('--out_dir', default='./pretrained_output', type=str)
    parser.add_argument("--final_linear_decay", action="store_true", help="Whether to do linear decay after cosin annealing.")
    return parser    

parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

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

transform_pretrain = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std),
])

#Saver
init_vals = (0.0, 10000.0)
comp_fns = [operator.gt, operator.lt]

'''
saver = Saver(
    args=locals(),
    ckpt_file=args.out_full_autoencoder_chkpt_file,
    best_val=init_vals,
    condition=comp_fns,
    save_several_mode=all,
)
'''

if args.load_init == 1:
    ckpt_path = ''
    train_losses = []
    start_epoch = 0
elif args.load_resume == 1:
    ckpt_path = args.out_full_autoencoder_chkpt_file

    # Load Checkpoint
    checkpoint = torch.load(ckpt_path)
    #start_epoch, state_dict, start_opt_dict, start_lr = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "state_dict", "optimizer_state_dict", "learning_ragt"],)

    # Load the .npz file
    loaded_data = np.load(f"{args.out_pretrain_loss_file}")

    # Extract arrays
    train_losses = list(loaded_data["train_losses"])
else:
    train_losses = []
    start_epoch = 0
    ckpt_path = ''

# Load Dataset
pretrain_dataset = NYUDv2Autoencoder("./nyud/rgb", transform_pretrain, img_mean, img_std)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

# Initialize Model and Optimizer
num_classes, num_tasks = 40, 2
model = HydraAutoencoder(num_classes=num_classes, num_tasks=num_tasks).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr_enc, betas=(0.9, 0.999), weight_decay=1e-6)

# ============ Cosine Annealing Scheduler ================#
learning_rate_enc = args.lr_enc  # Max LR, 9e-4
warmup_steps_enc = args.cas_warmup_steps_enc   # Steps for warm-up
total_steps = args.early_stop_iter if args.early_stop_iter is not None else args.max_iter  # Total training steps
min_lr_enc = args.cas_min_lr_enc         # Minimum LR after decay
final_lr_enc = args.cas_final_lr_enc         # Final minimum LR for convergence
T_0_enc = args.cas_T_0_enc
T_mult_enc = args.cas_T_mult_enc

cas_scheduler_enc = CustomScheduler(optimizer, warmup_steps_enc, total_steps, min_lr_enc, learning_rate_enc, final_lr_enc, T_0_enc, T_mult_enc, args.final_linear_decay)

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

# Loss function:
#loss_fn = nn.L1Loss()
loss_fn = nn.SmoothL1Loss(beta=1.0)

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

    index = 0
    for x_aug, x_target in pbar:
        x_aug, x_target = x_aug.cuda(), x_target.cuda()

        # Feature collapse check every few epochs
        if epoch % args.show_std_freq_epoch == 0 and epoch > 0:
            encoder = model.hydranet.extract_encoder(init=False).cuda()
            with torch.no_grad():
                z = encoder(x_aug)  # Output from layer8
                std = z.std(dim=0).mean().item()
                print(f"Epoch {epoch} | Encoder Feature Std: {std:.6f}")
                if std < 0.05:
                    print("Warning: Feature collapse suspected (avg std < 0.05).")            

        x_recon = model(x_aug) #(B, C, H/4, W/4)
        x_recon = F.interpolate(x_recon, size=x_target.shape[2:], mode="bilinear", align_corners=False)
        loss = loss_fn(x_recon, x_target)

        if loss.item() > 10:
            print(f"[Warning] High loss {loss.item():.2f} at epoch {epoch}, batch {index}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        avg_losses.append(loss.item())
        loss_meter.update(loss.item())
        pbar.set_description(
                "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )
        index += 1
 
    train_losses.append(np.mean(avg_losses))
    if epoch%args.save_freq == 0 and epoch > 0:
        checkpoint = {
        'state_dict': model.state_dict(),  # Save model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
        'epoch': epoch,  # Save current epoch
        'learning_rate': optimizer.param_groups[0]['lr'],  # Save current LR
        }
        torch.save(checkpoint, os.path.join(args.out_dir, os.path.basename(args.out_full_autoencoder_chkpt_file)))
        np.savez(f"{os.path.join(args.out_dir, os.path.basename(args.out_pretrain_loss_file))}",
                train_losses=train_losses)       

# Save encoder weights
torch.save(model.hydranet.extract_encoder(init=False).state_dict(), f'{os.path.join(args.out_dir, os.path.basename(args.out_encoder_chkpt_file))}')

# Save decoder weights (everything except encoder)
full_state_dict = model.hydranet.state_dict()
encoder_keys = model.hydranet.extract_encoder(init=False).state_dict().keys()
decoder_state_dict = {k: v for k, v in full_state_dict.items() if k not in encoder_keys}
torch.save(decoder_state_dict, f'{os.path.join(args.out_dir, os.path.basename(args.out_decoder_chkpt_file))}')

# Save full model (optional)
torch.save(model.hydranet.state_dict(), f'{os.path.join(args.out_dir, os.path.basename(args.out_full_autoencoder_chkpt_file))}')
            
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
plt.savefig(f"{os.path.join(args.out_dir, os.path.basename(args.out_pretrain_loss_figfile_name))}", dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
