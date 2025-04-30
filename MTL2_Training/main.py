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
from utils import Normalise, RandomCrop, ToTensor, RandomMirror, get_param_groups
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
from hydranet import *
from scheduler import *
from dataset import *
import gc

#clears Python-level unreferenced objects.
gc.collect()  # Collects unused Python objects
torch.cuda.empty_cache()  # Releases GPU memory

# ============= Argument ================#
def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--lr_sigma_seg', default=1e-3, type=float)
    parser.add_argument('--lr_sigma_depth', default=1e-4, type=float)
    parser.add_argument('--lr_enc', default=8e-3, type=float)
    parser.add_argument('--lr_dec', default=8e-4, type=float)
    parser.add_argument('--max_iter', default=5000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_freq', default=100, type=int)    
    parser.add_argument('--early_stop_iter', default=None, type=int)           
    parser.add_argument('--cas_warmup_steps_enc', default=100, type=int) # 3% - 5% of total iterations
    parser.add_argument('--cas_warmup_steps_dec', default=300, type=int) # 5% - 10% of total iterations
    parser.add_argument('--cas_min_lr_enc', default=1e-3, type=float)
    parser.add_argument('--cas_min_lr_dec', default=1e-4, type=float)
    parser.add_argument('--cas_final_lr_enc', default=8e-4, type=float)
    parser.add_argument('--cas_final_lr_dec', default=8e-5, type=float)
    parser.add_argument('--cas_T_0_enc', default=5000, type=int)
    parser.add_argument('--cas_T_0_dec', default=5000, type=int)
    parser.add_argument('--cas_T_mult_enc', default=1, type=int)
    parser.add_argument('--cas_T_mult_dec', default=1, type=int)
    parser.add_argument('--cas_warmup_steps_sigma_seg', default=0, type=int)
    parser.add_argument('--cas_warmup_steps_sigma_depth', default=0, type=int)
    parser.add_argument('--cas_min_lr_sigma_seg', default=1e-5, type=float)
    parser.add_argument('--cas_min_lr_sigma_depth', default=1e-6, type=float)
    parser.add_argument('--cas_final_lr_sigma_seg', default=8e-6, type=float)
    parser.add_argument('--cas_final_lr_sigma_depth', default=8e-7, type=float)
    parser.add_argument('--cas_T_0_sigma_seg', default=5000, type=int)
    parser.add_argument('--cas_T_0_sigma_depth', default=5000, type=int)
    parser.add_argument('--cas_T_mult_sigma_seg', default=1, type=int)
    parser.add_argument('--cas_T_mult_sigma_depth', default=1, type=int)
    parser.add_argument('--show_lr_freq_epoch', default=100, type=int)
    parser.add_argument('--load_init', default=0, type=int)
    parser.add_argument('--load_pretrained', default=0, type=int)
    parser.add_argument('--load_resume', default=1, type=int)
    parser.add_argument('--init_chkpt_file_enc', default=None, type=str)
    parser.add_argument('--init_chkpt_file_dec', default=None, type=str)
    parser.add_argument('--out_chkpt_file', default='./checkpoint.pth.tar', type=str)
    parser.add_argument('--load_metric_file', default='./metrics_loss_data.npz', type=str)
    parser.add_argument('--train_ratio', default=0.9, type=float)
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.05, type=float)
    parser.add_argument('--shuffle_dataset', default=0, type=int)
    parser.add_argument('--val_every', default=100, type=int)
    parser.add_argument('--freeze_enc_epoch', default=500, type=int)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    return parser    


parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
args = parser.parse_args()

depth = sorted(glob.glob("./nyud/depth/*.png"))
seg = sorted(glob.glob("./nyud/masks/*.png"))
images = sorted(glob.glob("./nyud/rgb/*.png"))
os.makedirs(args.output_dir, exist_ok=True)

# ============= Dataset & Dataloader ================#
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

transform_train = transforms.Compose([RandomMirror(), RandomCrop(crop_size)] + transform_common)
transform_val = transforms.Compose(transform_common)
train_batch_size = 4
val_batch_size = 4
train_file = "./train_list_depth.txt"
val_file = "./val_list_depth.txt"
test_file = "./test_list_depth.txt"
CMAP = np.load('cmap_nyud.npy')
if args.shuffle_dataset == 1:
    shuffle_and_split_data(train_file=train_file, val_file=val_file, test_file=test_file, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

#TRAIN DATALOADER
trainloader = DataLoader(HydranetDataset(train_file, transform_train),
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=True,
                          drop_last=False)

#VALIDATION DATALOADER
valloader = DataLoader(HydranetDataset(val_file, transform_val),
                       batch_size=val_batch_size,
                       shuffle=True,
                       num_workers=args.num_workers,
                       pin_memory=True,
                       drop_last=False)

# ============= Loss Deinition ================#
ignore_index = 0
ignore_depth = 0

crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index)
crit_depth = InvHuberLoss(ignore_index=ignore_depth)
#n_epochs = 1000
n_epochs = args.max_iter if args.early_stop_iter is None else args.early_stop_iter

#Saver
init_vals = (0.0, 10000.0)
comp_fns = [operator.gt, operator.lt]

saver = Saver(
    args=locals(),
    ckpt_file=os.path.join(args.output_dir, os.path.basename(args.out_chkpt_file)),
    output_dir=args.output_dir,
    best_val=init_vals,
    condition=comp_fns,
    save_several_mode=all,
)

# ============= Model Instance ================#
num_classes, num_tasks = 40, 2
hydranet_model = HydraNet(num_classes=num_classes, num_tasks=num_tasks, init_head=True)
#hydranet = nn.DataParallel(nn.Sequential(encoder, decoder).cuda()) # Use .cpu() if you prefer a slow death

if torch.cuda.is_available():
    _ = hydranet_model.cuda()

# ============ Optimizer ================#
'''
lr_encoder = 1e-2
lr_decoder = 1e-3
betas_encoder = (0.9, 0.99)
betas_decoder = (0.9, 0.999)
#momentum_encoder = 0.9
#momentum_decoder = 0.9
weight_decay_encoder = 1e-5
weight_decay_decoder = 1e-5
'''
lr_encoder = args.lr_enc
lr_decoder = args.lr_dec
betas_encoder = (0.9, 0.99)
betas_decoder = (0.9, 0.999)
#momentum_encoder = 0.9
#momentum_decoder = 0.9
weight_decay_encoder = 2e-4
weight_decay_decoder = 1e-4
'''Adam'''
# Extract encoder parameters (MobileNetV2)
# Extract decoder parameters (RefineNet)
encoder_params, decoder_params, segm_head_params = get_param_groups(hydranet_model)
optimizer_encoder = torch.optim.AdamW(encoder_params, lr=lr_encoder, betas=betas_encoder,weight_decay=weight_decay_encoder)
optimizer_decoder = torch.optim.AdamW([
    {'params': decoder_params, 'lr': lr_decoder},
    {'params': segm_head_params, 'lr': lr_decoder * 3},  # ← 3× higher LR for segm head
], betas=betas_decoder, weight_decay=weight_decay_decoder)

# ============ Cosine Annealing Scheduler ================#
learning_rate_enc = args.lr_enc  # Max LR, 9e-4
learning_rate_dec = args.lr_dec  # Max LR, 9e-4
warmup_steps_enc = args.cas_warmup_steps_enc   # Steps for warm-up
warmup_steps_dec = args.cas_warmup_steps_dec   # Steps for warm-up
total_steps = args.early_stop_iter if args.early_stop_iter is not None else args.max_iter  # Total training steps
min_lr_enc = args.cas_min_lr_enc         # Minimum LR after decay
min_lr_dec = args.cas_min_lr_dec         # Minimum LR after decay
final_lr_enc = args.cas_final_lr_enc         # Final minimum LR for convergence
final_lr_dec = args.cas_final_lr_dec         # Final minimum LR for convergence
T_0_enc = args.cas_T_0_enc
T_0_dec = args.cas_T_0_dec
T_mult_enc = args.cas_T_mult_enc
T_mult_dec = args.cas_T_mult_dec

cas_scheduler_enc = CustomScheduler(optimizer_encoder, warmup_steps_enc, total_steps, min_lr_enc, learning_rate_enc, final_lr_enc, T_0_enc, T_mult_enc)    
cas_scheduler_dec = CustomScheduler(optimizer_decoder, warmup_steps_dec, total_steps, min_lr_dec, learning_rate_dec, final_lr_dec, T_0_dec, T_mult_dec)

# ================Learnable Uncertainty Weighting (Kendall et al., 2018)===================#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas_sigma_seg = (0.9, 0.99)
betas_sigma_depth = (0.9, 0.999)
weight_decay_sigma_seg = 1e-5
weight_decay_sigma_depth = 1e-5

log_sigma_seg = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)  # For segmentation
log_sigma_depth = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)  # For depth

optimizer_sigma_seg = torch.optim.AdamW([log_sigma_seg], lr=args.lr_sigma_seg, betas=betas_sigma_seg, weight_decay=weight_decay_sigma_seg)
optimizer_sigma_depth = torch.optim.AdamW([log_sigma_depth], lr=args.lr_sigma_depth, betas=betas_sigma_depth, weight_decay=weight_decay_sigma_depth)

#cas_scheduler_sigma_seg = CustomScheduler(optimizer_sigma_seg, args.cas_warmup_steps_sigma_seg, total_steps, args.cas_min_lr_sigma_seg, args.lr_sigma_seg, args.cas_final_lr_sigma_seg, args.cas_T_0_sigma_seg, args.cas_T_mult_sigma_seg)    
#cas_scheduler_sigma_depth = CustomScheduler(optimizer_sigma_depth, args.cas_warmup_steps_sigma_depth, total_steps, args.cas_min_lr_sigma_depth, args.lr_sigma_depth, args.cas_final_lr_sigma_depth, args.cas_T_0_sigma_depth, args.cas_T_mult_sigma_depth)    

# ============ Load Initial / Reload ================#
if args.load_init == 1 and args.load_pretrained == 1:
    train_losses = []
    val_losses = []
    val_epochs = []
    mean_iou_values = []
    rmse_values = []
    start_epoch = None
elif args.load_init == 1 and args.load_pretrained == 0:
    ckpt_path = './weights/ExpKITTI_joint.ckpt'
    train_losses = []
    val_losses = []
    val_epochs = []
    mean_iou_values = []
    rmse_values = []
    start_epoch = None
elif args.load_resume == 1:
    ckpt_path = args.out_chkpt_file

    # Load the .npz file
    loaded_data = np.load(args.load_metric_file)

    # Extract arrays
    train_losses = list(loaded_data["train_losses"])
    val_losses = list(loaded_data["val_losses"])
    val_epochs = list(loaded_data["val_epochs"])
    mean_iou_values = list(loaded_data["mean_iou_values"])
    rmse_values = list(loaded_data["rmse_values"])
else:
    train_losses = []
    val_losses = []
    val_epochs = []
    mean_iou_values = []
    rmse_values = []
    start_epoch = None

if args.load_init == 1 and args.load_pretrained == 1 and args.load_resume == 0:
    # Load encoder weights
    if args.init_chkpt_file_enc is not None:
        encoder_ckpt = torch.load(args.init_chkpt_file_enc)
        encoder_state_dict = encoder_ckpt['state_dict'] if 'state_dict' in encoder_ckpt else encoder_ckpt

        # Normalize key names
        if any(k.startswith("encoder.layer") for k in encoder_state_dict):
            print("Detected SimCLR-style keys with 'encoder.' prefix")
            encoder_state_dict = {
                k.replace("encoder.", ""): v for k, v in encoder_state_dict.items()
            }

        # Only retain encoder keys
        encoder_keys = [k for k in hydranet_model.state_dict() if k.startswith("layer") or k.startswith("final_conv")]
        encoder_state_dict_filtered = {
            k: v for k, v in encoder_state_dict.items() if k in encoder_keys
        }

        # Load into full model (not .extract_encoder())
        load_state_dict(hydranet_model, encoder_state_dict_filtered, strict=False)
        print("Encoder weights loaded from:", args.init_chkpt_file_enc)        

    if args.init_chkpt_file_dec is not None:
        # Load decoder weights
        # Load the full checkpoint state_dict (no 'state_dict' key wrapper)
        decoder_ckpt = torch.load(args.init_chkpt_file_dec)  # path to pretrained_full.autoencoder.pth.tar

        # Get encoder keys from a clean HydraNet instance
        encoder_keys = hydranet_model.extract_encoder(init=False).state_dict().keys()

        # Filter only decoder weights
        decoder_state_dict = {
            k: v for k, v in decoder_ckpt.items()
            if k not in encoder_keys and not k.startswith("segm.")
        }    

        # Load them into the full HydraNet model
        load_state_dict(hydranet_model, decoder_state_dict, strict=False)
        print("Decoder weights loaded from:", args.init_chkpt_file_dec)

        print("Model has {} parameters".format(sum([p.numel() for p in hydranet_model.parameters()])))    

    if args.init_chkpt_file_dec is not None and args.init_chkpt_file_enc is not None:
        #-----------Check if Missing Key-----------#
        loaded_keys = set(list(encoder_state_dict_filtered.keys()) + list(decoder_state_dict.keys()))
        model_keys = set(hydranet_model.state_dict().keys())

        missing_keys = sorted(list(model_keys - loaded_keys))
        unexpected_keys = sorted(list(loaded_keys - model_keys))

        print(f"\n Total model keys: {len(model_keys)}")
        print(f"Missing keys (expected by model but not found in loaded weights): {len(missing_keys)}")
        for k in missing_keys:
            print(f"  - {k}")

        print(f"\nUnexpected keys (found in loaded weights but not used in model): {len(unexpected_keys)}")
        for k in unexpected_keys:
            print(f"  - {k}")    
    elif args.init_chkpt_file_enc is not None:
        #-----------Check if Missing Key-----------#
        loaded_keys = set(list(encoder_state_dict_filtered.keys()))
        model_keys = set(hydranet_model.extract_encoder(init=False).state_dict().keys())
        model_keys = set(k for k in hydranet_model.state_dict().keys() if k.startswith("layer") or k.startswith("final_conv"))

        missing_keys = sorted(list(model_keys - loaded_keys))
        unexpected_keys = sorted(list(loaded_keys - model_keys))

        print(f"\n Total model keys: {len(model_keys)}")
        print(f"Missing keys (expected by model but not found in loaded weights): {len(missing_keys)}")
        for k in missing_keys:
            print(f"  - {k}")

        print(f"\nUnexpected keys (found in loaded weights but not used in model): {len(unexpected_keys)}")
        for k in unexpected_keys:
            print(f"  - {k}")

    elif args.init_chkpt_file_dec is not None:
        #-----------Check if Missing Key-----------#
        loaded_keys = set(list(decoder_state_dict.keys()))
        model_keys = set(
            k for k in hydranet_model.state_dict().keys()
            if not (k.startswith("layer") or k.startswith("final_conv"))
        )

        missing_keys = sorted(list(model_keys - loaded_keys))
        unexpected_keys = sorted(list(loaded_keys - model_keys))

        print(f"\n Total model keys: {len(model_keys)}")
        print(f"Missing keys (expected by model but not found in loaded weights): {len(missing_keys)}")
        for k in missing_keys:
            print(f"  - {k}")

        print(f"\nUnexpected keys (found in loaded weights but not used in model): {len(unexpected_keys)}")
        for k in unexpected_keys:
            print(f"  - {k}")    
    
elif args.load_init == 1 and args.load_pretrained == 0 and args.load_resume == 0: #Load third-party pretrained
    # If the pretrained model has different num_classs in segm head, remove the segmentation head weights (since the number of classes changed)
    start_epoch, _, state_dict = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"], ret_ckpt=False)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "segm" not in k}
    load_state_dict(hydranet_model, filtered_state_dict)
    print(f"Load third-party weights from {ckpt_path}")
    print("Model has {} parameters".format(sum([p.numel() for p in hydranet_model.parameters()])))
elif args.load_resume == 1:
    # If the pretrained model has different num_classs in segm head, remove the segmentation head weights (since the number of classes changed)
    [start_epoch, _, state_dict], checkpoint = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"], ret_ckpt=True)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "segm" not in k}
    load_state_dict(hydranet_model, filtered_state_dict)
    print(f"Load resuming weights from {ckpt_path}")
    print("Model has {} parameters".format(sum([p.numel() for p in hydranet_model.parameters()])))

    # Restore the custom optimizer states
    optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder'])  # Restore encoder optimizer
    optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder'])  # Restore decoder optimizer
    optimizer_sigma_seg.load_state_dict(checkpoint['optimizer_sigma_seg'])  # Restore encoder optimizer
    optimizer_sigma_depth.load_state_dict(checkpoint['optimizer_sigma_depth'])  # Restore decoder optimizer

    # Restore the custom scheduler states
    cas_scheduler_enc.__dict__.update(checkpoint['scheduler_encoder'])
    cas_scheduler_dec.__dict__.update(checkpoint['scheduler_decoder'])    
    #cas_scheduler_sigma_seg.__dict__.update(checkpoint['scheduler_sigma_seg'])
    #cas_scheduler_sigma_depth.__dict__.update(checkpoint['scheduler_sigma_depth'])    
    print(f"Restored Encoder LR: {optimizer_encoder.param_groups[0]['lr']}")
    print(f"Restored Decoder LR: {optimizer_decoder.param_groups[0]['lr']}")
    print(f"Restored sigma_seg LR: {optimizer_sigma_seg.param_groups[0]['lr']}")
    print(f"Restored sigma_depth LR: {optimizer_sigma_depth.param_groups[0]['lr']}")
elif args.load_init == 0 and args.load_pretrained == 0 and args.load_resume == 0:
    print(f"No loading weights, random initialized weights.")

# ================Weight Initialization===================#
if start_epoch is None:
    start_epoch = 0

# Apply weight initialization ONLY to decoder layers
"""
for name, module in hydranet_model.named_modules():
    if not (name.startswith("layer") or name.startswith("final_conv")):  # Exclude encoder layers & final_conv
        HydraNet._initialize_weights(module)  # Apply initialization to non-encoder layers
"""

# Freeze encoder layers (optional)
if args.freeze_enc_epoch > 0:
    for name, param in hydranet_model.named_parameters():
        if name.startswith("layer") or name.startswith("final_conv"):  # Freeze encoder layers and final_conv
            param.requires_grad = False
        else:
            param.requires_grad = True  # Keep decoder layers trainable

for name, param in hydranet_model.named_parameters():
    print(f"{name}: {'Frozen' if not param.requires_grad else 'Trainable'}")

# ================Training Procedure===================#
def train(model, opts, crits, dataloader, train_losses, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    avg_losses = []
    for sample in pbar:
        loss = 0.0
        in_img = sample["image"].float().to(device)
        targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

        #FORWARD
        outputs = model(in_img)

        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            target = target.squeeze(dim=1)  # Ensure (B, H, W)
            target = torch.clamp(target, 0, num_classes - 1)  # Ensure valid class range
            out = F.interpolate(out, size=target.size()[1:], mode="bilinear", align_corners=False)  # Resize logits
            '''
            if crit == crit_segm:
                task_loss = crit(out, target)
                weighted_loss = (1.0 / (2.0 * torch.exp(2 * log_sigma_seg))) * task_loss + log_sigma_seg
            elif crit == crit_depth:
                task_loss = crit(out, target)
                weighted_loss = (1.0 / (2.0 * torch.exp(2 * log_sigma_depth))) * task_loss + log_sigma_depth

            loss += weighted_loss
            '''
            loss += loss_coeff * crit(out, target)  # Compute loss            

        # BACKWARD
        loss.backward()
        avg_losses.append(loss.item())

        #if grad_norm > 0.0:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()

        log_sigma_seg.data.clamp_(min=-3, max=3)
        log_sigma_depth.data.clamp_(min=-3, max=3)
        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )

        for opt in opts:
            opt.zero_grad()
    train_losses.append(np.mean(avg_losses))
    

def validate(model, metrics, dataloader, val_losses, crits, loss_coeffs, epoch_num=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    avg_losses = []
    loss_seg_meter = AverageMeter()
    loss_depth_meter = AverageMeter()
    loss_total_meter = AverageMeter()
    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

            # Forward
            outputs = model(input)
            #outputs = make_list(outputs)
            loss = 0.0
            for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
                target = target.squeeze(dim=1)  # Ensure (B, H, W)
                target = torch.clamp(target, 0, num_classes - 1)  # Ensure valid class range
                out = F.interpolate(out, size=target.size()[1:], mode="bilinear", align_corners=False)  # Resize logits
                if crit == crit_segm:
                    task_loss = crit(out, target)
                    loss_seg_meter.update(task_loss.item())
                elif crit == crit_depth:
                    task_loss = crit(out, target)
                    loss_depth_meter.update(task_loss.item())

                loss += loss_coeff*task_loss
                #loss += loss_coeff * crit(out, target)  # Compute loss            
            avg_losses.append(loss.item())
            loss_total_meter.update(loss.item())

            # Metric
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(out, size=target.shape[2:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1).cpu().numpy(),
                    target.squeeze(dim=1).cpu().numpy()
                )
            pbar.set_description(get_val(metrics)[1])
        print(f"Validation at Epoch {epoch_num}, Avg. Segm Loss: {loss_seg_meter.avg:.4f} | Avg. Depth Loss: {loss_depth_meter.avg:.4f} | Avg. Task Loss: {loss_total_meter.avg:.4f}")

    val_losses.append(np.mean(avg_losses))
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals

#==================Main Loop=================#
loss_coeffs = (0.5, 0.5)

print(f"start_epoch = {start_epoch}")
frozen_set = False
for i in range(start_epoch, n_epochs):
    cas_scheduler_enc.step(i)
    cas_scheduler_dec.step(i)
    #cas_scheduler_sigma_seg.step(i)
    #cas_scheduler_sigma_depth.step(i)

    # === Conditional Encoder Freezing ===
    if args.freeze_enc_epoch > 0 and frozen_set == False:
        if i < args.freeze_enc_epoch:
            for name, param in hydranet_model.named_parameters():
                if name.startswith("layer"):  # Freeze encoder
                    param.requires_grad = False
        else:
            for name, param in hydranet_model.named_parameters():
                param.requires_grad = True  # Unfreeze all
            frozen_set = True

    # Optional: print info
    if i in [0, 199, 200, 300, 500, 501]:
        print(f"[Epoch {i}] Encoder frozen? {not hydranet_model.layer1[0].weight.requires_grad}")


    if i % args.show_lr_freq_epoch == 0:
        current_lr_enc = optimizer_encoder.param_groups[0]['lr']
        current_lr_dec = optimizer_decoder.param_groups[0]['lr']
        current_lr_sigma_seg = optimizer_sigma_seg.param_groups[0]['lr']
        current_lr_sigma_depth = optimizer_sigma_depth.param_groups[0]['lr']
        print(f"Epoch {i} | Encoder Learning Rate: {current_lr_enc:.6f}, Decoder Learning Rate: {current_lr_dec:.6f}, Sigma Seg Learning Rate: {current_lr_sigma_seg:.6f}, Sigma Depth Learning Rate: {current_lr_sigma_depth:.6f}")    
    
    train(model=hydranet_model, opts=[optimizer_encoder, optimizer_decoder, optimizer_sigma_seg, optimizer_sigma_depth], crits=[crit_segm, crit_depth], dataloader=trainloader, train_losses=train_losses, loss_coeffs=loss_coeffs, grad_norm=0.0)

    if i%args.val_every == 0:
        metrics = [MeanIoU(num_classes), RMSE(ignore_val=ignore_depth)]

        with torch.no_grad():
            vals = validate(model=hydranet_model, metrics=metrics, dataloader=valloader, val_losses=val_losses, crits=[crit_segm, crit_depth], loss_coeffs=loss_coeffs, epoch_num=i)
            val_epochs.append(i)

            # Unpack validation metrics
            mean_iou, rmse = vals  # Assuming validate() returns (mIoU, RMSE)
            mean_iou_values.append(mean_iou)
            rmse_values.append(rmse)            

            saver.maybe_save(new_val=[mean_iou], dict_to_save={"state_dict": hydranet_model.state_dict(), "epoch": i})
            #saver.maybe_save(new_val=vals, dict_to_save={"state_dict": hydranet_model.state_dict(), "epoch": i})

    if i%args.save_freq == 0 and i > 0:
        checkpoint = {
            'state_dict': hydranet_model.state_dict(),  # Model weights
            'epoch': i,  # Save the current epoch
            'optimizer_encoder': optimizer_encoder.state_dict(),  # Save encoder optimizer state
            'optimizer_decoder': optimizer_decoder.state_dict(),  # Save decoder optimizer state
            'scheduler_encoder': cas_scheduler_enc.__dict__,  # Save encoder scheduler state
            'scheduler_decoder': cas_scheduler_dec.__dict__,  # Save decoder scheduler state
            'optimizer_sigma_seg': optimizer_sigma_seg.state_dict(),  # Save sigma_seg optimizer state
            'optimizer_sigma_depth': optimizer_sigma_depth.state_dict(),  # Save sigma_depth optimizer state
            #'scheduler_sigma_seg': cas_scheduler_sigma_seg.__dict__,  # Save sigma_seg scheduler state
            #'scheduler_sigma_depth': cas_scheduler_sigma_depth.__dict__,  # Save sigma_depth scheduler state
        }        
        base_chkpt_file_name = os.path.basename(args.out_chkpt_file)
        base_metric_file_name = os.path.basename(args.load_metric_file)
        save_ckpt_file = os.path.join(args.output_dir, base_chkpt_file_name)
        save_metric_file = os.path.join(args.output_dir, base_metric_file_name)
        torch.save(checkpoint, save_ckpt_file)
        np.savez(args.load_metric_file,
                train_losses=train_losses,
                val_losses=val_losses,
                val_epochs=val_epochs,
                mean_iou_values=mean_iou_values,
                rmse_values=rmse_values)
        # Open the file in append mode
        with open(f"{os.path.join(args.output_dir, 'learning_rate_log.txt')}", "a") as f:
            current_lr_enc = optimizer_encoder.param_groups[0]['lr']
            current_lr_dec = optimizer_decoder.param_groups[0]['lr']
            current_lr_sigma_seg = optimizer_sigma_seg.param_groups[0]['lr']
            current_lr_sigma_depth = optimizer_sigma_depth.param_groups[0]['lr']
            f.write(f"Epoch {i}: Encoder LR = {current_lr_enc}, Decoder LR = {current_lr_dec}, Sigma Seg Learning Rate: {current_lr_sigma_seg:.6f}, Sigma Depth Learning Rate: {current_lr_sigma_depth:.6f}\n")



# === Save Train Loss & Validation Loss Plot ===
epochs = list(range(1, len(train_losses) + 1))

# Interpolate validation losses to match every epoch
val_interp = np.interp(epochs, val_epochs, val_losses)  

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
plt.plot(val_epochs, val_losses, label="Validation Loss (Actual)", color="red", marker="s")
plt.plot(epochs, val_interp, '--', color="red", alpha=0.5, label="Validation Loss (Interpolated)")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Epochs")
plt.legend()
plt.grid(True)

# Save plot
save_loss_file = os.path.join(args.output_dir, "training_validation_loss.png")
plt.savefig(f"{save_loss_file}", dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# === Save Mean IoU & RMSE Plot ===
plt.figure(figsize=(8, 6))

# Plot Mean IoU
plt.plot(val_epochs, mean_iou_values, label="Mean IoU", color="green", marker="o")

# Plot RMSE
plt.plot(val_epochs, rmse_values, label="RMSE", color="purple", marker="s")

plt.xlabel("Epochs")
plt.ylabel("Metric Value")
plt.title("Validation Metrics (Mean IoU & RMSE) vs Epochs")
plt.legend()
plt.grid(True)

# Save plot
save_metric_png = os.path.join(args.output_dir, "miou_rmse_metrics.png")
plt.savefig(f"{save_metric_png}", dpi=300, bbox_inches='tight')
plt.close()
