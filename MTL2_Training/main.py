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

#clears Python-level unreferenced objects.
gc.collect()  # Collects unused Python objects
torch.cuda.empty_cache()  # Releases GPU memory

#Argument
def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--lr_enc', default=8e-3, type=float)
    parser.add_argument('--lr_dec', default=8e-4, type=float)
    parser.add_argument('--max_iter', default=5000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_freq', default=100, type=int)    
    parser.add_argument('--early_stop_iter', default=None, type=int)           
    parser.add_argument('--cas_warmup_steps_enc', default=20, type=int)
    parser.add_argument('--cas_warmup_steps_dec', default=30, type=int)
    parser.add_argument('--cas_min_lr_enc', default=1e-3, type=float)
    parser.add_argument('--cas_min_lr_dec', default=1e-4, type=float)
    parser.add_argument('--cas_final_lr_enc', default=8e-4, type=float)
    parser.add_argument('--cas_final_lr_dec', default=8e-5, type=float)
    parser.add_argument('--cas_T_0_enc', default=5000, type=int)
    parser.add_argument('--cas_T_0_dec', default=5000, type=int)
    parser.add_argument('--cas_T_mult_enc', default=1, type=int)
    parser.add_argument('--cas_T_mult_dec', default=1, type=int)
    parser.add_argument('--show_lr_freq_epoch', default=100, type=int)
    parser.add_argument('--load_init', default=0, type=int)
    parser.add_argument('--load_resume', default=1, type=int)
    parser.add_argument('--out_chkpt_file', default='./checkpoint.pth.tar', type=str)
    parser.add_argument('--train_split_ratio', default=0.8, type=float)
    parser.add_argument('--shuffle_dataset', default=0, type=int)
    return parser    


parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
args = parser.parse_args()

depth = sorted(glob.glob("./nyud/depth/*.png"))
seg = sorted(glob.glob("./nyud/masks/*.png"))
images = sorted(glob.glob("./nyud/rgb/*.png"))

#Dataset Definintion
class HydranetDataset(Dataset):

    def __init__(self, data_file, transform=None):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [x.decode("utf-8").strip("\n").split("\t") for x in datalist]
        self.root_dir = "nyud"
        self.transform = transform
        self.masks_names = ("segm", "depth")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        abs_paths = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]] # Will output list of nyud/*/00000.png
        sample = {}
        sample["image"] = np.array(Image.open(abs_paths[0]))

        for mask_name, mask_path in zip(self.masks_names, abs_paths[1:]):
            sample[f"{mask_name}"] = np.array(Image.open(mask_path))

        if self.transform:
            sample["names"] = self.masks_names
            sample = self.transform(sample)
            # the names key can be removed by the transformation
            if "names" in sample:
                del sample["names"]
        return sample


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

transform_train = transforms.Compose([RandomMirror(), RandomCrop(crop_size)] + transform_common)
transform_val = transforms.Compose(transform_common)
train_batch_size = 4
val_batch_size = 4
train_file = "./train_list_depth.txt"
val_file = "./val_list_depth.txt"
CMAP = np.load('cmap_nyud.npy')
if args.shuffle_dataset == 1:
    shuffle_and_split_data(train_file=train_file, val_file=val_file, train_split_ratio=args.train_split_ratio)

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

#Model Architecture
def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """3x3 Convolution: Depthwise:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    "1x1 Convolution: Pointwise"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)

def batchnorm(num_features):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__() # Python 3
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1) # Boolean/Condition
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        #residual = x
        out = self.output(x)
        if self.residual:
            return (out + x)#+residual
        else:
            return out


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

class CRPBlock(nn.Module):
    """CRP definition"""
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__()
        for i in range(n_stages):
            if groups:
                setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups= in_planes if i==0 else out_planes))
            else:
                setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups=1))

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x

class HydraNet(nn.Module):
    """Net Definition"""
    def __init__(self, in_channels=32, num_classes=6, num_tasks=2, agg_size=256, n_crp=4):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
    ## Encoder-MobileNetV2 ##
        self.mobilenet_config = [[1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                    ]
        self.in_channels = in_channels # number of input channels
        self.num_layers = len(self.mobilenet_config)
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)

        c_layer = 2
        for t,c,n,s in (self.mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_channels = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers)) # setattr(object, name, value)
            c_layer += 1

        ## Decoder-Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=False)
        self.crp3 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=False)
        self.crp2 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=False)
        self.crp1 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=True)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)# Define the Purple Pre-Head for Depth
        self.depth = conv3x3(256, 1, bias=True)# Define the Final layer of Depth
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)#: Call the Purple Pre-Head for Segm
        self.segm = conv3x3(256, self.num_classes, bias=True)#: Define the Final layer of Segmentation
        self.relu = nn.ReLU6(inplace=True)#: Define a RELU 6 Operation

        if self.num_tasks == 3:
            # Create a Normal Head
            self.pre_normal = conv1x1(256, 256, groups=256, bias=False)
            self.normal = conv3x3(256, 3, bias=True)

    def forward(self, x):
        # MOBILENET V2
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32

    # LIGHT-WEIGHT REFINENET
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # HEADS 
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        if self.num_tasks == 3:
            out_n = self.pre_normal(l3)
            out_n = self.relu(out_n)
            out_n = self.normal(out_n)
            return out_segm, out_d, out_n
        else:
            return out_segm, out_d

    def _make_crp(self, in_planes, out_planes, stages, groups=False):
        layers = [CRPBlock(in_planes, out_planes,stages, groups=groups)]
        return nn.Sequential(*layers)

#Loss Definition
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

#Model Instance
num_classes, num_tasks = 40, 2
hydranet_model = HydraNet(num_classes=num_classes, num_tasks=num_tasks)
#hydranet = nn.DataParallel(nn.Sequential(encoder, decoder).cuda()) # Use .cpu() if you prefer a slow death

#load pre-trained weight
#ckpt = torch.load('../../weights/ExpKITTI_joint.ckpt')
#hydranet_model.load_state_dict(ckpt['state_dict'])
if args.load_init == 1:
    ckpt_path = './weights/ExpKITTI_joint.ckpt'
    train_losses = []
    val_losses = []
    val_epochs = []
    mean_iou_values = []
    rmse_values = []
elif args.load_resume == 1:
    ckpt_path = args.out_chkpt_file

    # Load the .npz file
    loaded_data = np.load("metrics_loss_data.npz")

    # Extract arrays
    train_losses = list(loaded_data["train_losses"])
    val_losses = list(loaded_data["val_losses"])
    val_epochs = list(loaded_data["val_epochs"])
    mean_iou_values = list(loaded_data["mean_iou_values"])
    rmse_values = list(loaded_data["rmse_values"])

start_epoch, _, state_dict = saver.maybe_load(ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"],)
# Remove the segmentation head weights (since the number of classes changed)
filtered_state_dict = {k: v for k, v in state_dict.items() if "segm" not in k}
# Load only matching parameters
#hydranet_model.load_state_dict(filtered_state_dict, strict=False)
load_state_dict(hydranet_model, filtered_state_dict)
print("Model has {} parameters".format(sum([p.numel() for p in hydranet_model.parameters()])))

if torch.cuda.is_available():
    _ = hydranet_model.cuda()

if start_epoch is None:
    start_epoch = 0

# Freeze only the encoder part
for name, param in hydranet_model.named_parameters():
    if "layer" in name:  # Assuming encoder layers follow this naming convention
        param.requires_grad = False  # Freeze encoder
    else:
        param.requires_grad = True   # Keep decoder trainable

print("All encoder layers are frozen, while decoder layers remain trainable.")
for name, param in hydranet_model.named_parameters():
    print(f"{name}: {'Frozen' if not param.requires_grad else 'Trainable'}")
'''
# Freeze all layers except pre_depth, depth, pre_segm, segm
for name, param in hydranet_model.named_parameters():
    if not any(key in name for key in ["pre_depth", "depth", "pre_segm", "segm"]):
        param.requires_grad = False  # Freeze layers
    else:
        param.requires_grad = True   # Keep these layers trainable

print("All layers frozen except pre_depth, depth, pre_segm, and segm.")
'''
#################### Optimizer #########################
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
weight_decay_encoder = 1e-5
weight_decay_decoder = 1e-5
'''Adam'''
# Extract encoder parameters (MobileNetV2)
# Extract decoder parameters (RefineNet)
encoder_params, decoder_params = get_params(hydranet_model)
optimizer_encoder = torch.optim.Adam(encoder_params, lr=lr_encoder, betas=betas_encoder,weight_decay=weight_decay_encoder)
optimizer_decoder = torch.optim.Adam(decoder_params, lr=lr_decoder, betas=betas_decoder,weight_decay=weight_decay_decoder)

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
            loss += loss_coeff * crit(out, target)  # Compute loss            

            '''
            print(f"Prediction shape: {out.shape}, Target shape: {target.shape}")
            print(f"Target dtype: {target.dtype}, Unique values: {target.unique()}")
            loss += loss_coeff * crit(
                F.interpolate(
                    out, size=target.size()[2:], mode="bilinear", align_corners=False
                ).squeeze(dim=1),
                target.squeeze(dim=1),
            )
            '''

        # BACKWARD
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        avg_losses.append(loss.item())

        #if grad_norm > 0.0:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()

        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )
    train_losses.append(np.mean(avg_losses))
    

def validate(model, metrics, dataloader, val_losses, crits, loss_coeffs):
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
    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

            # Forward
            outputs = model(input)
            #outputs = make_list(outputs)
            loss = 0
            for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
                target = target.squeeze(dim=1)  # Ensure (B, H, W)
                target = torch.clamp(target, 0, num_classes - 1)  # Ensure valid class range
                out = F.interpolate(out, size=target.size()[1:], mode="bilinear", align_corners=False)  # Resize logits
                loss += loss_coeff * crit(out, target)  # Compute loss            
            avg_losses.append(loss.item())

            # Metric
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(out, size=target.shape[2:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1).cpu().numpy(),
                    target.squeeze(dim=1).cpu().numpy()
                )
            pbar.set_description(get_val(metrics)[1])
    val_losses.append(np.mean(avg_losses))

    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals

#==================Main Loop=================#
val_every = 100
loss_coeffs = (0.5, 0.5)

print(f"start_epoch = {start_epoch}")
for i in range(start_epoch, n_epochs+1):
    cas_scheduler_enc.step(i)
    cas_scheduler_dec.step(i)
    if i % args.show_lr_freq_epoch == 0:
        current_lr_enc = optimizer_encoder.param_groups[0]['lr']
        current_lr_dec = optimizer_decoder.param_groups[0]['lr']
        print(f"Epoch {i} | Encoder Learning Rate: {current_lr_enc:.6f}, Decoder Learning Rate: {current_lr_dec:.6f}")    
    
    train(model=hydranet_model, opts=[optimizer_encoder, optimizer_decoder], crits=[crit_segm, crit_depth], dataloader=trainloader, train_losses=train_losses, loss_coeffs=loss_coeffs, grad_norm=0.0)

    if i%val_every == 0:
        metrics = [MeanIoU(num_classes), RMSE(ignore_val=ignore_depth)]

        with torch.no_grad():
            vals = validate(model=hydranet_model, metrics=metrics, dataloader=valloader, val_losses=val_losses, crits=[crit_segm, crit_depth], loss_coeffs=loss_coeffs)
            val_epochs.append(i)

            # Unpack validation metrics
            mean_iou, rmse = vals  # Assuming validate() returns (mIoU, RMSE)
            mean_iou_values.append(mean_iou)
            rmse_values.append(rmse)            

        saver.maybe_save(new_val=vals, dict_to_save={"state_dict": hydranet_model.state_dict(), "epoch": i})

    if i%args.save_freq == 0 and i > 0:
        checkpoint = {
        'state_dict': hydranet_model.state_dict(),
        'epoch': i,
        }
        torch.save(checkpoint, args.out_chkpt_file)
        np.savez("metrics_loss_data.npz",
                train_losses=train_losses,
                val_losses=val_losses,
                val_epochs=val_epochs,
                mean_iou_values=mean_iou_values,
                rmse_values=rmse_values)
        # Open the file in append mode
        with open("learning_rate_log.txt", "a") as f:
            current_lr_enc = optimizer_encoder.param_groups[0]['lr']
            current_lr_dec = optimizer_decoder.param_groups[0]['lr']
            f.write(f"Epoch {i}: Encoder LR = {current_lr_enc}, Decoder LR = {current_lr_dec}\n")



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
plt.savefig("training_validation_loss.png", dpi=300, bbox_inches='tight')
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
plt.savefig("miou_rmse_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
