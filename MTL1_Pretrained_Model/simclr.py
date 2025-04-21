import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from hydranet import HydraNet

class PixelwiseProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 512, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.normalize(self.conv2(x), dim=1)  # Normalize across channel dim
        return x

# Define SimCLR Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=1)  # SimCLR benefits from normalized projections.

# SimCLR Encoder using HydraNet's encoder
class SimCLR(nn.Module):
    def __init__(self, num_classes=40, num_tasks=2):
        super().__init__()

        # Use HydraNet’s encoder
        full_model = HydraNet(num_classes=num_classes, num_tasks=num_tasks)
        self.encoder = nn.Module()  # Custom container
        self.encoder.layer1 = full_model.layer1
        self.encoder.layer2 = full_model.layer2
        self.encoder.layer3 = full_model.layer3
        self.encoder.layer4 = full_model.layer4
        self.encoder.layer5 = full_model.layer5
        self.encoder.layer6 = full_model.layer6
        self.encoder.layer7 = full_model.layer7
        self.encoder.layer8 = full_model.layer8        

        #self.encoder = HydraNet(num_classes=num_classes, num_tasks=num_tasks).extract_encoder(init=True)
        #self.pool = nn.AdaptiveAvgPool2d((1,1))  # Global pooling
        #self.projection = ProjectionHead(1280, 128)  # Projection head
        #self.projection = ProjectionHead(320, 128)  # Projection head
        #self.projection = ProjectionHead(320, 256)  # Projection head
        self.projection = PixelwiseProjectionHead(320, 256)

    def forward(self, x):
        #feat_map = self.encoder(x)  # (B, 320, H, W)
        # Pass through each encoder layer explicitly
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.layer5(x)
        x = self.encoder.layer6(x)
        x = self.encoder.layer7(x)
        feat_map = self.encoder.layer8(x)  # (B, 320, H, W)

        z1 = self.projection(feat_map)  # (B, 256, H, W)
        # Multi-scale feature maps (1x, 1/2x, 1/4x)
        z_half = F.interpolate(z1, scale_factor=0.5, mode='bilinear', align_corners=False)
        z_quarter = F.interpolate(z1, scale_factor=0.25, mode='bilinear', align_corners=False)
        return [z1, z_half, z_quarter]  # list of (B, D, H', W')

def info_nce_loss(z_i, z_j, temperature=0.07):
    """
    Computes InfoNCE loss.
    z_i: [batch (N), feature_dim (d)] - images through augmented transform1
    z_j: [batch (N), feature_dim (d)] - same images through augmented transform2
    - temperature: Scaling factor for logits.
    """

    batch_size = z_i.shape[0]

    # Concatenate both views
    z = torch.cat([z_i, z_j], dim=0)  # (2N, 128)
    z = F.normalize(z, p=2, dim=1)  # Normalize embeddings

    # Compute similarity matrix
    similarity_matrix = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Mask out self-comparisons
    mask = torch.eye(2 * batch_size, dtype=torch.bool).cuda()
    similarity_matrix.masked_fill_(mask, 1e-15)  # Use -1e9 instead of -inf

    # Compute labels
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).cuda()

    # Compute **two** cross-entropy losses like CLIP
    loss_i = F.cross_entropy(similarity_matrix, labels, reduction="mean")
    loss_j = F.cross_entropy(similarity_matrix.T, labels, reduction="mean")

    # Final symmetric loss
    loss = (loss_i + loss_j) / 2  # Symmetric loss

    return loss    

def dense_contrastive_loss(z1, z2, temperature=0.07):
    """
    z1, z2: (B, D, H, W) feature maps from two augmented views
    """
    B, D, H, W = z1.shape
    z1 = z1.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
    z2 = z2.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).long().to(z1.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)

    return (loss_i + loss_j) / 2

def multi_scale_dense_contrastive_loss(z1_list, z2_list, temperature=0.07, weights=None):
    """
    z1_list, z2_list: lists of multi-scale feature maps [(B, D, H, W), ...]
    weights: list of weights per scale, default equal
    """
    if weights is None:
        weights = [1.0 / len(z1_list)] * len(z1_list)  # Equal weights

    total_loss = 0.0
    for z1, z2, w in zip(z1_list, z2_list, weights):
        B, D, H, W = z1.shape
        z1_flat = z1.permute(0, 2, 3, 1).reshape(-1, D)
        z2_flat = z2.permute(0, 2, 3, 1).reshape(-1, D)

        z1_flat = F.normalize(z1_flat, dim=1)
        z2_flat = F.normalize(z2_flat, dim=1)

        logits = torch.mm(z1_flat, z2_flat.t()) / temperature
        labels = torch.arange(z1_flat.size(0)).long().to(z1_flat.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.t(), labels)

        total_loss += w * (loss_i + loss_j) / 2

    return total_loss

