import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from hydranet import HydraNet

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
        return x

# SimCLR Encoder using HydraNet's encoder
class SimCLR(nn.Module):
    def __init__(self, num_classes=40, num_tasks=2):
        super().__init__()

        # Use HydraNet’s encoder
        self.encoder = HydraNet(num_classes=num_classes, num_tasks=num_tasks).extract_encoder()
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # Global pooling
        #self.projection = ProjectionHead(1280, 128)  # Projection head
        self.projection = ProjectionHead(320, 128)  # Projection head

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).squeeze()  # Global pooling
        x = self.projection(x)  # Projection to lower-dimensional space
        return x

'''
def info_nce_loss(z_i, z_j, temperature=0.5):
    """
    Computes InfoNCE loss.
    z_i: [batch (N), feature_dim (d)] - images through augmented transform1
    z_j: [batch (N), feature_dim (d)] - same images through augmented transform2
    - temperature: Scaling factor for logits.
    """    
    print(f"z_i.shape = {z_i.shape}")
    input()
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2N, 128) Concatenate both views
    z = F.normalize(z, p=2, dim=1)  # Normalize embeddings

    # Compute similarity matrix
    similarity_matrix = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Mask out self-comparisons
    mask = torch.eye(2 * batch_size, dtype=torch.bool).cuda()
    similarity_matrix = similarity_matrix.masked_fill(mask, 1e-15)
    #similarity_matrix = similarity_matrix.masked_fill(mask, -float("inf"))

    # Compute log-softmax over all pairs
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).cuda()
    loss = F.cross_entropy(similarity_matrix, labels)  # Contrastive loss

    return loss
'''

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

