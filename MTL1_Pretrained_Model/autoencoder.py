import torch
import torch.nn as nn
import torch.nn.functional as F
from hydranet import HydraNet

class HydraAutoencoder(nn.Module):
    def __init__(self, num_classes=40, num_tasks=2):
        super().__init__()
        self.hydranet = HydraNet(num_classes=num_classes, num_tasks=num_tasks)

        # Replace task heads with a shared RGB reconstruction head
        self.hydranet.pre_depth = nn.Identity()
        self.hydranet.depth = nn.Identity()
        self.hydranet.pre_segm = nn.Identity()
        self.hydranet.segm = nn.Conv2d(256, 3, kernel_size=1)  # RGB recon head replaces segm

        if self.hydranet.num_tasks == 3:
            self.hydranet.pre_normal = nn.Identity()
            self.hydranet.normal = nn.Identity()

    def forward(self, x):
        recon, _ = self.hydranet(x)  # Output: recon from segm head, discard depth
        return recon