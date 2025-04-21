from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os

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

# Dataset for Self-Supervised Learning (No Labels)
class NYUDv2Autoencoder(Dataset):
    def __init__(self, data_path, transform, img_mean, img_std):
        self.image_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
        self.transform = transform
        #self.normalize = transform.Normalize(mean=img_mean, std=img_std)
        self.normalize_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])  # Only tensor + normalize        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.normalize_target(image)          # nprmalized targe
        input_view = self.transform(image)             # already normalized
        return input_view, target        