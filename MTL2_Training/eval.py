import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os

from hydranet import HydraNet
from dataset import HydranetDataset
from utils import MeanIoU, RMSE, Normalise, ToTensor

# === Argument Parser ===
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate HydraNet on test set")
    parser.add_argument("--checkpoint_file", type=str, required=True,
                        help="Path to checkpoint file (.pth.tar)")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs",
                        help="Directory to save the results")
    return parser.parse_args()

# === Evaluation Script ===
def main():
    args = get_args()

    # Configs
    test_file = './test_list_depth.txt'
    num_classes = 40
    num_tasks = 2
    batch_size = 4
    num_workers = 4
    ignore_depth = 0

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    img_scale = 1.0 / 255
    depth_scale = 5000.0
    normalise_params = [img_scale, img_mean.reshape((1, 1, 3)), img_std.reshape((1, 1, 3)), depth_scale]

    transform = transforms.Compose([Normalise(*normalise_params), ToTensor()])
    test_dataset = HydranetDataset(test_file, transform)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydraNet(num_classes=num_classes, num_tasks=num_tasks).to(device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Metrics
    metrics = [MeanIoU(num_classes), RMSE(ignore_val=ignore_depth)]
    for metric in metrics:
        metric.reset()

    # Evaluate
    with torch.no_grad():
        for sample in tqdm(testloader, desc="Evaluating"):
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in testloader.dataset.masks_names]
            outputs = model(input)

            for out, target, metric in zip(outputs, targets, metrics):
                target = target.squeeze(dim=1)
                out = F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                metric.update(out.squeeze(dim=1).cpu().numpy(), target.cpu().numpy())

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "final_test_metrics.txt")
    with open(report_path, 'w') as f:
        for m in metrics:
            result = f"{m.name}: {m.val():.6f}"
            print(result)
            f.write(result + "\n")

    print(f"\n Results saved to {report_path}")

if __name__ == "__main__":
    main()
