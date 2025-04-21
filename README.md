
# HydraNet: Multi-Task Learning with Pretrained Encoder and Decoder

This project implements a multi-task learning pipeline on the NYUv2 dataset using:

- **Encoder pretrained via multi-scale dense pixel-wise SimCLR contrastive learning**
- **Decoder pretrained via RGB autoencoder reconstruction**
- **Learnable task uncertainty loss weighting (Kendall et al., 2018)**
- **Cosine annealing learning rate scheduling with warmup**
- **MobileNetV2-based encoder and lightweight RefineNet decoder**

---

## 🧠 Features

- Lightweight MobileNetV2 encoder
- Efficient RefineNet-style decoder
- Semantic segmentation and depth estimation
- Optional encoder freezing for warm-up
- Plots for loss, mIoU, and RMSE

---

## 📁 Directory Overview

```
MTL1_Pretrained_Model
├── autoencoder.py            # HydraAutoencoder for decoder pretraining
├── simclr.py                 # SimCLR encoder pretraining
├── hydranet.py               # HydraNet model architecture
├── scheduler.py              # Learning rate schedulers
├── dataset.py                # Dataset handling (NYUv2)
├── utils.py                  # Losses, metrics, and visualizations
├── model_helpers.py          # Model loading, saving utilities
├── train_list_depth.txt      # Training split
├── val_list_depth.txt        # Validation split
├── test_list_depth.txt       # Testing split
├── cmap_nyud.npy             # Color map for segmentation
├── req.txt                   # Python environment dependencies
├── pretrained_autocoder_main.py # Script for autoencoder training
├── args.json                 # Arguments for reproducibility
├── baseline_result/          # Stores pretrained checkpoints & metric logs
└── weights/                  # Pretrained weights
```

```
MTL2_Training
```
├── main.py                  # Fine-tuning entry point
├── hydranet.py              # Full HydraNet model
├── dataset.py               # Dataset loading and transforms
├── scheduler.py             # Cosine annealing + warmup
├── utils.py                 # Metric functions (mIoU, RMSE)
├── model_helpers.py         # Checkpoint management
├── merge_shuff.py           # Data splitting utility
├── args.json                # Training arguments
├── weights/                 # Contains pretrained encoder & decoder
├── baseline_result/         # Logs: training_loss, miou, rmse
├── cmap_nyud.npy            # Color palette
├── train_list_depth.txt     # Training split
├── val_list_depth.txt       # Validation split
├── test_list_depth.txt      # Testing split
├── download_dataset.sh      # Auto-download NYUv2 dataset
```
---

## 🛠️ Setup

**Requirements:**

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- numpy, matplotlib, Pillow, tqdm

```bash
conda env create -f environment.yml
```

---

## 🧪 Pretraining

### 1. Pretrain Encoder (SimCLR)

```bash
cd ./MTL1_Pretrained_Model
CUDA_VISIBLE_DEVICES=0 python ./main.py --load_init 1 --load_resume 0 --out_chkpt_file ./pretrained_all.320_256.multiscale_dense.pth.tar --out_encoder_chkpt_file ./pretrained_hydranet_encoder.320_256.multiscale_dense.pth.tar --out_pretrain_loss_file ./pretrained_loss_file.320_256.multiscale_dense.npz  --out_pretrain_loss_figfile_name ./pretrained_loss_fig.320_256.multiscale_dense.png  --lr_enc 6e-4 --max_iter 5000 --cas_warmup_steps_enc 300 --cas_T_0_enc 8000 --cas_min_lr_enc 2e-4 --cas_final_lr_enc 1e-5
```

### 2. Pretrain Decoder (Autoencoder)

```bash
cd ./MTL1_Pretrained_Model
CUDA_VISIBLE_DEVICES=4 python ./pretrain_autoencoder_main.py --lr_enc 3e-4 --cas_min_lr_enc 6e-5 --cas_final_lr_enc 1e-6 --cas_T_0_enc 5000 --max_iter 5000 --cas_warmup_steps_enc 200 --batch_size 32 --save_freq 50 --show_std_freq_epoch 50 --show_lr_freq_epoch 50 --load_init 1 --load_resume 0 --out_full_autoencoder_chkpt_file pretrained_full.autoencoder.luc.pth.tar
```

---

## Multi-Task Training

```bash
cd ./MTL2_Training
UDA_VISIBLE_DEVICES=1 python ./main.py --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./pretrained_full.autoencoder.luc.pth.tar --init_chkpt_file_dec ./pretrained_full.autoencoder.luc.pth.tar --out_chkpt_file ./checkpoint.resume_baseline.pth.tar --max_iter 5001 --cas_T_0_enc 5001 --cas_T_0_dec 5001 --lr_enc 1e-3 --cas_min_lr_enc 9e-5 --cas_final_lr_enc 1e-6 --lr_dec 5e-4 --cas_min_lr_dec 5e-5 --cas_final_lr_dec 1e-6 --cas_warmup_steps_enc 150 --cas_warmup_steps_dec 300 --val_every 100 --save_freq 500 --show_lr_freq_epoch 500 --freeze_enc_epoch 500
```

---

## Visualization

Under folder `baseline_result`:

- `training_validation_loss.png`: training/validation loss curves
- `miou_rmse_metrics.png`: validation mIoU and RMSE trends

---

## 🧪 Evaluation Metrics

| Metric | Description        |
|--------|--------------------|
| mIoU   | Mean Intersection over Union (segmentation) |
| RMSE   | Root Mean Square Error (depth)              |

---

## 📌 Notes

- Segmentation: `num_classes=40`
- You can freeze the encoder for the first 200 epochs via:

```python
if epoch < 200:
    freeze_encoder()
```

Adjust based on loss curves or encoder drift.

---

## 🧾 Citation

```bibtex
@inproceedings{kendall2018multi,
  title={Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle={CVPR},
  year={2018}
}
```
