"""
DisentangleCascade — main entry point.
Run:  python main.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ISICDataset, get_transforms, prepare_isic_data
from src.model import DisentangleCascade
from src.train import train_model, evaluate, build_weighted_criterion


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET_PATH     = '/kaggle/input/isic-2019-skin-lesion-images-for-classification/'
CSV_PATH         = DATASET_PATH + 'ISIC_2019_Training_GroundTruth.csv'
SELECTED_CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC']

IMAGE_SIZE  = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR          = 1e-4
HSIC_WEIGHT = 0.1
SAVE_PATH   = 'disentangle_cascade_best.pth'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("DisentangleCascade  —  ISIC 2019  (8 classes)")
    print("=" * 60)

    # 1. Data
    (train_paths, val_paths, test_paths,
     train_labels, val_labels, test_labels,
     class_names, label_to_idx) = prepare_isic_data(
        DATASET_PATH, CSV_PATH, selected_classes=SELECTED_CLASSES)

    num_classes = len(class_names)

    train_ds = ISICDataset(train_paths, train_labels, get_transforms(IMAGE_SIZE, 'train'))
    val_ds   = ISICDataset(val_paths,   val_labels,   get_transforms(IMAGE_SIZE, 'val'))
    test_ds  = ISICDataset(test_paths,  test_labels,  get_transforms(IMAGE_SIZE, 'val'))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model
    model = DisentangleCascade(num_classes=num_classes, backbone='resnet50', pretrained=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Train
    history = train_model(
        model, train_loader, val_loader, train_labels,
        num_classes=num_classes, num_epochs=NUM_EPOCHS,
        lr=LR, hsic_weight=HSIC_WEIGHT,
        device=DEVICE, save_path=SAVE_PATH)

    # 4. Test  (macro metrics — consistent with Tables 1, 6, 7 in the paper)
    model.load_state_dict(torch.load(SAVE_PATH))
    criterion = build_weighted_criterion(train_labels, num_classes, DEVICE)
    _, test_acc, test_f1, test_rec, test_pre = evaluate(
        model, test_loader, criterion, DEVICE, HSIC_WEIGHT, desc="Test")

    print("\n" + "=" * 60)
    print("Test results  (macro-averaged)")
    print(f"  Accuracy  : {test_acc:.2f}%")
    print(f"  Macro F1  : {test_f1:.4f}")
    print(f"  Macro Rec : {test_rec:.4f}")
    print(f"  Macro Pre : {test_pre:.4f}")
    print("=" * 60)
