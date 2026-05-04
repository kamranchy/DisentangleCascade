import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size=224, mode='train'):
    """MICCAI-style augmentation pipeline."""
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.HueSaturationValue(20, 30, 20, p=0.8),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.3),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class ISICDataset(Dataset):
    """ISIC skin lesion dataset."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, self.labels[idx]


def prepare_isic_data(dataset_path, csv_path, selected_classes=None,
                      test_size=0.2, val_size=0.1, random_state=42):
    """
    Build stratified train / val / test splits from ISIC CSV.

    Returns
    -------
    train_paths, val_paths, test_paths,
    train_labels, val_labels, test_labels,
    class_names, label_to_idx
    """
    df = pd.read_csv(csv_path)
    exclude = {'image', 'Image', 'ISIC_ID'}
    class_cols = [c for c in df.columns if c not in exclude]
    df['label'] = df[class_cols].idxmax(axis=1)

    if selected_classes:
        df = df[df['label'].isin(selected_classes)]
        class_cols = selected_classes

    print(f"Classes  : {class_cols}")
    print(f"Distribution:\n{df['label'].value_counts()}\n")

    image_paths, labels = [], []
    for cls in class_cols:
        folder = os.path.join(dataset_path, cls)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith('.jpg'):
                image_paths.append(os.path.join(folder, f))
                labels.append(cls)

    label_to_idx = {l: i for i, l in enumerate(class_cols)}
    labels_enc = np.array([label_to_idx[l] for l in labels])

    print(f"Total images: {len(image_paths)}")

    trval_p, test_p, trval_l, test_l = train_test_split(
        image_paths, labels_enc, test_size=test_size,
        stratify=labels_enc, random_state=random_state)

    val_adj = val_size / (1 - test_size)
    train_p, val_p, train_l, val_l = train_test_split(
        trval_p, trval_l, test_size=val_adj,
        stratify=trval_l, random_state=random_state)

    print(f"Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}")
    return train_p, val_p, test_p, train_l, val_l, test_l, class_cols, label_to_idx
