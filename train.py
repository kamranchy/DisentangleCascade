from dataset import *
from model import *
from utils import *
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    DATASET_PATH = 'your_dataset_path'
    CSV_PATH = 'your_csv_path'

    SELECTED_CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC']

    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Preparing dataset...")
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels, class_names, label_to_idx = prepare_isic_data(
        DATASET_PATH, CSV_PATH, selected_classes=SELECTED_CLASSES
    )

    train_dataset = ISICDataset(train_paths, train_labels, transform=get_transforms(IMAGE_SIZE, 'train'))
    val_dataset = ISICDataset(val_paths, val_labels, transform=get_transforms(IMAGE_SIZE, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DisentangleCascade(num_classes=len(class_names))

    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE
    )
