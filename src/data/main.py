import os
import gzip
import shutil
import h5py
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================
# 1. Config
# =========================
DATA_DIR = "data"

TRAIN_X_GZ = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_train_x.h5.gz")
TRAIN_Y_GZ = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_train_y.h5.gz")
VALID_X_GZ = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_valid_x.h5.gz")
VALID_Y_GZ = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_valid_y.h5.gz")

TRAIN_X_PATH = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_train_x.h5")
TRAIN_Y_PATH = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_train_y.h5")
VALID_X_PATH = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_valid_x.h5")
VALID_Y_PATH = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_valid_y.h5")

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

TRAIN_LIMIT = 5000
VALID_LIMIT = 1000


# =========================
# 2. unzip if needed
# =========================
def unzip_gz_if_needed(gz_path, h5_path):
    if os.path.exists(h5_path):
        print(f"[Skip] already exists: {h5_path}")
        return

    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Missing file: {gz_path}")

    print(f"[Unzip] {gz_path} -> {h5_path}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(h5_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"[Done] {h5_path}")


def prepare_data_files():
    unzip_gz_if_needed(TRAIN_X_GZ, TRAIN_X_PATH)
    unzip_gz_if_needed(TRAIN_Y_GZ, TRAIN_Y_PATH)
    unzip_gz_if_needed(VALID_X_GZ, VALID_X_PATH)
    unzip_gz_if_needed(VALID_Y_GZ, VALID_Y_PATH)


# =========================
# 3. Read h5
# =========================
def load_h5(x_path, y_path, limit=None):
    with h5py.File(x_path, "r") as fx:
        x = fx["x"][:]

    with h5py.File(y_path, "r") as fy:
        y = fy["y"][:]

    y = np.array(y).reshape(-1)

    if limit is not None:
        x = x[:limit]
        y = y[:limit]

    return x, y


# =========================
# 4. Dataset
# =========================
class PCamDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# =========================
# 5. Model
# =========================
class SimplePCamCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =========================
# 6. Eval
# =========================
def evaluate(model, loader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc


# =========================
# 7. Train
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# =========================
# 8. Main
# =========================
def main():
    prepare_data_files()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading data...")
    x_train, y_train = load_h5(TRAIN_X_PATH, TRAIN_Y_PATH, limit=TRAIN_LIMIT)
    x_valid, y_valid = load_h5(VALID_X_PATH, VALID_Y_PATH, limit=VALID_LIMIT)

    print("Train shape:", x_train.shape, y_train.shape)
    print("Valid shape:", x_valid.shape, y_valid.shape)

    train_dataset = PCamDataset(x_train, y_train, transform=train_transform)
    valid_dataset = PCamDataset(x_valid, y_valid, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SimplePCamCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_auc = evaluate(model, valid_loader, device)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_pcam_baseline.pth")
            print("Saved best model.")

    print("Training finished. Best Val AUC:", best_auc)


if __name__ == "__main__":
    main()