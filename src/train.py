"""
Training script for the galaxy classifier.

Run locally (CPU, for testing with a small --subset_frac) or on
Kaggle/Colab (GPU, full dataset). See notebooks/train_kaggle.ipynb for the
Kaggle-ready version of this same logic.

Example:
    python train.py \
        --images_dir /path/to/images_training_rev1 \
        --labels_csv /path/to/training_solutions_rev1.csv \
        --epochs 5 \
        --subset_frac 0.1
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import GalaxyZooDataset, CLASS_NAMES
from model import build_model, get_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--output_dir", default="../outputs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset_frac", type=float, default=None,
                    help="Use a fraction of the data for fast iteration (e.g. 0.1)")
    p.add_argument("--val_split", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, eval_transform = get_transforms()

    # Load full dataset once with eval_transform, then split; we'll swap
    # the train subset's transform below.
    full_dataset = GalaxyZooDataset(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        transform=eval_transform,
        subset_frac=args.subset_frac,
    )

    # Stratified split: a plain random split can easily starve the rare
    # 'other' class from one side entirely (this happened in an earlier
    # run - only 2 'other' examples landed in validation out of ~1800).
    # Stratifying on label keeps class proportions consistent across
    # train/val.
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        stratify=full_dataset.labels,
        random_state=42,
    )

    # Separate dataset instances for train/val so they can have different
    # transforms (augmentation only on train) without affecting each other.
    train_dataset_aug = GalaxyZooDataset(
        images_dir=args.images_dir, labels_csv=args.labels_csv,
        transform=train_transform, subset_frac=args.subset_frac,
    )
    train_ds = Subset(train_dataset_aug, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_labels = [full_dataset.labels[i] for i in train_idx]
    val_labels = [full_dataset.labels[i] for i in val_idx]
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"Train class counts: {np.bincount(train_labels, minlength=len(CLASS_NAMES))}")
    print(f"Val class counts:   {np.bincount(val_labels, minlength=len(CLASS_NAMES))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    # Class-weighted loss so the model doesn't just ignore the rare
    # 'other' class - weights computed from the full (pre-split) label
    # distribution.
    class_weights = torch.tensor(full_dataset.get_class_weights()).to(device)
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best val_acc, saved to {save_path}")

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
