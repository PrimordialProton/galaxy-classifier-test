"""
Evaluation script: loads a trained model, computes accuracy + confusion
matrix on a validation/test split, and saves a few misclassified examples
for the portfolio write-up (these are often the most interesting part of
the writeup - showing where and why the model struggles).

Example:
    python evaluate.py \
        --images_dir /path/to/images_training_rev1 \
        --labels_csv /path/to/training_solutions_rev1.csv \
        --model_path ../outputs/best_model.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from dataset import GalaxyZooDataset, CLASS_NAMES
from model import build_model, get_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="../outputs")
    p.add_argument("--subset_frac", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_transform = get_transforms()

    dataset = GalaxyZooDataset(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        transform=eval_transform,
        subset_frac=args.subset_frac,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
