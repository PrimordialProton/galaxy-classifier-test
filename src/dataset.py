"""
PyTorch Dataset for the Galaxy Zoo - The Galaxy Challenge dataset (Kaggle).

Expects:
  - An images directory containing files named "<GalaxyID>.jpg"
  - A labels CSV (training_solutions_rev1.csv) with columns including
    GalaxyID, Class1.1, Class1.2, Class1.3 (vote fractions)

Labels are collapsed to a single 3-class integer label by taking the
argmax of the three Class1.* vote fractions:
    0 = elliptical/smooth   (Class1.1)
    1 = spiral/features-disk (Class1.2)
    2 = star/artifact/other (Class1.3)
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["elliptical", "spiral", "other"]
LABEL_COLUMNS = ["Class1.1", "Class1.2", "Class1.3"]


class GalaxyZooDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None, subset_frac=None, seed=42):
        """
        Args:
            images_dir: path to directory of "<GalaxyID>.jpg" images
            labels_csv: path to training_solutions_rev1.csv
            transform: torchvision transform to apply to each image
            subset_frac: if set (e.g. 0.1), randomly sample this fraction
                of the data. Useful for fast local iteration before running
                a full training job on Kaggle.
            seed: random seed for the subsample
        """
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        missing = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"labels_csv is missing expected columns: {missing}")

        if subset_frac is not None:
            df = df.sample(frac=subset_frac, random_state=seed).reset_index(drop=True)

        df["label"] = np.argmax(df[LABEL_COLUMNS].values, axis=1)

        self.galaxy_ids = df["GalaxyID"].astype(str).tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, idx):
        galaxy_id = self.galaxy_ids[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.images_dir, f"{galaxy_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """
        Inverse-frequency class weights, for use with e.g.
        nn.CrossEntropyLoss(weight=...) to counteract class imbalance
        (the 'other'/star-artifact class is naturally very rare in
        Galaxy Zoo, since it's essentially a garbage-collection bucket
        for non-galaxy objects that ended up in the imaging).
        """
        counts = np.bincount(self.labels, minlength=len(CLASS_NAMES))
        counts = np.maximum(counts, 1)  # avoid divide-by-zero if a class is absent
        weights = counts.sum() / (len(CLASS_NAMES) * counts)
        return weights.astype(np.float32)
