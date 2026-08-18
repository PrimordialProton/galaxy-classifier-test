# Galaxy Classifier

A CNN-based classifier that sorts galaxy images into broad morphological
categories (spiral / elliptical / other) using crowd-sourced labels from the
[Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) project.

## Motivation

Modern sky surveys (JWST, Euclid, and the upcoming Vera Rubin Observatory)
produce far more galaxy images than astronomers can visually classify by
hand. Projects like Galaxy Zoo crowdsource human classification precisely
because of this bottleneck. This project explores whether a relatively
small, fine-tuned CNN can approximate that classification task, as a step
toward automated sorting/anomaly-flagging pipelines for large imaging
datasets.

## Approach

- **Data**: [Galaxy Zoo - The Galaxy Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
  (Kaggle), which provides galaxy images plus crowd-sourced vote fractions
  across a full morphology decision tree.
- **Label simplification (v1)**: rather than predicting the full decision
  tree, the three top-level vote fractions (`Class1.1` = smooth/elliptical,
  `Class1.2` = features/disk (spiral-like), `Class1.3` = star/artifact) are
  collapsed into a single 3-way label by taking the argmax. This keeps the
  first version of the project scoped and tractable.
- **Model**: ResNet18 pretrained on ImageNet, fine-tuned on galaxy images
  (transfer learning rather than training from scratch, given the dataset
  size and limited compute).
- **Training environment**: Kaggle notebooks (free GPU), since the dataset
  is already hosted there. Trained weights are then pulled down locally for
  inference/evaluation.

## Repo structure

```
src/
  dataset.py    - PyTorch Dataset for Galaxy Zoo images + labels
  model.py      - ResNet18 setup for fine-tuning
  train.py      - training loop
  evaluate.py   - accuracy, confusion matrix, misclassified examples
notebooks/
  train_kaggle.ipynb - notebook version for running on Kaggle's GPU
outputs/
  (trained model weights, evaluation plots)
```

## Results

Final model: ResNet18 (ImageNet-pretrained, `layer4` + new classifier head
fine-tuned), trained for 15 epochs on the full Galaxy Zoo training set with
a stratified 85/15 train/val split and class-weighted cross-entropy loss.

**Overall validation accuracy: ~84%**

```
              precision    recall  f1-score   support

  elliptical       0.80      0.86      0.83      4004
      spiral       0.88      0.83      0.86      5224
       other       0.23      0.56      0.32         9

    accuracy                           0.84      9237
   macro avg       0.64      0.75      0.67      9237
weighted avg       0.84      0.84      0.84      9237
```

**Elliptical vs. spiral** — the two well-represented classes — are classified
reliably (f1 0.83 and 0.86 respectively), which is the core result of this
project.

**The "other" (star/artifact) class** is a different story, and the process
of handling it is arguably the more interesting part of this project:

- "Other" is a naturally rare, garbage-collection-style category in Galaxy
  Zoo (non-galaxy objects that ended up in the imaging) — it made up only
  9 out of 9,237 validation examples even using the full dataset.
- A first attempt using a plain random train/val split happened to put
  almost all "other" examples on one side, making its reported metrics
  (0.00 recall) essentially meaningless rather than a real measure of model
  quality.
- Switching to a **stratified split** (preserving class proportions across
  train/val) plus **class-weighted loss** (inverse-frequency weighting,
  penalizing mistakes on "other" more heavily) brought "other" recall up
  from 0% to roughly 55-65%, at the cost of low precision (0.23-0.25) - the
  model now finds most "other" galaxies but also has a fair number of false
  positives.
- With only 9 validation examples, the exact recall/precision numbers for
  this class fluctuate by 10+ percentage points between runs from sampling
  noise alone - not something further training epochs can fix. This is a
  data volume limitation, not a modeling one.

Training for 15 epochs (vs. an initial 5) gave a modest additional gain on
the main classes (~1 point of accuracy) and plateaued - diminishing returns
past that point with this architecture/data setup.

## Next steps

- Expand beyond 3 broad classes to the fuller Galaxy Zoo decision tree
- Add anomaly detection (autoencoder-based outlier scoring) as a v2
- Pull live JWST imaging from MAST as a stretch goal, instead of the
  pre-packaged Kaggle dataset
