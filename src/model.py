"""
ResNet18-based classifier for galaxy morphology.

Uses ImageNet-pretrained weights and replaces the final fully-connected
layer for our 3-class problem. By default, only the final layer (and
optionally the last residual block) is trained, with the rest of the
network frozen - a standard transfer learning setup that trains fast
and works well on smaller datasets.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(num_classes=3, unfreeze_last_block=True, pretrained=True):
    """
    pretrained=True (default, used for training): downloads ImageNet
    weights as the starting point for fine-tuning.

    pretrained=False (used for inference-only scripts, e.g. the
    classifier pipeline): skips that download entirely and starts from
    a randomly-initialized architecture. This is correct whenever you're
    about to immediately overwrite every parameter with your own trained
    checkpoint via load_state_dict() anyway - the ImageNet weights would
    just be discarded, so there's no reason to fetch them (and no reason
    to depend on network access to PyTorch's CDN at inference time).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Optionally unfreeze the last residual block (layer4) for more
    # capacity to adapt to galaxy images, which look quite different
    # from ImageNet's everyday objects.
    if unfreeze_last_block:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the final classification layer (always trainable)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_transforms():
    """Standard ImageNet-style preprocessing, resized for galaxy thumbnails."""
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # galaxies have no inherent "up"
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform
