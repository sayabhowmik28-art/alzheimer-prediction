import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /16
        )
        reduced = img_size // 16
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_transfer_model(backbone: str, num_classes: int, freeze_backbone: bool = False):
    """backbone: 'resnet18' or 'resnet50'"""
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    # the new head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def build_model(backbone: str, num_classes: int, img_size: int = 224, freeze_backbone: bool = False):
    if backbone == "custom_cnn":
        return CustomCNN(num_classes=num_classes, img_size=img_size)
    return build_transfer_model(backbone, num_classes, freeze_backbone=freeze_backbone)
