import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transforms(train: bool):
    """MRI-appropriate augmentation. Kept mild — scans are grayscale-ish and
    aggressive color/geometric augmentation can create anatomically
    unrealistic images."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


def build_dataloaders(data_dir: str, batch_size: int = config.BATCH_SIZE):
    """Returns train_loader, val_loader, test_loader, class_names, class_weights."""
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Expected a 'train' folder inside {data_dir} with one "
            f"subfolder per class. See README.md for the expected layout."
        )

    full_train = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    class_names = full_train.classes  # alphabetical order from folder names

    val_size = int(len(full_train) * config.VAL_SPLIT)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.SEED)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    # validation subset should use eval transforms (no augmentation) —
    # ImageFolder applies transform at __getitem__ time via the underlying
    # dataset object, so we wrap it with a separate ImageFolder instance.
    val_dataset_eval = datasets.ImageFolder(train_dir, transform=get_transforms(train=False))
    val_subset.dataset = val_dataset_eval

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    test_loader = None
    if os.path.isdir(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)

    # class weights (inverse frequency) to counter imbalance in the loss function
    targets = [full_train.targets[i] for i in train_subset.indices]
    class_counts = torch.zeros(len(class_names))
    for t in targets:
        class_counts[t] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_names)

    return train_loader, val_loader, test_loader, class_names, class_weights
