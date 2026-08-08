import os
import sys
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import build_dataloaders
from src.model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Alzheimer's MRI classifier")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, default=os.path.join(config.OUTPUT_DIR, config.CHECKPOINT_NAME))
    p.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]

    model = build_model(backbone, num_classes=len(class_names), img_size=config.IMG_SIZE).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, _, _ = build_dataloaders(args.data_dir)
    if test_loader is None:
        raise FileNotFoundError(
            f"No 'test' folder found inside {args.data_dir}. Add one with the "
            f"same class-subfolder structure as 'train' to evaluate."
        )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, args.output_dir)
    print(f"\nSaved classification_report.txt and confusion_matrix.png to {args.output_dir}")


def plot_confusion_matrix(cm, class_names, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)


if __name__ == "__main__":
    main()
