import os
import sys
import csv
import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import build_dataloaders
from src.model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train Alzheimer's MRI classifier")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--backbone", type=str, default="resnet18",
                    choices=["custom_cnn", "resnet18", "resnet50"])
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names, class_weights = build_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(args.backbone, num_classes=len(class_names),
                         img_size=config.IMG_SIZE, freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        history.append((epoch, train_loss, train_acc, val_loss, val_acc, elapsed))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "backbone": args.backbone,
                "val_acc": val_acc,
            }, os.path.join(args.output_dir, config.CHECKPOINT_NAME))
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    plot_history(history, args.output_dir)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {os.path.join(args.output_dir, config.CHECKPOINT_NAME)}")


def plot_history(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h[0] for h in history]
    train_loss = [h[1] for h in history]
    train_acc = [h[2] for h in history]
    val_loss = [h[3] for h in history]
    val_acc = [h[4] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)


if __name__ == "__main__":
    main()
