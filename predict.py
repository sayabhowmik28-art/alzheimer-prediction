import os
import sys
import argparse

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_transforms
from src.model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Predict Alzheimer's stage from a single MRI image")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=os.path.join(config.OUTPUT_DIR, config.CHECKPOINT_NAME))
    return p.parse_args()


def predict(image_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]

    model = build_model(backbone, num_classes=len(class_names), img_size=config.IMG_SIZE).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    ranked = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    return ranked


if __name__ == "__main__":
    args = parse_args()
    results = predict(args.image, args.checkpoint)
    print(f"\nPredictions for {args.image}:")
    for label, prob in results:
        print(f"  {label:<20s} {prob*100:5.2f}%")
    print(f"\n=> Predicted class: {results[0][0]} ({results[0][1]*100:.2f}% confidence)")
