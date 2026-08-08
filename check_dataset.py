import os
import sys
import argparse
from collections import defaultdict

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Validate dataset folder before training")
    p.add_argument("--data_dir", type=str, default="data")
    return p.parse_args()


def scan_split(split_dir, split_name):
    problems = []
    warnings = []
    counts = {}

    if not os.path.isdir(split_dir):
        problems.append(f"[{split_name}] Folder not found: {split_dir}")
        return counts, problems, warnings

    found_classes = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    if not found_classes:
        problems.append(f"[{split_name}] No class subfolders found inside {split_dir}. "
                         f"Expected one folder per class, e.g. {split_dir}/NonDemented/")
        return counts, problems, warnings

    # class name mismatch vs config.py
    expected = set(config.CLASS_NAMES)
    found = set(found_classes)
    unexpected = found - expected
    missing = expected - found
    if unexpected:
        warnings.append(f"[{split_name}] Folder names not in config.CLASS_NAMES: {sorted(unexpected)} "
                         f"— these will still be used (the code reads folder names directly), "
                         f"but double check for typos.")
    if missing:
        warnings.append(f"[{split_name}] Classes listed in config.CLASS_NAMES but missing here: "
                         f"{sorted(missing)}")

    for cls in found_classes:
        cls_dir = os.path.join(split_dir, cls)
        files = os.listdir(cls_dir)
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS]
        non_image_files = [f for f in files if os.path.splitext(f)[1].lower() not in VALID_EXTS
                            and not f.startswith(".")]

        if non_image_files:
            warnings.append(f"[{split_name}/{cls}] {len(non_image_files)} non-image file(s) found "
                             f"(e.g. {non_image_files[:3]}) — these will be skipped or may error "
                             f"depending on your loader.")

        if len(image_files) == 0:
            problems.append(f"[{split_name}/{cls}] Empty — no image files found.")
            counts[cls] = 0
            continue

        # check a sample of images actually open correctly
        bad_files = []
        sample = image_files if len(image_files) <= 40 else image_files[::len(image_files)//40]
        for fname in sample:
            fpath = os.path.join(cls_dir, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception as e:
                bad_files.append((fname, str(e)))

        if bad_files:
            warnings.append(f"[{split_name}/{cls}] {len(bad_files)} corrupt/unreadable image(s) "
                             f"in sample, e.g. {bad_files[0][0]}: {bad_files[0][1]}")

        counts[cls] = len(image_files)

    return counts, problems, warnings


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    all_problems, all_warnings = [], []

    train_counts, p, w = scan_split(train_dir, "train")
    all_problems += p
    all_warnings += w

    test_counts, p, w = scan_split(test_dir, "test")
    all_problems += p
    all_warnings += w

    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print("\nImage counts:")
    print(f"{'Class':<22}{'Train':>10}{'Test':>10}")
    all_classes = sorted(set(train_counts) | set(test_counts))
    for cls in all_classes:
        print(f"{cls:<22}{train_counts.get(cls, '-'):>10}{test_counts.get(cls, '-'):>10}")

    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    print(f"\nTotal train images: {total_train}")
    print(f"Total test images:  {total_test}")

    # imbalance check
    if train_counts:
        max_c, min_c = max(train_counts.values()), min(train_counts.values())
        if min_c > 0 and max_c / min_c > 5:
            all_warnings.append(
                f"Strong class imbalance in train set: largest class has "
                f"{max_c} images, smallest has {min_c} ({max_c/min_c:.1f}x). "
                f"train.py already uses class-weighted loss to help with this, "
                f"but very small classes (<50 images) will still be hard to learn."
            )

    if all_warnings:
        print(f"\n⚠ WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  - {w}")

    if all_problems:
        print(f"\n✗ PROBLEMS ({len(all_problems)}) — fix these before training:")
        for p in all_problems:
            print(f"  - {p}")
        print("\nResult: NOT READY to train.")
        sys.exit(1)
    else:
        print("\n✓ No blocking problems found. Dataset looks ready for training.")
        if all_warnings:
            print("  (Review the warnings above — they won't stop training but may affect results.)")


if __name__ == "__main__":
    main()
