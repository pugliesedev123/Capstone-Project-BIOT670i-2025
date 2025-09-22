#!/usr/bin/env python3
import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import argparse

# We keep imports at top. Everything else happens in main().


def edge_fill(img: Image.Image):
    # Find the average color around the outer border of the image to apply.
    wimg = img if img.mode in ("RGB", "L") else img.convert("RGB")
    a = np.asarray(wimg)

    if wimg.mode == "L":  # grayscale
        edges = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
        return int(np.round(edges.mean()))
    else:  # RGB
        edges = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]], axis=0)
        return tuple(int(v) for v in np.round(edges.mean(axis=0)))


def pad_to_square(img: Image.Image, fill=None):
    # Make the image square by adding borders.
    # If no fill color is given, we compute one from the image edges.
    w, h = img.size
    if w == h:
        return img
    if fill is None:
        fill = edge_fill(img)
    s = max(w, h)
    pad_w = (s - w) // 2
    pad_h = (s - h) // 2
    padding = (pad_w, pad_h, s - w - pad_w, s - h - pad_h)
    return F.pad(img, padding, fill=fill, padding_mode="constant")


def geom_transform(img: Image.Image, degrees=5, scale=(0.8, 1.2), out_size=256, expand=True):
    # Rotate slightly
    # Scale slightly (zoom in or out)
    # Pad to make square
    # Resize to fixed size
    # Use edge color as fill to avoid black borders
    fill = edge_fill(img)
    angle = random.uniform(-degrees, degrees)
    sc = random.uniform(*scale)

    x = F.rotate(img, angle, expand=expand, fill=fill)
    x = F.affine(x, angle=0.0, translate=(0, 0), scale=sc, shear=(0.0, 0.0), fill=fill)
    x = pad_to_square(x, fill=fill)
    return F.resize(x, [out_size, out_size])


def is_taxon_dir(name: str) -> bool:
    # A class folder is either taxon-'' or mineral-''.
    n = name.lower()
    return n.startswith("taxon-") or n.startswith("mineral-")


def norm_class_key(name: str) -> str:
    # Normalize a class name so different owners map to the same class key.
    return name.strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Combine owners into a single dataset, send 1/5 to val, and save augmented images for the rest")
    parser.add_argument("--input-root", default="data/train", help="Root with owner-* subdirectories")
    parser.add_argument("--val-root", default="data/val/owner-combined", help="Where to copy validation images")
    parser.add_argument("--aug-root", default="data/augmented/owner-combined", help="Where to save augmented images")
    parser.add_argument("--aug-per-image", type=int, default=3, help="How many augmented samples per training image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and transforms")
    args = parser.parse_args()

    # Apply seed as early as possible so all randomness is controlled
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    IMAGE_EXTS = (".png", ".jpg", ".jpeg")

    # Define the image augmentation pipeline.
    # We do a small geometric transform, then a few light appearance tweaks, then to tensor.
    augment = transforms.Compose([
        transforms.Lambda(lambda img: geom_transform(img, degrees=5, scale=(0.8, 1.2), out_size=256)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.25),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomEqualize(p=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # Clear and recreate output folders.
    for folder in [args.val_root, args.aug_root]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted existing folder: {folder}")
        os.makedirs(folder, exist_ok=True)

    # Scan owner-* folders and collect all images by their class folder.
    taxon_to_images = {}

    for owner_folder in os.listdir(args.input_root):
        owner_path = os.path.join(args.input_root, owner_folder)
        if not (os.path.isdir(owner_path) and owner_folder.startswith("owner-")):
            continue

        for entry in os.listdir(owner_path):
            entry_path = os.path.join(owner_path, entry)
            if not (os.path.isdir(entry_path) and is_taxon_dir(entry)):
                continue

            taxon_key = norm_class_key(entry)
            taxon_to_images.setdefault(taxon_key, [])

            # Walk nested folders and collect images for this class
            for root, _, files in os.walk(entry_path):
                for f in files:
                    if f.lower().endswith(IMAGE_EXTS):
                        taxon_to_images[taxon_key].append(os.path.join(root, f))

    # ---------- process each taxon ----------
    # For each class split 1/5 to val and augment the remaining 4/5
    # If not >= 20 images, skip and report
    skipped = []
    for taxon_key, combined_images in taxon_to_images.items():
        if len(combined_images) < 20:
            skipped.append((taxon_key, len(combined_images)))
            continue

        random.shuffle(combined_images)
        n_total = len(combined_images)
        n_val = n_total // 5
        val_images = combined_images[:n_val]
        train_images = combined_images[n_val:]

        aug_taxon_dir = os.path.join(args.aug_root, taxon_key)
        val_taxon_dir = os.path.join(args.val_root, taxon_key)
        os.makedirs(aug_taxon_dir, exist_ok=True)
        os.makedirs(val_taxon_dir, exist_ok=True)

        # Copy validation images as-is so you can fairly measure performance later
        for src in val_images:
            try:
                shutil.copy2(src, os.path.join(val_taxon_dir, os.path.basename(src)))
            except Exception as e:
                print(f"Failed to copy validation image {src}: {e}")

        # For each training image, write several augmented versions
        desc = f"Augmenting {taxon_key} (train {len(train_images)}, val {len(val_images)})"
        for img_path in tqdm(train_images, desc=desc, unit="img"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {img_path}: {e}")
                continue

            base, _ = os.path.splitext(os.path.basename(img_path))

            for i in range(args.aug_per_image):
                try:
                    tensor = augment(img)                 # apply random transforms
                    out = transforms.ToPILImage()(tensor) # back to PIL for saving
                    out_name = f"{base}_aug_{i}.png"      # print file name plus augmentation number
                    out.save(os.path.join(aug_taxon_dir, out_name))
                except Exception as e:
                    print(f"Failed to augment {img_path}: {e}")

    if skipped:
        print("\nSkipped classes (fewer than 20 baseline images):")
        for name, count in skipped:
            print(f"  - {name}: {count} images")
    else:
        print("\nNo classes were skipped.")


if __name__ == "__main__":
    main()
