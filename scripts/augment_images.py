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


def geom_transform(img: Image.Image, degrees=5, scale=(0.8, 1.2), out_size=256, expand=True, disabled_args=[]):
    # Rotate slightly
    # Scale slightly (zoom in or out)
    # Pad to make square
    # Resize to fixed size
    # Use edge color as fill to avoid black borders
    fill = edge_fill(img)
    angle = random.uniform(-degrees, degrees)
    sc = random.uniform(*scale)
    x = img

    # disable-tf arguments:
    if("rotate" not in disabled_args):
        x = F.rotate(img, angle, expand=expand, fill=fill)
    if ("zoom" not in disabled_args):
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
    # Command-Line Arguments
    parser = argparse.ArgumentParser(description="Combine owners into a single dataset, send 1/5 to val, and save augmented images for the rest")
    parser.add_argument("--input-root", default="data/train", help="Root with owner-* subdirectories")
    parser.add_argument("--val-root", default="data/val/owner-combined", help="Where to copy validation images")
    parser.add_argument("--aug-root", default="data/augmented/owner-combined", help="Where to save augmented images")
    parser.add_argument("--aug-per-image", type=int, default=3, help="How many augmented samples per training image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and transforms")
    parser.add_argument("--console-print", action='store_true', help="Print extra details to console")
    parser.add_argument("--exclude-classes", action='store_true', help="Remove select classes marked with an '-' from taxa-config.py")
    parser.add_argument("--include-config-classes-only", action='store_true', help="Include only classes in taxa-config.py and start with a '+'")
    parser.add_argument("--threshold", type=int, help="Generate class balance by defining a threshold that will remove classes if they do not an image count that exceeds this number. Randomly excise images from classes that exceed this number until they are equal to the threshold.")
    parser.add_argument("--disable-tf", action='append', type=str.lower, default=[], help="Disable specific transformations for augmentation. The following transforms can be disabled by repeatedly calling the argument: rotate, scale, zoom, horizontalflip, verticalflip, grayscale, equalize, sharpen")
    parser.add_argument("--disable-ca", action='append', type=str.lower, default=[], help="Disable specific class for augmentation (eg. --disable-ca exogyra_sp)")
    args = parser.parse_args()

    # Apply seed as early as possible so all randomness is controlled
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    IMAGE_EXTS = (".png", ".jpg", ".jpeg")
    if(len(args.disable_tf) > 0 and args.console_print):
        for argument in args.disable_tf:
            print(f"[INFO] Disabling the following argument {argument}")

    # Define the image augmentation pipeline.
    # We do a small geometric transform, then a few light appearance tweaks, then to tensor.
    # Used ChatGPT to get this list-comprehension logic
    on = lambda n: n not in args.disable_tf
    augment = transforms.Compose([operation for operation in [
        transforms.Lambda(lambda img: geom_transform(img, degrees=5, scale=(0.8, 1.2), out_size=256, disabled_args=args.disable_tf)),
        on("sharpen") and transforms.RandomAdjustSharpness(1.5, p=0.25),
        on("grayscale") and transforms.RandomGrayscale(p=0.2),
        on("equalize") and transforms.RandomEqualize(p=0.25),
        on("horizontalflip") and transforms.RandomHorizontalFlip(p=0.5),
        on("verticalflip") and transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ] if operation])


    # Clear and recreate output folders.
    for folder in [args.val_root, args.aug_root]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted existing folder: {folder}")
        os.makedirs(folder, exist_ok=True)

    # Declare exclusion and inclusion lists.
    taxon_exclusion_list = []
    taxon_inclusion_list = []

    # If arguments that define subset of classes using taxa-config.py are present:
    # seek those files out and append those classes to the above lists
    if(args.include_config_classes_only or args.exclude_classes):
        if(os.path.isfile('taxa-config.txt')):
            file = open("taxa-config.txt", "r")
            line = file.readline()
            while line:
                # + icon defines classes we need to include
                if(args.include_config_classes_only and line[0] == "+"):
                    taxon_inclusion_list.append(line.strip("-+\n"))
                # - icon defines classes we need to exclude
                elif(args.exclude_classes and line[0] == "-"):
                    taxon_exclusion_list.append(line.strip("-+\n"))
                line = file.readline()
            file.close()
        else:
            print("[INFO] The file taxa-config.txt does not live in the directory. Run utils/taxa_for_config.py to generate.")
            exit()
    
    if args.include_config_classes_only:
        print("\n[INFO] Including only the the following taxa for classification:")
        for name in taxon_inclusion_list:
            print(f"  - {name}")
    if args.exclude_classes:
        print("\n[INFO] Removing the following taxa from classification:")
        for name in taxon_exclusion_list:
            print(f"  - {name}")

    # Scan owner-* folders and collect all images by their class folder.
    # This loop will exclude classes if they are explicetely removed via taxa-config.py
    taxon_to_images = {}  

    for owner_folder in os.listdir(args.input_root):
        owner_path = os.path.join(args.input_root, owner_folder)
        if not (os.path.isdir(owner_path) and owner_folder.startswith("owner-")):
            continue
        for entry in os.listdir(owner_path):
            entry_path = os.path.join(owner_path, entry)
            if not (os.path.isdir(entry_path) and is_taxon_dir(entry)):
                continue
            elif(args.exclude_classes and entry.replace("taxon-","") in taxon_exclusion_list):
                continue
            elif(args.include_config_classes_only and entry.replace("taxon-","") not in taxon_inclusion_list):
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
    threshold_skipped = []
    for taxon_key, combined_images in taxon_to_images.items():
        if len(combined_images) < 20:
            skipped.append((taxon_key, len(combined_images)))
            continue
        
        random.shuffle(combined_images)
        n_total = len(combined_images)
        n_val = n_total // 5

        # skip classes if they don't exceed the threshold limit
        if len(combined_images) < args.threshold or (n_total - n_val) < args.threshold:
            threshold_skipped.append((taxon_key, len(combined_images)))
            continue
        
        # if classes exceed the threshold limit, remove some of the combined images until the threshold is reached
        if n_total > args.threshold:
            for delta in range(n_total - args.threshold):
                random_element = random.choice(combined_images)
                combined_images.remove(random_element)
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

        # If a class has been disabled from augmentation, move those training images to the augmentation folder unedited
        if taxon_key.replace("taxon-","") in args.disable_ca:
            for src in train_images:
                try:
                    shutil.copy2(src, os.path.join(aug_taxon_dir, os.path.basename(src)))
                except Exception as e:
                    print(f"Failed to copy training image {src}: {e}")
            continue

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

    if args.console_print:
        if skipped:
            print("\n[INFO] Skipped classes with fewer than 20 baseline images:")
            for name, count in skipped:
                print(f"  - {name}: {count} images")
        elif(args.threshold):
            print(f"\n[INFO] Skipped classes that missed manually definied threshold of {args.threshold}:")
            for name, count in threshold_skipped:
                print(f"  - {name}: {count} images")
        else:
            print("\n[INFO] No classes were skipped.")


if __name__ == "__main__":
    main()