from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os
import random
import shutil
import torch
import torchvision.transforms.functional as F

# Known dimensions from mobile devices (WxH and HxW for portrait/landscape)
known_mobile_dims = {
    (4032, 3024), (3024, 4032),
    (4000, 3000), (3000, 4000),
    (4624, 3472), (3472, 4624),
    (4080, 3072), (3072, 4080)
}


def crop_center_if_mobile(img: Image.Image) -> Image.Image:
    w, h = img.size
    if (w, h) in known_mobile_dims:
        side = int(min(w, h) * (2/3))  # Crop to 2/3 of shorter side
        left = (w - side) // 2
        top = (h - side) // 2
        right = left + side
        bottom = top + side
        img = img.crop((left, top, right, bottom))
    return img

# Function to pad an image to a square


def pad_to_square(img: Image.Image):
    w, h = img.size
    max_side = max(w, h)
    pad_w = (max_side - w) // 2
    pad_h = (max_side - h) // 2
    padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
    return F.pad(img, padding, fill=0, padding_mode='constant')

# Apply your standard pre-resize to keep outputs consistent with augment()


def preprocess(img: Image.Image) -> Image.Image:
    img = pad_to_square(img)
    return img.resize((224, 224), Image.BICUBIC)


augment = transforms.Compose([
    # transforms.Lambda(crop_center_if_mobile),         # Crop center square if mobile
    # Pad if not already square
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),                    # Resize to square
    transforms.RandomRotation(9),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ColorJitter(brightness=0.06, contrast=0.06,
                           saturation=0.04, hue=0.01),
    # Occasionally convert to grayscale (10% of the time)
    transforms.RandomGrayscale(p=0.1),
    # Additional subtle color shift, possibly redundant
    transforms.ColorJitter(brightness=0.02, contrast=0.02,
                           saturation=0.02, hue=0.02),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.005 *
                      torch.randn_like(x)),  # Add light noise
    transforms.RandomErasing(p=0.1, scale=(0.01, 0.04)),
])

# Paths
input_root = 'data/train'
val_root = 'data/val/owner-combined'
aug_root = 'data/augmented/owner-combined'

# Optional cleanup
for folder in ['data/val', 'data/augmented']:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted existing folder: {folder}")

os.makedirs(val_root, exist_ok=True)
os.makedirs(aug_root, exist_ok=True)

# Collect all taxon folder paths across owners
taxon_to_paths = {}

for owner_folder in os.listdir(input_root):
    owner_path = os.path.join(input_root, owner_folder)
    if not os.path.isdir(owner_path) or not owner_folder.startswith('owner-'):
        continue

    for root, dirs, files in os.walk(owner_path):
        for d in dirs:
            taxon_path = os.path.join(root, d)
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(taxon_path)):
                taxon_name = os.path.relpath(taxon_path, owner_path)
                taxon_to_paths.setdefault(taxon_name, []).append(taxon_path)

# Process each taxon
for taxon_name, paths in taxon_to_paths.items():
    print(f"Processing taxon: {taxon_name}")
    combined_images = []

    for path in paths:
        for file in os.listdir(path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                combined_images.append(os.path.join(path, file))

    if not combined_images:
        continue

    # No split: originals are the validation set, and we augment the same originals for training
    val_images = combined_images
    train_images = combined_images

    aug_taxon_dir = os.path.join(aug_root, taxon_name)
    val_taxon_dir = os.path.join(val_root, taxon_name)
    os.makedirs(aug_taxon_dir, exist_ok=True)
    os.makedirs(val_taxon_dir, exist_ok=True)

    # Save validation images (unaltered originals)
    for img_path in val_images:
        try:
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(val_taxon_dir, filename))
        except Exception as e:
            print(f"Failed to copy validation image {img_path}: {e}")

    # Augment training images from originals
    for idx, img_path in enumerate(tqdm(train_images, desc=f"Augmenting {taxon_name}", unit="img")):
        try:
            img = Image.open(img_path).convert('RGB')

            # --- Baseline outputs that must exist ---
            # 1) Grayscale version
            g = preprocess(img.convert('L').convert('RGB'))
            g.save(os.path.join(aug_taxon_dir, f"{idx}_base_gray.png"))

            # 2) Background-removed placeholder (pass-through for now)
            # TODO: Replace 'img' with a real background-removed image
            rb = preprocess(img)
            rb.save(os.path.join(aug_taxon_dir, f"{idx}_base_bgclean.png"))

            # 3) Grayscale of background-removed
            rb_g = preprocess(rb.convert('L').convert('RGB'))
            rb_g.save(os.path.join(aug_taxon_dir,
                      f"{idx}_base_bgclean_gray.png"))

            fixed_count = 3

            # 4) Zoom-in if mobile
            if img.size in known_mobile_dims:
                z = preprocess(crop_center_if_mobile(img))
                z.save(os.path.join(aug_taxon_dir,
                       f"{idx}_base_mobile_zoom.png"))
                fixed_count = 4

            # --- Fill the rest up to 20 total outputs with random augmentations ---
            needed = 20 - fixed_count
            for i in range(needed):
                augmented = augment(img)
                save_path = os.path.join(aug_taxon_dir, f"{idx}_rand_{i}.png")
                transforms.ToPILImage()(augmented).save(save_path)

        except Exception as e:
            print(f"Failed to process training image {img_path}: {e}")
