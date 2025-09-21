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

augment = transforms.Compose([
    transforms.Lambda(crop_center_if_mobile),         # Crop center square if mobile
    transforms.Lambda(pad_to_square),                 # Pad if not already square
    transforms.Resize((224, 224)),                    # Resize to square
    transforms.RandomRotation(9),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.01),
    transforms.RandomGrayscale(p=0.1),                                                  # Occasionally convert to grayscale (10% of the time)
    transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),  # Additional subtle color shift, possibly redundant
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.005 * torch.randn_like(x)),  # Add light noise
    transforms.RandomErasing(p=0.1, scale=(0.01, 0.04)),
])

# Paths
input_root = 'data/train'
val_root = 'data/val/owner-combined'
aug_root = 'data/augmented/owner-combined'

#input_root = r"C:\Users\bryan\PycharmProjects\Capstone-Project-BIOT670i-2025\data\train"
#val_root = r"C:\Users\bryan\PycharmProjects\Capstone-Project-BIOT670i-2025\data\val"
#aug_root = r"C:\Users\bryan\PycharmProjects\Capstone-Project-BIOT670i-2025\data\augmented"


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

    random.shuffle(combined_images)
    split_idx = len(combined_images) // 5
    val_images = combined_images[:split_idx]
    train_images = combined_images[split_idx:]

    aug_taxon_dir = os.path.join(aug_root, taxon_name)
    val_taxon_dir = os.path.join(val_root, taxon_name)
    os.makedirs(aug_taxon_dir, exist_ok=True)
    os.makedirs(val_taxon_dir, exist_ok=True)

    # Save validation images (unaltered, crop on mobile images)
    for img_path in val_images:
        try:
            img = crop_center_if_mobile(Image.open(img_path).convert('RGB'))
            filename = os.path.basename(img_path)
            img.save(os.path.join(val_taxon_dir, filename))
        except Exception as e:
            print(f"Failed to process validation image {img_path}: {e}")

    # Augment training images
    for idx, img_path in enumerate(tqdm(train_images, desc=f"Augmenting {taxon_name}", unit="img")):
        try:
            img = Image.open(img_path).convert('RGB')
            for i in range(20):  # 20 augmentations per image
                augmented = augment(img)
                save_path = os.path.join(aug_taxon_dir, f"{idx}_{i}.png")
                transforms.ToPILImage()(augmented).save(save_path)
        except Exception as e:
            print(f"Failed to process training image {img_path}: {e}")