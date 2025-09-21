import json
from tqdm import tqdm
import time
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models import vgg16, VGG16_Weights

# Function to pad an image to a square
def pad_to_square(img: Image.Image):
    w, h = img.size
    max_side = max(w, h)
    pad_w = (max_side - w) // 2
    pad_h = (max_side - h) // 2
    padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
    return F.pad(img, padding, fill=0, padding_mode='constant')

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='Train fossil image classifier')
parser.add_argument('--use-augmented', action='store_true', help='Use augmented dataset')
args = parser.parse_args()

# Directory Setup
base_train_dir = 'data/augmented' if args.use_augmented else 'data/train'
train_dir = os.path.join(base_train_dir, 'owner-combined')
val_dir = 'data/val/owner-combined'


# Rebuild owner-combined from scratch (if using original dataset)
if not args.use_augmented:
    if os.path.exists(train_dir):
        print(f"[CLEAN] Removing existing directory: {train_dir}")
        shutil.rmtree(train_dir)

    print(f"[INFO] Merging owner-* folders into: {train_dir}")
    os.makedirs(train_dir, exist_ok=True)

    for owner_folder in os.listdir(base_train_dir):
        owner_path = os.path.join(base_train_dir, owner_folder)
        if not os.path.isdir(owner_path) or not owner_folder.startswith('owner-'):
            continue

        for taxon_folder in os.listdir(owner_path):
            taxon_path = os.path.join(owner_path, taxon_folder)
            if not os.path.isdir(taxon_path):
                continue

            valid_images = [
                f for f in os.listdir(taxon_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not valid_images:
                print(f"[SKIP] {taxon_path} has no valid images.")
                continue

            dest_path = os.path.join(train_dir, taxon_folder)
            os.makedirs(dest_path, exist_ok=True)

            for img_file in valid_images:
                src = os.path.join(taxon_path, img_file)
                base_name = f"{owner_folder}_{img_file}"
                dst = os.path.join(dest_path, base_name)
                shutil.copyfile(src, dst)

# Create val/ if it doesn't exist (only when using original dataset)
if not os.path.exists(val_dir) and not args.use_augmented:
    os.makedirs(val_dir, exist_ok=True)
    for class_folder in os.listdir(train_dir):
        input_class_path = os.path.join(train_dir, class_folder)
        val_class_path = os.path.join(val_dir, class_folder)

        if not os.path.isdir(input_class_path):
            continue

        os.makedirs(val_class_path, exist_ok=True)
        images = [f for f in os.listdir(input_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        split_idx = len(images) // 5
        val_images = images[:split_idx]

        for img_file in val_images:
            src_path = os.path.join(input_class_path, img_file)
            dst_path = os.path.join(val_class_path, img_file)
            shutil.copyfile(src_path, dst_path)

# Validate Folders Exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

# Transforms
train_transforms = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(8),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.01),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.005 * torch.randn_like(x)),
    transforms.RandomErasing(p=0.1, scale=(0.01, 0.04)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load Data
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# Save class names
os.makedirs('models', exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(train_data.classes, f)

# Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("[WARN] GPU not detected — training on CPU will be slow.")

print(f"[INFO] Using device: {device}")

model = vgg16(weights=VGG16_Weights.DEFAULT)

print("[INFO] Loaded pretrained VGG-16")

#number of classes
num_classes = len(train_data.classes)

in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

print(f"[INFO] Updated final layer for {num_classes} classes")

model = model.to(device)
print("[INFO] Model moved to device")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)
print("[INFO] Optimizer and loss function initialized")

print("[INFO] Starting training loop...")

for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    print(f"\n[INFO] Starting Epoch {epoch+1}...")

    # Wrap DataLoader in tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / (batch_idx + 1)
        train_acc = 100 * correct / total

        # Update tqdm bar with current metrics
        progress_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Acc': f"{train_acc:.2f}%"})

    epoch_time = time.time() - start_time
    print(f"[INFO] Epoch {epoch+1} Complete | "
          f"Loss: {running_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Time: {int(epoch_time // 60)}m {int(epoch_time % 60)}s")

# Save Model
torch.save(model.state_dict(), 'models/fossil_vgg16.pt')
print("Model saved to models/fossil_vgg16.pt")