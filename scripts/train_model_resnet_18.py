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
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def pad_to_square(img: Image.Image):
    # Add padding so width equals height
    # This centers the original image in a square canvas
    w, h = img.size
    max_side = max(w, h)
    pad_w = (max_side - w) // 2
    pad_h = (max_side - h) // 2
    padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
    # Fill uses black here. You can change this if desired.
    return F.pad(img, padding, fill=0, padding_mode='constant')

def main():
        
    # Command-Line Arguments
    parser = argparse.ArgumentParser(description='Train fossil image classifier')
    parser.add_argument('--use-augmented', action='store_true', help='Use augmented dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--model-path', type=str, default='models/fossil_resnet18.pt', help='Path to save model weights')
    parser.add_argument('--index-path', type=str, default='models/train_index.pt', help='Path to save training embedding index')
    args = parser.parse_args()

    # Apply seed as early as possible so all randomness is controlled
    set_seed(args.seed)

    # ---------------- PATCHED: Directory Setup ----------------
    # If using augmented, look for merged data in data/augmented/owner-combined
    # If not using augmented, still WRITE the merged owner-combined into data/augmented/owner-combined,
    # but READ sources from data/train/owner-*
    if args.use_augmented:
        source_root = 'data/augmented'   # not used for merging below
        train_dir = os.path.join('data/augmented', 'owner-combined')
    else:
        source_root = 'data/train'       # where owner-* live
        train_dir = os.path.join('data/augmented', 'owner-combined')

    val_dir = 'data/val/owner-combined'

    # Rebuild owner-combined from scratch when using the original dataset
    # This merges all owner-* folders from data/train into data/augmented/owner-combined
    if not args.use_augmented:
        if os.path.exists(train_dir):
            print(f"[CLEAN] Removing existing directory: {train_dir}")
            shutil.rmtree(train_dir)

        print(f"[INFO] Merging owner-* folders from {source_root} into: {train_dir}")
        os.makedirs(train_dir, exist_ok=True)

        # Walk each owner folder, then each taxon subfolder, and copy images into combined layout
        for owner_folder in os.listdir(source_root):
            owner_path = os.path.join(source_root, owner_folder)
            if not os.path.isdir(owner_path) or not owner_folder.startswith('owner-'):
                continue

            for taxon_folder in os.listdir(owner_path):
                taxon_path = os.path.join(owner_path, taxon_folder)
                if not os.path.isdir(taxon_path):
                    continue

                # Accept common image formats
                valid_images = [
                    f for f in os.listdir(taxon_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if not valid_images:
                    print(f"[SKIP] {taxon_path} has no valid images.")
                    continue

                # Make class folder in the combined train directory
                dest_path = os.path.join(train_dir, taxon_folder)
                os.makedirs(dest_path, exist_ok=True)

                # Copy images and prefix filenames with owner name to avoid clashes
                for img_file in valid_images:
                    src = os.path.join(taxon_path, img_file)
                    base_name = f"{owner_folder}_{img_file}"
                    dst = os.path.join(dest_path, base_name)
                    shutil.copyfile(src, dst)

    # Create val set once when it does not exist and we are using the original dataset
    # This takes a small slice from each class for validation
    if not os.path.exists(val_dir) and not args.use_augmented:
        os.makedirs(val_dir, exist_ok=True)
        for class_folder in os.listdir(train_dir):
            input_class_path = os.path.join(train_dir, class_folder)
            val_class_path = os.path.join(val_dir, class_folder)

            if not os.path.isdir(input_class_path):
                continue

            os.makedirs(val_class_path, exist_ok=True)
            images = [f for f in os.listdir(
                input_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)  # random split per class
            split_idx = len(images) // 5  # 20 percent to validation
            val_images = images[:split_idx]

            # Copy chosen images into the validation folder
            for img_file in val_images:
                src_path = os.path.join(input_class_path, img_file)
                dst_path = os.path.join(val_class_path, img_file)
                shutil.copyfile(src_path, dst_path)

    # Validate that train and val folders exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Extra check to make sure each validation class has at least one image
    for class_folder in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(
            class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            raise RuntimeError(f"[ERROR] Validation folder {class_path} is empty!")

    # ImageNet statistics used for pretrained ResNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Transforms
    # These resize and normalize images to the format ResNet expects
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load Data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

    # DataLoaders feed batches of images to the model
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # Compute per-class weights from the training set
    # Classes with fewer samples get higher weight to balance the loss
    targets = torch.tensor(train_data.targets)
    class_counts = torch.bincount(targets)
    class_weights = (1.0 / class_counts.float())
    # Normalization so average weight is near 1
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    print("[INFO] Class counts:", class_counts.tolist())
    print("[INFO] Class weights:", class_weights.tolist())

    # Save class names next to the model path for later use by the predictor
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    with open(os.path.join(os.path.dirname(args.model_path) or ".", "class_names.json"), "w") as f: json.dump(train_data.classes, f)

    # Pick GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] GPU not detected - training on CPU will be slow.")

    # Load a ResNet-18 with pretrained weights to get a strong starting point
    model = models.resnet18(pretrained=True)
    print("[INFO] Loaded pretrained ResNet-18")

    # Replace the final layer to match the number of fossil classes
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    print(f"[INFO] Updated final layer for {len(train_data.classes)} classes")

    # Move model to device
    model = model.to(device)
    print("[INFO] Model moved to device")

    # Use weighted cross entropy to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Train only the final layer for now to keep things simple and fast
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    print("[INFO] Optimizer and loss function initialized")

    print("[INFO] Starting training loop...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        print(f"\n[INFO] Starting Epoch {epoch+1}/{args.epochs}...")

        # Wrap the loader with a progress bar for live feedback
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            # Reset gradients, run forward pass, compute loss, backprop, and update weights
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # Track metrics for display
            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = running_loss / (batch_idx + 1)
            train_acc = 100 * correct / total

            # Show current loss and accuracy on the progress bar
            progress_bar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Acc': f"{train_acc:.2f}%"})

        epoch_time = time.time() - start_time
        print(f"[INFO] Epoch {epoch+1} Complete | "
            f"Loss: {running_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Time: {int(epoch_time // 60)}m {int(epoch_time % 60)}s")


    # Save trained weights so you can load them later for prediction
    torch.save(model.state_dict(), args.model_path)
    print(f"[INFO] Model saved to {args.model_path}")

    # This builds a lookup table from training images to their feature vectors
    # Later you can find the nearest training image to a new query image
    print("[INFO] Building training embedding index...")

    # Create an embedder that outputs 512-length features by removing the final layer
    embedder = models.resnet18(pretrained=False)
    embedder.fc = nn.Identity()

    # Copy all trained weights except the final classifier layer
    state = model.state_dict()
    state_no_fc = {k: v for k, v in state.items() if not k.startswith('fc.')}
    missing, unexpected = embedder.load_state_dict(state_no_fc, strict=False)
    if missing:
        print(f"[INFO] Embedder missing keys: {missing}")
    if unexpected:
        print(f"[INFO] Embedder unexpected keys: {unexpected}")
    embedder = embedder.to(device).eval()

    # Use deterministic preprocessing to index the training set
    index_loader = DataLoader(
        datasets.ImageFolder(train_dir, transform=val_transforms),
        batch_size=max(64, args.batch_size), shuffle=False
    )

    # Collect features, labels, and original file paths
    all_vecs = []
    all_labels = []
    all_paths = [p for p, _ in index_loader.dataset.samples]
    class_to_idx = index_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with torch.no_grad():
        for imgs, labels in tqdm(index_loader, desc="Indexing", unit="batch"):
            imgs = imgs.to(device)
            vecs = embedder(imgs)                                # shape [B, 512]
            vecs = torch.nn.functional.normalize(vecs, dim=1)    # unit length for cosine similarity
            all_vecs.append(vecs.cpu())
            all_labels.extend(labels.tolist())

    # Stack all feature tensors into one big matrix
    embeddings = torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0, 512)

    # Package the index for saving
    index_obj = {
        "embeddings": embeddings,                 # FloatTensor [N, 512]
        "labels": torch.tensor(all_labels),       # LongTensor [N]
        "paths": all_paths,                       # list[str]
        "class_to_idx": class_to_idx,             # dict
        "idx_to_class": idx_to_class,             # dict
    }

    # Save the index for use by the prediction script
    os.makedirs(os.path.dirname(args.index_path) or ".", exist_ok=True)
    torch.save(index_obj, args.index_path)
    print(f"[INFO] Saved training index to {args.index_path}")


if __name__ == "__main__":
    main()
