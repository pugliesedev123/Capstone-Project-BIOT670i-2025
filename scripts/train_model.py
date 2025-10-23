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


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    cand = f"{root}_{i}{ext}"
    while os.path.exists(cand):
        i += 1
        cand = f"{root}_{i}{ext}"
    return cand


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def ensure_min_val_samples(train_dir: str, val_dir: str, min_per_class: int = 1):
    os.makedirs(val_dir, exist_ok=True)
    for img_class in sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))):
        timg_class = os.path.join(train_dir, img_class)
        vimg_class = os.path.join(val_dir, img_class)
        os.makedirs(vimg_class, exist_ok=True)
        val_imgs = [f for f in os.listdir(vimg_class) if f.lower().endswith(IMAGE_EXTS)]
        if len(val_imgs) >= min_per_class:
            continue
        train_imgs = [f for f in os.listdir(timg_class) if f.lower().endswith(IMAGE_EXTS)]
        if not train_imgs:
            continue
        random.shuffle(train_imgs)
        need = min_per_class - len(val_imgs)
        for f in train_imgs[:need]:
            src = os.path.join(timg_class, f)
            dst = os.path.join(vimg_class, f)
            dst = unique_path(dst)
            shutil.copy2(src, dst)


# --- Helper to build combined train and move split into val without touching data/train ---
def build_combined_and_val(input_config: str, exclude_classes: bool, include_config_classes_only: bool, console_print: bool, threshold: int, source_root: str, combined_train_dir: str, val_dir: str, min_total_per_class: int = 20, split_frac: float = 0.20):
    # Clean rebuild
    if os.path.exists(combined_train_dir):
        print(f"[CLEAN] Removing existing directory: {combined_train_dir}")
        shutil.rmtree(combined_train_dir)
    if os.path.exists(val_dir):
        print(f"[CLEAN] Removing existing directory: {val_dir}")
        shutil.rmtree(val_dir)

    os.makedirs(combined_train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Declare exclusion and inclusion lists.
    taxon_exclusion_list = []
    taxon_inclusion_list = []

    # If arguments that define subset of classes using taxa-config are present:
    # seek those files out and append those classes to the above lists
    if(include_config_classes_only or exclude_classes):
        if(os.path.isfile(input_config)):
            file = open(input_config, "r")
            line = file.readline()
            while line:
                # + icon defines classes we need to include
                if(include_config_classes_only and line[0] == "+"):
                    taxon_inclusion_list.append(line.strip("-+\n"))
                # - icon defines classes we need to exclude
                elif(exclude_classes and line[0] == "-"):
                    taxon_exclusion_list.append(line.strip("-+\n"))
                line = file.readline()
            file.close()
        else:
            print(f"[INFO] The file {input_config} does not live in the directory. Run utils/taxa_for_config.py to generate.")
            exit()
    
    if include_config_classes_only:
        print("\n[INFO] Including only the the following taxa for classification:")
        for name in taxon_inclusion_list:
            print(f"  - {name}")
    if exclude_classes:
        print("\n[INFO] Removing the following taxa from classification:")
        for name in taxon_exclusion_list:
            print(f"  - {name}")

    # Gather all images by class across owner-* folders
    class_to_paths = {}
    for owner in os.listdir(source_root):
        owner_path = os.path.join(source_root, owner)
        if not (os.path.isdir(owner_path) and owner.startswith("owner-")):
            continue
        for img_class in os.listdir(owner_path):
            img_class_path = os.path.join(owner_path, img_class)
            if not os.path.isdir(img_class_path):
                continue
            elif(exclude_classes and img_class.replace("taxon-","") in taxon_exclusion_list):
                continue
            elif(include_config_classes_only and img_class.replace("taxon-","") not in taxon_inclusion_list):
                continue
            imgs = [os.path.join(img_class_path, f) for f in os.listdir(img_class_path)
                    if f.lower().endswith(IMAGE_EXTS)]
            if imgs:
                class_to_paths.setdefault(img_class, []).extend(imgs)

    # Copy to combined, then move split to val
    for img_class, paths in sorted(class_to_paths.items()):
        total = len(paths)

        if (threshold != None) and (total > threshold):
            for delta in range(total - threshold):
                random_element = random.choice(paths)
                paths.remove(random_element)
            total = len(paths)

        if total < min_total_per_class:
            if (console_print):
                print(f"[SKIP] Class '{img_class}' has only {total} images (need ≥ {min_total_per_class})")
            continue
        elif (threshold != None) and (total < threshold):
            if (console_print):
                print(f"[SKIP] Class '{img_class}' is smaller than the {threshold} image threshold")
            continue

        random.shuffle(paths)
        train_img_class_dir = os.path.join(combined_train_dir, img_class)
        val_img_class_dir = os.path.join(val_dir, img_class)
        os.makedirs(train_img_class_dir, exist_ok=True)
        os.makedirs(val_img_class_dir, exist_ok=True)
        
        copied_files = []
        for src in paths:
            base = os.path.basename(src)
            dst = unique_path(os.path.join(train_img_class_dir, base))
            shutil.copy2(src, dst)
            copied_files.append(dst)
        
        move_k = int(len(copied_files) * split_frac)
        if move_k > 0:
            random.shuffle(copied_files)
            to_move = copied_files[:move_k]
            for p in to_move:
                dst = unique_path(os.path.join(val_img_class_dir, os.path.basename(p)))
                shutil.move(p, dst)
        if (console_print):
            print(f"[INFO] Class '{img_class}': total={total}, moved_to_val={move_k}, remaining_train={len(copied_files) - move_k}")

# Returns model, embedder, and optimizer for resnet models
def build_resnet_model(model_name: str, num_classes: int, use_pretrain: bool):
    model_name = model_name.lower() #case insensitve

    if model_name == "resnet18":
        model = models.resnet18(pretrained=use_pretrain)
        if use_pretrain:
            print("[INFO] Loaded pretrained resnet18")

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(f"[INFO] Updated final layer for {num_classes} classes")

        head_params = model.fc.parameters()
        optimizer = optim.Adam(head_params, lr=0.001)
        print("[INFO] Optimizer and loss function initialized")

        embedder = build_resnet_embedder(models.resnet18(pretrained=False), model)

        return model, embedder, optimizer

    if model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrain)
        if use_pretrain:
            print("[INFO] Loaded pretrained resnet50")

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(f"[INFO] Updated final layer for {num_classes} classes")

        head_params = model.fc.parameters()
        # build optimizer using head_params
        optimizer = optim.Adam(head_params, lr=0.001)
        print("[INFO] Optimizer and loss function initialized")

        # build embedder
        embedder = build_resnet_embedder(models.resnet50(pretrained=False), model)

        return model, embedder, optimizer

    if model_name == "resnet34":
        model = models.resnet34(pretrained=use_pretrain)
        if use_pretrain:
            print("[INFO] Loaded pretrained resnet34")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(f"[INFO] Updated final layer for {num_classes} classes")

        head_params = model.fc.parameters()
        optimizer = optim.Adam(head_params, lr=0.001)
        print("[INFO] Optimizer and loss function initialized")

        embedder = build_resnet_embedder(models.resnet34(pretrained=False), model)

        return model, embedder, optimizer

# Builds an embedder model for resnet structure
def build_resnet_embedder(embedder, model):
    embedder.fc = nn.Identity()
    state = model.state_dict()
    state_no_fc = {k: v for k, v in state.items() if not k.startswith('fc.')}
    missing, unexpected = embedder.load_state_dict(state_no_fc, strict=False)
    if missing:
        print(f"[INFO] Embedder missing keys: {missing}")
    if unexpected:
        print(f"[INFO] Embedder unexpected keys: {unexpected}")

    return embedder

def build_vgg16_model(use_pretrain: bool, num_of_target_classes: int):
    model = models.vgg16(pretrained=use_pretrain)
    if use_pretrain:
        print("[INFO] Loaded pretrained VGG16")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_of_target_classes)
    head_params = model.classifier[-1].parameters()

    optimizer = optim.Adam(head_params, lr=0.001)
    print("[INFO] Optimizer and loss function initialized")

    embedder = build_vgg16_embedder(models.vgg16(pretrained=False), model)

    return model, embedder, optimizer

# Builds embedder for VGG16
def build_vgg16_embedder(embedder, model):
    embedder.classifier[-1] = nn.Identity()
    state = model.state_dict()
    state_no_fc = {k: v for k, v in state.items() if not k.startswith(f'classifier.{6}.')}
    missing, unexpected = embedder.load_state_dict(state_no_fc, strict=False)
    if missing:
        print(f"[INFO] Embedder missing keys: {missing}")
    if unexpected:
        print(f"[INFO] Embedder unexpected keys: {unexpected}")

    return embedder


def build_densenet121_model(use_pretrain: bool, num_of_target_classes: int):
    model = models.densenet121(pretrained=use_pretrain)
    if use_pretrain:
        print("[INFO] Loaded pretrained densenet121")
    model.classifier = nn.Linear(model.classifier.in_features, num_of_target_classes)
    head_params = model.classifier.parameters()
    optimizer = optim.Adam(head_params, lr=0.001)
    print("[INFO] Optimizer and loss function initialized")

    embedder = build_densenet121_embedder(models.densenet121(pretrained=False), model)

    return model, embedder, optimizer

# Builds embedder for densenet121
def build_densenet121_embedder(embedder, model):
    embedder.classifier = nn.Identity()
    state = model.state_dict()
    state_no_fc = {k: v for k, v in state.items() if not k.startswith(f'classifier.{6}.')}
    missing, unexpected = embedder.load_state_dict(state_no_fc, strict=False)
    if missing:
        print(f"[INFO] Embedder missing keys: {missing}")
    if unexpected:
        print(f"[INFO] Embedder unexpected keys: {unexpected}")

    return embedder

def main():
    # Command-Line Arguments
    parser = argparse.ArgumentParser(description='Train fossil image classifier')
    parser.add_argument('--use-augmented', action='store_true', help='Use augmented dataset')
    parser.add_argument("--console-print", action='store_true', help="Print extra details to console")
    parser.add_argument('--use-pre-train', action='store_true', default=True, help='Use pre-trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument("--input-config", default="taxa-config.txt", help="Taxa file to guide for your augmentation")
    parser.add_argument('--model-path', type=str, default=None, help='Path to save model weights')
    parser.add_argument("--model", type=str, default='resnet18', choices=["resnet18", "resnet34", "resnet50", "vgg16", "densenet121"], help="Determines which model to train")
    parser.add_argument("--threshold", type=int, help="Generate class balance by defining a threshold that will remove classes if they do not an image count that exceeds this number. Randomly excise images from classes that exceed this number until they are equal to the threshold.")
    parser.add_argument("--exclude-classes", action='store_true', help="Remove select classes marked with an '-' from taxa-config")
    parser.add_argument("--include-config-classes-only", action='store_true', help="Include only classes in taxa-config and start with a '+'")
    args = parser.parse_args()

    # Location where model state will be stored
    if args.model_path:
        path_to_model = args.model_path
        #print(f"Model path supplied, model will be stored at:\n{path_to_model}")
    else:
        path_to_model = f"models/fossil_{args.model}.pt"
        #print(f"No model specified, will be stored at:\n{path_to_model}")

    # Apply seed as early as possible so all randomness is controlled
    set_seed(args.seed)

    # ---------------- Directory Setup ----------------
    # If using augmented, look for merged data in data/augmented/owner-combined
    # If not using augmented, still write the merged owner-combined into data/augmented/owner-combined,
    # but READ sources from data/train/owner-*
    if args.use_augmented:
        source_root = 'data/augmented'  # not used for merging below
        train_dir = os.path.join('data/augmented', 'owner-combined')
    else:
        source_root = 'data/train'  # where owner-* live
        train_dir = os.path.join('data/augmented', 'owner-combined')

    val_dir = 'data/val/owner-combined'

    # Rebuild owner-combined from scratch when using the original dataset
    # This merges all owner-* folders from data/train into data/augmented/owner-combined
    # and then MOVES floor (N/4) to data/val/owner-combined to avoid leakage.
    if not args.use_augmented:
        print(f"[INFO] Building combined train and val from {source_root}")
        build_combined_and_val(
            input_config=args.input_config,
            exclude_classes=args.exclude_classes,
            include_config_classes_only=args.include_config_classes_only,
            console_print=args.console_print,
            threshold=args.threshold,
            source_root=source_root,
            combined_train_dir=train_dir,
            val_dir=val_dir,
            min_total_per_class=20,  # Eligibility threshold
            split_frac=0.25  # Move floor(N/4) to val
        )

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

    # ImageNet statistics used for pretraining
    if args.model in ['resnet18', 'resnet34', 'resnet50', 'densenet121']:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
    elif args.model in ['vgg16']:
        IMAGENET_MEAN = [0.48235, 0.45882, 0.40784]
        IMAGENET_STD = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]

    # Transforms
    # These resize and normalize images to the expected format
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

    # Load Data, the training set comes from data/augmented/owner-combined not the original training folders.
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

    if (args.console_print):
        print("[INFO] Class counts:", class_counts.tolist())
        print("[INFO] Class weights:", class_weights.tolist())

    # Save class names next to the model path for later use by the predictor
    os.makedirs(os.path.dirname(path_to_model) or ".", exist_ok=True)
    with open(os.path.join(os.path.dirname(path_to_model) or ".", "class_names.json"), "w") as f:
        json.dump(train_data.classes, f)

    # Pick GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.console_print):
        print(f"[INFO] Using device: {device}")
        if device.type == 'cuda':
            print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] GPU not detected - training on CPU will be slow.")

    # Build a model, embedder, and optimizer based on the model argument
    if args.model in ['vgg16']:
        model, embedder, optimizer = build_vgg16_model(args.use_pre_train, len(class_counts))
    elif args.model in ['densenet121']:
        model, embedder, optimizer = build_densenet121_model(args.use_pre_train, len(class_counts))
    elif args.model in  ['resnet18', 'resnet34', 'resnet50']:
        model, embedder, optimizer = build_resnet_model(args.model, len(class_counts), args.use_pre_train)
    
    # Move model to device
    model = model.to(device)
    print("[INFO] Model moved to device")

    # Move embedder to device
    embedder = embedder.to(device).eval()
    print("[INFO] Embedder moved to device")

    # Use weighted cross entropy to handle class imbalance
    if (args.model in ['densenet121']):
        criterion = nn.CrossEntropyLoss()
    elif (args.model in ['resnet18', 'resnet34', 'resnet50', 'vgg16']):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
   
    # Train only the final layer for now to keep things simple and fast

    print("[INFO] Starting training loop...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        print(f"\n[INFO] Starting Epoch {epoch + 1}/{args.epochs}...")

        # Wrap the loader with a progress bar for live feedback
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")

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
        print(f"[INFO] Epoch {epoch + 1} Complete | "
              f"Loss: {running_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Time: {int(epoch_time // 60)}m {int(epoch_time % 60)}s")

    # Save trained weights so you can load them later for prediction
    torch.save(model.state_dict(), path_to_model)
    print(f"[INFO] Model saved to {path_to_model}")

if __name__ == "__main__":
    main()