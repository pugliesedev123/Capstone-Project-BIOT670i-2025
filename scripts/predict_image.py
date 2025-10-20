#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def is_image(filename):
    extensions = (".png", ".jpg", ".jpeg")
    return filename.lower().endswith(extensions)

# Updated using ChatGPT
def load_embedder_from_classifier(model_path_str, classifier_state, device):
    # pick arch and set Identity at the correct place

    if "vgg16" in model_path_str:
        embedder = models.vgg16(weights=None)
        embedder.classifier[6] = nn.Identity()     # 4096-d features
        drop_prefixes = ("classifier.6.",)
    elif "densenet121" in model_path_str:
        embedder = models.densenet121(weights=None)
        embedder.classifier = nn.Identity()        # 1024-d features
        drop_prefixes = ("classifier.",)
    elif "resnet18" in model_path_str:
        embedder = models.resnet18(weights=None)
        embedder.fc = nn.Identity()                # 512-d features
        drop_prefixes = ("fc.",)
    elif "resnet34" in model_path_str:
        embedder = models.resnet34(weights=None)
        embedder.fc = nn.Identity()
        drop_prefixes = ("fc.",)
    elif "resnet50" in model_path_str:
        embedder = models.resnet50(weights=None)
        embedder.fc = nn.Identity()                # 2048-d features
        drop_prefixes = ("fc.",)

    # unwrap and strip DDP prefix
    state = classifier_state.get("state_dict", classifier_state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # drop the head weights for this arch
    keep = {k: v for k, v in state.items() if not any(k.startswith(p) for p in drop_prefixes)}

    embedder.load_state_dict(keep, strict=False)
    return embedder.to(device).eval()



def main():

    # Command-Line Arguments
    parser = argparse.ArgumentParser(description="Predict fossil classes for images in a folder (recursively).")
    parser.add_argument("--example-dir", required=True, help="Path to folder containing example images (processed recursively).")
    parser.add_argument("--console-print", action='store_true', help="Print extra details to console")
    parser.add_argument("--top-predictions", type=int, default=3, help="How many top guesses to record for each image.")
    parser.add_argument("--neighbors", type=int, default=3, help="How many closest training images to record.")
    parser.add_argument("--model-path", default="models/fossil_resnet18.pt", help="Path to the trained model weights file.")
    parser.add_argument("--class-names", default="models/class_names.json", help="Path to class_names.json file.")
    parser.add_argument("--index-path", default="", help="Path to the saved training feature index.")
    parser.add_argument("--output-dir", default="output", help="Folder where the CSV will be saved.")
    args = parser.parse_args()

    args.index_path = Path(args.index_path) if args.index_path else Path("")
    has_index = args.index_path.is_file()


    # Pick GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.class_names, "r") as f:
        class_names = json.load(f)
    if isinstance(class_names, dict):
        try:
            class_names = [class_names[str(i)] for i in range(len(class_names))]
        except Exception:
            class_names = list(class_names.values())

    # Build the classifier model shape to match training
    if (args.model_path == 'models/fossil_resnet18.pt'):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif (args.model_path == 'models/fossil_resnet34.pt'):
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif (args.model_path == 'models/fossil_resnet50.pt'):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif (args.model_path == 'models/fossil_vgg16.pt'):
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(class_names))
    elif (args.model_path == 'models/fossil_densenet121.pt'):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(class_names))

    # Generated fix using ChatGPT
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Load checkpoint once
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # 1) Classifier: DO NOT drop the head
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Make sure the classifier runs in eval mode on the right device
    model = model.to(device).eval()

    # 2) Embedder: use a copy with the head removed
    state_for_embedder = dict(state)  # shallow copy is fine for Tensors

    if "resnet" in args.model_path:
        state_for_embedder.pop("fc.weight", None); state_for_embedder.pop("fc.bias", None)
    elif "vgg16" in args.model_path:
        state_for_embedder.pop("classifier.6.weight", None); state_for_embedder.pop("classifier.6.bias", None)
    elif "densenet121" in args.model_path:
        state_for_embedder.pop("classifier.weight", None); state_for_embedder.pop("classifier.bias", None)

    embedder = load_embedder_from_classifier(args.model_path, state_for_embedder, device)


    if(has_index):
        # Load the saved training feature index
        index_obj = torch.load(args.index_path, map_location="cpu")
        index_embeddings = index_obj["embeddings"].float()  # shape [N, 512], N is number of training images
        index_labels = index_obj["labels"].long()           # class id for each embedding
        index_paths = index_obj["paths"]                    # file path for each training image
        idx_to_class = index_obj.get("idx_to_class", None)
        if idx_to_class is None:
            # If not stored, build it from the inverse of class_to_idx
            class_to_idx = index_obj["class_to_idx"]
            idx_to_class = {v: k for k, v in class_to_idx.items()}

    if "resnet" in args.model_path or "densenet121" in args.model_path:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
    elif "vgg16" in args.model_path:
        IMAGENET_MEAN = [0.48235, 0.45882, 0.40784]
        IMAGENET_STD = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),    # shorter side to 256
        transforms.CenterCrop(224),       # cut a 224x224 square from center
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Make sure the output folder exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Build a file name for the CSV that includes the input folder name and a timestamp
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = os.path.basename(os.path.abspath(args.example_dir)) or "examples"
    csv_path = os.path.join(args.output_dir, f"predictions_{args.model_path.replace("models/fossil_","").replace(".pt","")}_{base_name}_{ts}.csv")

    header = [
        "filename",
        "parent_folder",
    ]

     # Columns for top class predictions
    for i in range(1, args.top_predictions + 1):
        header += [
            f"class_prediction_{i}",
            f"class_prediction_{i}_confidence_percentage",
            f"class_prediction_{i}_IsAccurate"
        ]

    if(has_index):
        # Columns for nearest neighbor results
        for k in range(1, args.neighbors + 1):
            header += [
                f"nearest_neighbor_{k}_label",
                f"nearest_neighbor_{k}_cosine_similarity", # Must be able to explain this value
                f"nearest_neighbor_{k}_path"
            ]
    rows = []

    if not os.path.isdir(args.example_dir):
        raise FileNotFoundError(f"Input folder not found: {args.example_dir}")

    with torch.no_grad():
        for root, _, files in os.walk(args.example_dir):
            parent_folder = os.path.basename(root)

            # Keep only image files
            image_files = [f for f in files if is_image(f)]
            if not image_files:
                continue

            # Process each image in sorted order for stable output
            for file in sorted(image_files):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")  # ensure 3 channels
                except Exception as e:
                    print(f"Skipping {img_path}: failed to open ({e})")
                    continue

                # Prepare the image for the model
                x = transform(img).unsqueeze(0).to(device)  # add batch dimension

                # 1) Classify the image
                logits = model(x)                   # raw scores for each class
                probs = F.softmax(logits, dim=1)    # turn scores into probabilities

                # Pick the top K classes
                top_probs, top_indices = torch.topk(probs, args.top_predictions)
                top_probs = top_probs[0].cpu().numpy()
                top_indices = top_indices[0].cpu().numpy()

                if(args.console_print):
                    rel_path_for_print = os.path.relpath(img_path, args.example_dir)
                    print(f"\n{rel_path_for_print}:")

                # Start the CSV row with file name and parent folder name
                row = [file, parent_folder]

                # Add the top K class predictions to the row and print them
                for i in range(args.top_predictions):
                    
                    predicted_class = class_names[top_indices[i]]   # map index to class name
                    
                    confidence_pct = float(top_probs[i] * 100.0)    # show as percent

                    class_accurate = ""
                    if predicted_class.replace("taxon-","") == (parent_folder.replace("taxon-","") or parent_folder):
                        class_accurate = "Yes"
                    else:
                        class_accurate = "No"

                    if(args.console_print):
                        print(f"{i+1}. {predicted_class} ({confidence_pct:.2f}% confidence)")
                    row += [predicted_class, f"{confidence_pct:.2f}", class_accurate]

                # 2) Find nearest neighbors from the training index
                # First get features for the query image using the embedder
                q = embedder(x).squeeze(0)            # shape [512]
                q = F.normalize(q, dim=0)             # make length 1 so cosine works as dot

                # Compute cosine similarity to all training features in the index
                # This is a dot product since both sides are normalized
                if(has_index):
                    sims = torch.matmul(index_embeddings, q.cpu())  # shape [N]
                    # Pick the top K closest matches
                    vals, inds = torch.topk(sims, k=min(args.neighbors, sims.numel()), largest=True)

                    # Add neighbor info to the row
                    for s, i_idx in zip(vals.tolist(), inds.tolist()):
                        lbl_idx = int(index_labels[i_idx])          # class id of the neighbor
                        lbl = idx_to_class.get(lbl_idx, str(lbl_idx))  # class name of the neighbor
                        path = index_paths[i_idx]                   # file path to the neighbor image
                        row += [lbl, f"{s:.6f}", path]

                rows.append(row)


    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved predictions to: {csv_path}")


if __name__ == "__main__":
    main()