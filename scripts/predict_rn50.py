#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def is_image(filename):
    extensions = (".png", ".jpg", ".jpeg")
    return filename.lower().endswith(extensions)


def load_embedder_from_classifier(classifier_state, device):
    """
    Build a copy of ResNet-50 that outputs features instead of class scores.
    """
    embedder = models.resnet50(weights=None)
    embedder.fc = nn.Identity()  # Replace the last layer so outputs are 512-length features

    # Remove the classifier weights from the trained state dict
    # We keep everything except the last layer named "fc.*"
    state_no_fc = {k: v for k, v in classifier_state.items() if not k.startswith("fc.")}
    embedder.load_state_dict(state_no_fc, strict=False)

    embedder = embedder.to(device).eval()
    return embedder


def main():
    # Command-Line Arguments
    parser = argparse.ArgumentParser(description="Predict fossil classes for images in a folder (recursively).")
    parser.add_argument("--example-dir", required=True,
                        help="Path to folder containing example images (processed recursively).")
    parser.add_argument("--top-predictions", type=int, default=3, help="How many top guesses to record for each image.")
    parser.add_argument("--neighbors", type=int, default=3, help="How many closest training images to record.")
    parser.add_argument("--model-path", default="models/fossil_resnet50.pt",
                        help="Path to the trained model weights file.")
    parser.add_argument("--class-names", default="models/class_names.json", help="Path to class_names.json file.")
    parser.add_argument("--index-path", default="models/train_index.pt",
                        help="Path to the saved training feature index.")
    parser.add_argument("--output-dir", default="output", help="Folder where the CSV will be saved.")
    args = parser.parse_args()

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
    model = models.resnet50(weights=None)
    # Replace the last layer to have the right number of classes
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    # Load trained weights
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()  # eval mode means no training behavior like dropout

    # Build an embedder that gives feature vectors instead of class scores
    embedder = load_embedder_from_classifier(state, device)

    # Load the saved training feature index
    index_obj = torch.load(args.index_path, map_location="cpu")
    index_embeddings = index_obj["embeddings"].float()  # shape [N, 512], N is number of training images
    index_labels = index_obj["labels"].long()  # class id for each embedding
    index_paths = index_obj["paths"]  # file path for each training image
    idx_to_class = index_obj.get("idx_to_class", None)
    if idx_to_class is None:
        # If not stored, build it from the inverse of class_to_idx
        class_to_idx = index_obj["class_to_idx"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # shorter side to 256
        transforms.CenterCrop(224),  # cut a 224x224 square from center
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Make sure the output folder exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Build a file name for the CSV that includes the input folder name and a timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(os.path.abspath(args.example_dir)) or "examples"
    csv_path = os.path.join(args.output_dir, f"predictions_{base_name}_{ts}.csv")

    header = [
        "filename",
        "parent_folder",
    ]

    # Columns for top class predictions
    for i in range(1, args.top_predictions + 1):
        header += [
            f"class_prediction_{i}",
            f"class_prediction_{i}_confidence_percentage"
        ]

    # Columns for nearest neighbor results
    for k in range(1, args.neighbors + 1):
        header += [
            f"nearest_neighbor_{k}_label",
            f"nearest_neighbor_{k}_cosine_similarity",
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
                logits = model(x)  # raw scores for each class
                probs = F.softmax(logits, dim=1)  # turn scores into probabilities

                # Pick the top K classes
                top_probs, top_indices = torch.topk(probs, args.top_predictions)
                top_probs = top_probs[0].cpu().numpy()
                top_indices = top_indices[0].cpu().numpy()

                rel_path_for_print = os.path.relpath(img_path, args.example_dir)
                print(f"\n{rel_path_for_print}:")

                # Start the CSV row with file name and parent folder name
                row = [file, parent_folder]

                # Add the top K class predictions to the row and print them
                for i in range(args.top_predictions):
                    predicted_class = class_names[top_indices[i]]  # map index to class name
                    confidence_pct = float(top_probs[i] * 100.0)  # show as percent
                    print(f"{i + 1}. {predicted_class} ({confidence_pct:.2f}% confidence)")
                    row += [predicted_class, f"{confidence_pct:.2f}"]

                # 2) Find nearest neighbors from the training index
                # First get features for the query image using the embedder
                q = embedder(x).squeeze(0)  # shape [512]
                q = F.normalize(q, dim=0)  # make length 1 so cosine works as dot

                # Compute cosine similarity to all training features in the index
                # This is a dot product since both sides are normalized
                sims = torch.matmul(index_embeddings, q.cpu())  # shape [N]
                # Pick the top K closest matches
                vals, inds = torch.topk(sims, k=min(args.neighbors, sims.numel()), largest=True)

                # Add neighbor info to the row
                for s, i_idx in zip(vals.tolist(), inds.tolist()):
                    lbl_idx = int(index_labels[i_idx])  # class id of the neighbor
                    lbl = idx_to_class.get(lbl_idx, str(lbl_idx))  # class name of the neighbor
                    path = index_paths[i_idx]  # file path to the neighbor image
                    row += [lbl, f"{s:.6f}", path]

                rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved predictions to: {csv_path}")


if __name__ == "__main__":
    main()