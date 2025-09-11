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


def main():
    parser = argparse.ArgumentParser(description="Predict fossil classes for images in a folder (recursively).")
    parser.add_argument("--example-dir", help="Path to folder containing example images (processed recursively).")
    parser.add_argument("--top-predictions", type=int, default=3, help="Number of top predictions to record.")
    parser.add_argument("--model-path", default="models/fossil_resnet18.pt", help="Path to the model weights .pt file.")
    parser.add_argument("--class-names", default="models/class_names.json", help="Path to class_names.json.")
    parser.add_argument("--output-dir", default="output", help="Folder where the CSV will be saved.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    with open(args.class_names, "r") as f:
        class_names = json.load(f)
    if isinstance(class_names, dict):
        try:
            class_names = [class_names[str(i)] for i in range(len(class_names))]
        except Exception:
            class_names = list(class_names.values())

    # Load model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Prepare output CSV
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(os.path.abspath(args.example_dir)) or "examples"
    csv_path = os.path.join(args.output_dir, f"predictions_{base_name}_{ts}.csv")

    # CSV header
    header = [
        "Filename",
        "Parent_Folder",
    ]
    for i in range(1, args.top_predictions + 1):
        header += [f"Class_Prediction{i}", f"Class_Prediction{i}_confidence_percentage"]

    rows = []

    # Walk through all subfolders
    if not os.path.isdir(args.example_dir):
        raise FileNotFoundError(f"Input folder not found: {args.example_dir}")

    with torch.no_grad():
        for root, _, files in os.walk(args.example_dir):
            parent_folder = os.path.basename(root)

            image_files = [f for f in files if is_image(f)]
            if not image_files:
                continue

            for file in sorted(image_files):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Skipping {img_path}: failed to open ({e})")
                    continue

                inp = transform(img).unsqueeze(0).to(device)
                output = model(inp)
                probabilities = F.softmax(output, dim=1)

                top_probs, top_indices = torch.topk(probabilities, args.top_predictions)
                top_probs = top_probs[0].cpu().numpy()
                top_indices = top_indices[0].cpu().numpy()

                # Print to terminal
                rel_path_for_print = os.path.relpath(img_path, args.example_dir)
                print(f"\n{rel_path_for_print}:")
                row = [file, parent_folder]

                for i in range(args.top_predictions):
                    predicted_class = class_names[top_indices[i]]
                    confidence_pct = float(top_probs[i] * 100.0)
                    print(f"{i+1}. {predicted_class} ({confidence_pct:.2f}% confidence)")
                    row += [predicted_class, f"{confidence_pct:.2f}"]

                rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved predictions to: {csv_path}")


if __name__ == "__main__":
    main()
