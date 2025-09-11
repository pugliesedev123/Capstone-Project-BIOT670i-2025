# Using Deep Learning to Identify Fossils of the Atlantic Coastal Plain

In this project, we train neural networks to classify fossil images from across the Eastern United States, spanning the Cretaceous beds of New Jersey to the diverse coastal deposits of Maryland, the Carolinas, and Florida.  

We provide a Python toolkit for image preprocessing and augmentation, ResNet-18 training, and evaluating or predicting on new images. The goal is a practical framework for researchers and fossil enthusiasts alike.

---

# Fossil Image Preprocessing & Training

This repository contains scripts to preprocess fossil image datasets (cropping, padding, augmentation), train a ResNet-18 classifier, and run predictions on new fossil images or folders.

---

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# (Optional) create a virtual environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

# Install required libraries
pip install torch torchvision tqdm pillow
```

All other imports are from the Python standard library (`os`, `shutil`, `argparse`, etc.).

---

## Dataset Layout

Place images in **owner** and their sub-**taxon** folders:

```
data/train/
  owner-Wikimedia/
    taxon-Scapanorhynchus_texanus/
    taxon-Cretolamna_appendiculata/
    taxon-Belemnitella_americana/
    taxon-Ostrea_cf_congesta/
  owner-author/
    taxon-Cliona_sp/
    mineral-Pyrite_Nodule/
```

Each `owner-*` folder contains multiple taxon or mineral folders.  
Each taxon folder contains `.jpg`, `.jpeg`, or `.png` images.

---

## Utilities

### Convert HEIC images

If your dataset includes `.heic` images (e.g., from iPhones), use `convert_heics.py`:

```bash
python convert_heics.py --target-dir data/train
```

This converts all `.heic` files under `data/train` into `.jpg` files.

---

## Preprocess and Augment Images

Run the augmentation script to build a validation split and augmented data:

```bash
python augment_images.py
```

This will:
- Build `data/val/owner-combined` (validation set)
- Create `data/augmented/owner-combined` with 20 augmented variants per image

---

## Train the Model

Train a ResNet-18 classifier on your dataset:

```bash
# Train on raw data
python train_model.py

# Train on augmented data
python train_model.py --use-augmented
```

What happens:
- Owner folders are merged into `owner-combined`
- Validation split is created if missing
- Pretrained ResNet-18 is fine-tuned for your fossil taxa
- Model weights and class names are saved in `models/`

---

## Outputs

After preprocessing and training, you will have:

- `models/class_names.json` — list of class labels  
- `models/fossil_resnet18.pt` — trained ResNet-18 weights 

---

## Predict on New Images or Folders

After training, classify new fossils using `predict_image.py`.

### Predict on a folder of example images

```bash
python predict_image.py \\
  --example-dir example_images \\
  --topn 3 \\
  --model-path models/fossil_resnet18.pt \\
  --class-names models/class_names.json \\
  --output-dir output
```

### Arguments

- `--example-dir` Path to the folder containing example images (recursively processed).  
- `--top-predictions` Number of top predictions to record (default: 3).  
- `--model-path` Path to the model weights `.pt` file (default: `models/fossil_resnet18.pt`).  
- `--class-names` Path to `class_names.json` (default: `models/class_names.json`).  
- `--output-dir` Directory where the CSV of predictions will be saved (default: `output`).  

### Example output

```
example_images/shark_tooth.png
  1. Scapanorhynchus texanus (87.32%)
  2. Cretolamna appendiculata (10.15%)
  3. Odontaspis vertebrosa (2.53%)

example_images/bivalve.png
  1. Ostrea cf. congesta (62.11%)
  2. Pycnodonte vesicularis (21.40%)
  3. Exogyra costata (12.77%)
```

The script also writes predictions to a timestamped CSV in the specified `--output-dir`.

---

## Recommended Workflow

1. **Convert HEIC photos (if any) → `convert_heics.py`**  
   Standardize image formats so everything is usable by the training scripts.
2. **Augment dataset and build validation split → `augment_images.py`**  
   Expand training data with image transformations and create a proper validation set.  
3. **Train model → `train_model.py`**  
   Fine-tune a ResNet-18 on your fossil images, saving weights and class labels.  
4. **Predict on new or example fossils → `predict_image.py`**  
   Run the trained model on fresh images to get top predictions with confidence scores.  


---

## Example Fossil Taxa Used

The dataset currently includes fossils such as:

- **Scapanorhynchus texanus** — Goblin shark tooth (Cretaceous, NJ)  
- **Cretolamna appendiculata** — Mackerel shark tooth (Cretaceous–Paleogene, NJ/MD)  
- **Belemnitella americana** — Belemnite guard (Cretaceous, NJ)

These examples will be extended as the dataset grows.

---
