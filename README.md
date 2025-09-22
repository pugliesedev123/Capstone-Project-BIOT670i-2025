# Using Deep Learning to Identify Fossils of the Atlantic Coastal Plain

In this project, we train neural networks to classify fossil images from across the Eastern United States, spanning the Cretaceous beds of New Jersey to the diverse coastal deposits of Maryland, the Carolinas, and Florida.  

We provide a Python toolkit for:
- **Image preprocessing and augmentation** (crop, pad, rotate, zoom, flips, etc.)
- **ResNet-18 training** with class imbalance handling
- **Prediction** on new images with nearest-neighbor lookups to training examples  

The goal is a practical framework for researchers and fossil enthusiasts alike.

---

# Fossil Image Preprocessing & Training

This repository contains scripts to preprocess fossil image datasets, train a ResNet-18 classifier, and run predictions on new fossil images or folders.  
Each stage is seed-controlled for reproducibility.

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
pip install torch torchvision tqdm pillow numpy
```

All other imports are from the Python standard library (`os`, `shutil`, `argparse`, etc.).

---

## Dataset Layout

Place images in **owner** and their sub-**taxon** folders:

```
data/train/
  owner-Wikimedia/
    taxon-scapanorhynchus_texanus/
    taxon-cretolamna_appendiculata/
    taxon-belemnitella_americana/
    taxon-exogyra_sp/
  owner-author/
    taxon-isurus_sp/
    mineral-pyrite_nodule/
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
- Build `data/val/owner-combined` (validation set, ~20% per class)  
- Create `data/augmented/owner-combined` with augmented variants per image (configurable with `--aug-per-image`)  

### Arguments

The script accepts several arguments to control how preprocessing and augmentation are performed:

- `--input-root` (default: `data/train`) Path to the root folder containing the **owner-** subdirectories. Each owner folder should contain one or more `taxon-*` or `mineral-*` subfolders with images.
- `--val-root` (default: `data/val/owner-combined`) Path where the script will copy validation images. A new folder is created for each class, and 1/5 of the images from that class are stored here.  
- `--aug-root` (default: `data/augmented/owner-combined`) Path where augmented training images will be saved.   Each class will have its own subfolder with augmented versions of the training images.  
- `--aug-per-image` (default: `3`) Number of augmented samples to create for each original training image. Example: 100 training images with `--aug-per-image=3` produce 300 augmented images.  
- `--seed` (default: `42`) Random seed for shuffling, splitting, and augmentation. Use the same seed to reproduce the same validation split and augmented dataset.  

### Notes

- Classes with fewer than **20 images** are skipped to avoid poor-quality splits.  
- Validation images are copied unchanged, while training images are **augmented** using a mix of geometric and appearance transforms (rotation, scaling, flips, sharpness, grayscale, histogram equalization).  
- Outputs are placed under the `--val-root` and `--aug-root` directories in a layout compatible with PyTorch’s `ImageFolder`.  

---

## Train the Model

Train a ResNet-18 classifier on your dataset:

```bash
# Train on raw data (merges data/train/owner-* into data/augmented/owner-combined first)
python train_model.py

# Train on prebuilt augmented data
python train_model.py --use-augmented
```

What happens:
- Owner folders are merged into **`data/augmented/owner-combined`** when `--use-augmented` is **not** set (sources read from `data/train/owner-*`)  
- A validation split is created if missing: `data/val/owner-combined`  
- Pretrained ResNet-18 is fine-tuned for your fossil taxa  
- Class imbalance is addressed with **per-class loss weights**  
- Model weights, class names, and a **feature embedding index** are saved in `models/`

### Arguments

- `--use-augmented` (flag) If set, read training data from `data/augmented/owner-combined`. If not set, the script **builds** `data/augmented/owner-combined` by merging `data/train/owner-*` into that location before training.
- `--seed` (default: `42`) Controls shuffling, splits, and other randomness via a unified seed helper.
- `--batch-size` (default: `16`) Training and validation batch size.
- `--epochs` (default: `5`) Number of training epochs.
- `--model-path` (default: `models/fossil_resnet18.pt`) Where to save the trained model weights.
- `--index-path` (default: `models/train_index.pt`) Where to save the training **embedding index** (512-d features + labels + file paths).

---

## Outputs

After preprocessing and training, you will have:

- `models/class_names.json` — list of class labels  
- `models/fossil_resnet18.pt` — trained ResNet-18 weights  
- `models/train_index.pt` — embedding index of training images (for nearest neighbors)  

---

## Seeds and Reproducibility

All scripts accept a `--seed` argument to make results more repeatable:

```bash
python train_model.py --seed 123
python augment_images.py --seed 123
```

The seed affects:
- Dataset shuffling and validation split  
- Random data augmentations (rotations, flips, zoom, etc.)  
- Torch model initialization  

> Note: Exact reproducibility across machines/GPUs may require deterministic backends (e.g., setting `torch.backends.cudnn.deterministic = True` and disabling benchmark). Your training script includes these controls in `set_seed`.

---

## Predict on New Images or Folders

After training, classify new fossils using `predict_image.py`.

### Predict on a folder of example images

```bash
python predict_image.py   --example-dir example_images   --top-predictions 3   --neighbors 3   --model-path models/fossil_resnet18.pt   --class-names models/class_names.json   --index-path models/train_index.pt   --output-dir output
```

### Arguments

- `--example-dir` Path to the folder containing example images (recursively processed).  
- `--top-predictions` Number of top predictions to record (default: 3).  
- `--neighbors` Number of nearest neighbors from the training set to include (default: 3).  
- `--model-path` Path to the model weights `.pt` file.  
- `--class-names` Path to `class_names.json`.  
- `--index-path` Path to training embedding index `.pt`.  
- `--output-dir` Directory where the CSV of predictions will be saved.  

### Example output

```
example_images/shark_tooth.png
  1. Scapanorhynchus texanus (87.32%)
  2. Cretolamna appendiculata (10.15%)
  3. Odontaspis vertebrosa (2.53%)

Nearest neighbors:
  - Scapanorhynchus texanus | 0.984231 | data/train/...
  - Scapanorhynchus texanus | 0.981027 | data/train/...
  - Cretolamna appendiculata | 0.923876 | data/train/...

example_images/bivalve.png
  1. Ostrea cf. congesta (62.11%)
  2. Pycnodonte vesicularis (21.40%)
  3. Exogyra costata (12.77%)
```

The script also writes predictions and nearest neighbor results to a timestamped CSV in the specified `--output-dir`.

### Why nearest neighbors matter

Alongside top class predictions, the script also shows **nearest neighbors** from the training dataset based on feature similarity.  
This helps you:
- See which training fossils most influenced the model’s decision  
- Spot potential misclassifications (if the nearest examples look wrong)  
- Build confidence in the prediction by comparing to actual known fossils  

In short, nearest neighbors provide **transparent, human-checkable evidence** for each classification.  

---

## Recommended Workflow

1. **Convert HEIC photos (if any) → `convert_heics.py`**  
   Standardize image formats so everything is usable by the training scripts.  
2. **Augment dataset and build validation split → `augment_images.py`**  
   Expand training data with transformations and create a proper validation set.  
3. **Train model → `train_model.py`**  
   Fine-tune a ResNet-18, saving weights, class labels, and training embeddings.  
4. **Predict on new or example fossils → `predict_image.py`**  
   Run the trained model on fresh images to get top predictions and closest training examples.  
