# Using Deep Learning to Identify Fossils of the Atlantic Coastal Plain

In this project, we train neural networks to classify fossil images from across the Eastern United States, spanning the Cretaceous beds of New Jersey to the diverse coastal deposits of Maryland, the Carolinas, and Florida.  

We provide a Python toolkit for:
- **Image preprocessing and augmentation** (crop, colourize, rotate, zoom, flips, etc.)
- **ResNet-18, ResNet-34, ResNet-50, VGG16, and DenseNet121 training** with class imbalance handling
- **Prediction** on new images with support for nearest-neighbor lookups on ResNet training examples  

The goal is a practical framework for researchers and fossil enthusiasts alike.

# Fossil Image Preprocessing & Training

This repository contains scripts to preprocess fossil image datasets, train an image classifier, and run predictions on new fossil images or folders.  
Each stage is seed-controlled for reproducibility.

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
pip install torch torchvision tqdm pillow numpy psutil
```

All other imports are from the Python standard library (`os`, `shutil`, `argparse`, etc.).

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

## Utilities

### Convert HEIC images

If your dataset includes `.heic` images (e.g., from iPhones), use `convert_heics.py`:

```bash
python convert_heics.py --target-dir data/train
```

This converts all `.heic` files under `data/train` into `.jpg` files.

### Get File List

If you need a report on your dataset and clean it of empty taxon folders, use `get_file_list.py`:

```bash
python get_file_list.py
```

### Update File Names

If you need to update file names to match their taxon folder definition, use `update_file_name.py`:

```bash
python update_file_name.py
```

### Generate a taxa-config File

If you wish to generate a taxa-config file that can modify which classes you seek to classify, use `taxa_for_config.py`:

```bash
python taxa_for_config.py
```

This traverses all files in the training folder and returns a taxa-config.txt file that can be used in augmentation and training.

## Preprocess and Augment Images

Run the augmentation script to build a validation split and augmented data:

```bash
python augment_images.py
```

This will:
- Build `data/val/owner-combined` (validation set)  
- Create `data/augmented/owner-combined` with augmented variants per image (configurable with `--aug-per-image`)  

### Arguments

The script accepts several arguments to control how preprocessing and augmentation are performed:

- `--input-root` (default: `data/train`) Path to the root folder containing the **owner-** subdirectories. Each owner folder should contain one or more `taxon-*` or `mineral-*` subfolders with images.
- `--val-root` (default: `data/val/owner-combined`) Path where the script will copy validation images. A new folder is created for each class, and 1/5 of the images from that class are stored here.  
- `--aug-root` (default: `data/augmented/owner-combined`) Path where augmented training images will be saved.   Each class will have its own subfolder with augmented versions of the training images.  
- `--aug-per-image` (default: `3`) Number of augmented samples to create for each original training image. Example: 100 training images with `--aug-per-image=3` produce 300 augmented images.  
- `--seed` (default: `42`) Random seed for shuffling, splitting, and augmentation. Use the same seed to reproduce the same validation split and augmented dataset.  
- `--console-print` Print extra details to console.
- `--exclude-classes` Boolean to remove select classes marked with an '-' from taxa-config.txt.
- `--include-config-classes-only` Boolean to include only classes in taxa-config.txt and start with a '+'.
- `--threshold` Generate class balance by defining a threshold that will remove classes if they do not an image count that exceeds this number. Randomly excise images from classes that exceed this number until they are equal to the threshold.
- `--disable-tf` Disable specific transformations for augmentation. The following transforms can be disabled by repeatedly calling the argument: rotate, scale, zoom, horizontalflip, verticalflip, grayscale, equalize, sharpen.
- `--disable-ca` Disable specific class for augmentation (eg. --disable-ca exogyra_sp).

### Notes

- Classes with fewer than **20 images** are skipped to avoid poor-quality splits.  
- Validation images are copied unchanged, while training images are **augmented** using a mix of geometric and appearance transforms (rotation, scaling, flips, sharpness, grayscale, histogram equalization).  
- Outputs are placed under the `--val-root` and `--aug-root` directories in a layout compatible with PyTorch’s `ImageFolder`.

## Train the Model

Train a classifier on your dataset:

```bash
# Train on raw data (merges data/train/owner-* into data/augmented/owner-combined first)
python train_model.py

# Train on prebuilt augmented data
python train_model.py --use-augmented
```

What happens:
- Owner folders are merged into **`data/augmented/owner-combined`** when `--use-augmented` is **not** set (sources read from `data/train/owner-*`)  
- A validation split is created if missing: `data/val/owner-combined`  
- Pretrained model is fine-tuned for your fossil taxa  
- Class imbalance is addressed with **per-class loss weights**  
- Model weights, class names, and a **feature embedding index** are saved in `models/`

### Arguments

- `--use-augmented` (flag  If set, read training data from `data/augmented/owner-combined`. If not set, the script **builds** `data/augmented/owner-combined` by merging `data/train/owner-*` into that location before training.
- `--console-print` (flag) If set, prints extra details to the console during setup and training.
- `--use-pre-train` (default: `True`) Load ImageNet-pretrained weights for the selected backbone before training.
- `--seed` (default: `42`) Controls shuffling, splits, and other randomness via a unified seed helper.
- `--batch-size` (default: `16`) Training and validation batch size.
- `--epochs` (default: `5`) Number of training epochs.
- `--input-config` (default: `taxa-config.txt`) Path to the taxa config used to guide class inclusion and augmentation.
- `--model-path` (default: `models/fossil_resnet18.pt`) Where to save the trained model weights.
- `--index-path` (default: `models/train_index.pt`) Where to save the training **embedding index** (512-d features + labels + file paths).
- `--model` (default: `resnet18`, choices: `resnet18`, `resnet34`, `resnet50`, `vgg16`, `densenet121`) Select the model architecture to train.
- `--threshold` (integer) If set, cap each class at this image count. Randomly remove images from classes above the threshold and drop classes below it to balance the dataset.
- `--exclude-classes` (flag) If set, remove any classes marked with `-` in the taxa config.
- `--include-config-classes-only` (flag) If set, include only classes that appear in the taxa config and are marked with `+`.

## Outputs

After preprocessing and training, you will have any one of the following depending on your runthrough

- `models/class_names.json` — list of class labels  

- `models/fossil_resnet18.pt` — trained ResNet-18 weights 
- `models/fossil_resnet34.pt` — trained ResNet-34 weights  
- `models/fossil_resnet50.pt` — trained ResNet-50 weights  
- `models/fossil_vgg16.pt` — trained VGG-16 weights  
- `models/fossil_densenet121.pt` — trained DenseNet-121 weights  

- `models/train_index.pt` — (only compatible with ResNet models) embedding index of training images for nearest neighbors

#### Seeds and Reproducibility

All scripts accept a `--seed` argument to make results more repeatable:

```bash
python train_model.py --seed 123
python augment_images.py --seed 123
```

The seed affects:
- Dataset shuffling and validation split  
- Random data augmentations (rotations, flips, zoom, etc.)  
- Torch model initialization  

## Predict on New Images or Folders

After training, classify new fossils using `predict_image.py`.

### Arguments

- `--example-dir` **(required)** Path to the folder with example images. Processed recursively.
- `--console-print` (flag) Print extra details to the console.
- `--top-predictions` (default: `3`) How many top guesses to record for each image.
- `--neighbors` (default: `3`) How many closest training images to record.
- `--model-path` (default: `models/fossil_resnet18.pt`) Path to the trained model weights file.
- `--class-names` (default: `models/class_names.json`) Path to the `class_names.json` file.
- `--index-path` (default: `blank.file`) Path to the saved training feature index.
- `--output-dir` (default: `output`) Folder where the CSV will be saved.

### Example console output

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

The script also writes predictions results to a timestamped CSV in the specified `--output-dir`.

### Why nearest neighbors matter

Alongside top class predictions, the script can also show **nearest neighbors** from the training dataset based on feature similarity (for ResNet models only). This helps you:
- See which training fossils most influenced the model’s decision  
- Spot potential misclassifications (if the nearest examples look wrong)  
- Build confidence in the prediction by comparing to actual known fossils  

## Recommended Workflow

After building your training set, run scripts in the following order for best results:

1. **Convert HEIC photos (if any) > `convert_heics.py`**  
   Standardize image formats so everything is usable by the training scripts.  
2. **Generate a taxa-config.txt file > `taxa_for_config.py`**  
   Generate an easily modified taxa-config file which can modify which classes you wish to use.
3. **Augment dataset and build validation split > `augment_images.py`**  
   Expand training data with transformations and create a proper validation set.  
4. **Train model > `train_model.py`**  
   Fine-tune a ResNet-18, saving weights, class labels, and training embeddings.  
5. **Predict on new or example fossils > `predict_image.py`**  
   Run the trained model on fresh images to get top predictions and closest training examples.  
