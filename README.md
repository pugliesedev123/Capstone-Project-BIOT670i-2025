# Using Deep Learning to Identify Fossils of the Atlantic Coastal Plain

This project trains neural networks to classify fossil images from across the Eastern United States, spanning the Cretaceous beds of New Jersey to diverse coastal deposits of Maryland, the Carolinas, and Florida.

It provides a Python toolkit for:
- **Image preprocessing and augmentation** (crop, colorize, rotate, zoom, flips, etc.)
- **ResNet-18, ResNet-34, ResNet-50, VGG16, and DenseNet121 training** with basic class-imbalance handling
- **Prediction** on new images to measure classifier performance and generate top-k guesses

The goal is a practical framework for researchers and fossil enthusiasts alike.

Portions of this work were completed as part of the University of Maryland Global Campus M.S. Biotechnology capstone (BIOT670i-2025). More information on project goals can be found in `Project_Description.pdf`.

Portions of these results were also used to create the poster "<em>Analyzing the Impact of Increased Species Diversity on Machine Learning-Based Image Classification within the Megatoothed Sharks</em>" for the joint Mid-Atlantic and NorthEast (MANE) Geobiology Symposium at the Edelman Fossil Park and Museum on Friday, 27 February 2026.

A copy of the poster can be found at `Megatoothed_Sharks_MANE_Presentation.pdf` and assosciate scripts are found in `Megatoothed_Sharks_MANE_Presentation.zip`.
Image credits and licensing details can be found at: `Megatoothed_Sharks_Deep_Learning_Credit_Citation.pdf`.



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

pip install torch torchvision tqdm pillow numpy psutil
```

All other imports are from the Python standard library (for example: `os`, `shutil`, `argparse`).



## Dataset layout

Place images in **owner** folders, with subfolders that represent classes (taxa):

```
data/train/
  owner-Wikimedia/
    taxon-scapanorhynchus_texanus/
    taxon-cretolamna_appendiculata/
    taxon-belemnitella_americana/
    taxon-exogyra_sp/
  owner-author/
    taxon-isurus_sp/
```

- Each `owner-*` folder contains one or more `taxon-*` subfolders.
- Each class folder contains `.jpg`, `.jpeg`, `.png` images (and optionally `.heic` or `.gif` before conversion).

Requests for sample formatted data can be made out to either [pugliesdev123@gmail.com](mailto:pugliesdev123@gmail.com) or [digsitedetective@gmail.com](mailto:digsitedetective@gmail.com).



## Taxa config file (`taxa-config.txt`)

Several scripts can optionally read a `taxa-config.txt` file to include/exclude classes.

Recommended convention per line (one class per line):
- `+taxon-exogyra_sp` include explicitly
- `-taxon-sphenodiscus_sp` exclude explicitly
- `taxon-mosasaurus_maximus` neutral (treated as “present” if discovered on disk)

When flags are used:

- `--exclude-classes`
  - Any line starting with `-` is excluded.

- `--include-config-classes-only`
  - Only classes listed with `+` are included.
  - Any discovered class not listed with `+` is excluded.

If neither flag is set, the config file can still be used as a reference list, but discovered classes are included based on the folder tree.



## Utilities

### Convert HEIC or GIF images

If your dataset includes `.heic` or `.gif` images (e.g., from iPhones), use:

```bash
python converter_to_jpg.py --target-dir data/train --file-type all
```

This converts all image files under the target directory into `.jpg` files (preserving the folder structure).

### Get a dataset report (and optionally remove empty class folders)

```bash
python get_file_list.py
```

Suggested output to document in the script and in the README (if applicable):
- Total images per class
- Total classes discovered
- Empty folders found (and whether they were removed)

### Update file names to match their class folder

```bash
python update_file_name.py
```

This is typically used to enforce consistent naming such as:
`taxon-exogyra_sp/exogyra_sp_0001.jpg`

### Generate `taxa-config.txt` from your dataset

```bash
python taxa_for_config.py
```

This traverses the training folder and writes a `taxa-config.txt` file that can be used to:
- Exclude rare/problem classes
- Run experiments on a selected subset of classes
- Enforce consistent class ordering



## Preprocess and augment images

Run the augmentation script to build a validation split and augmented training set:

```bash
python augment_images.py
```

Default behavior:
- Builds `data/val/owner-combined` (validation set)
- Creates `data/augmented/owner-combined` (augmented training set)

### Key behavior

- Each class is split into train/validation, typically **80/20** (1/5 moved to validation).
- Validation images are copied unchanged.
- Training images are augmented using a mix of geometric and appearance transforms (rotation, scaling, flips, sharpness, grayscale, histogram equalization).

### Arguments

- `--input-root` (default: `data/train`)
  - Root folder containing `owner-*` subdirectories.

- `--input-config` (default: `taxa-config.txt`)
  - Taxa config file path.

- `--val-root` (default: `data/val/owner-combined`)
  - Output folder for validation images.

- `--aug-root` (default: `data/augmented/owner-combined`)
  - Output folder for augmented training images.

- `--aug-per-image` (default: `3`)
  - Number of augmented copies per original training image.

- `--seed` (default: `42`)
  - Seed used for shuffling, splitting, and augmentation.

- `--console-print` (flag)
  - Print extra details to the console.

- `--exclude-classes` (flag)
  - Exclude classes marked with `-` in `taxa-config.txt`.

- `--include-config-classes-only` (flag)
  - Include only classes marked with `+` in `taxa-config.txt`.

- `--threshold` (integer)
  - If set, balance classes to a cap:
    - Drop classes with fewer than `threshold` images.
    - Randomly downsample classes with more than `threshold` images to match the cap.

- `--disable-tf` (repeatable; disables transforms)
  - Supported values (example list): `rotate`, `scale`, `zoom`, `horizontalflip`, `verticalflip`, `grayscale`, `equalize`, `sharpen`
  - Example:
    ```bash
    python augment_images.py --disable-tf grayscale --disable-tf equalize
    ```

- `--disable-ca` (repeatable; disables class augmentation)
  - Disable augmentation for a specific class label (example):
    ```bash
    python augment_images.py --disable-ca taxon-exogyra_sp
    ```

### Notes

- Classes with fewer than **20 images** may be skipped to avoid low-quality splits.
  - If this threshold is enforced in code, consider surfacing it as a named argument such as `--min-images` in a future update.



## Train the model

Train a classifier on your dataset:

```bash
# Option A: train from raw data (expects training images under data/train/)
python train_model.py

# Option B: train from a prebuilt augmented set (expects data/augmented/owner-combined/)
python train_model.py --use-augmented
```

### What happens

- If `--use-augmented` is NOT set:
  - Owner folders under `data/train/owner-*` are merged into `data/augmented/owner-combined` before training.

- If a validation set is missing:
  - A split is created at `data/val/owner-combined`.

- A pretrained model is fine-tuned for your classes.
- Basic class imbalance is addressed with per-class loss weights.
- Model artifacts are saved to `models/`.

### Arguments

- `--use-augmented` (flag)
  - If set, train from `data/augmented/owner-combined`.
  - If not set, build `data/augmented/owner-combined` from `data/train/owner-*` first.

- `--console-print` (flag)
  - Print extra setup/training details.

- `--use-pre-train` (default: `True`)
  - Load ImageNet-pretrained weights for the selected backbone.

- `--seed` (default: `42`)
  - Controls shuffling/splits and other randomness via a unified seed helper.

- `--batch-size` (default: `16`)
  - Train/validation batch size.

- `--epochs` (default: `5`)
  - Number of training epochs.

- `--input-config` (default: `taxa-config.txt`)
  - Path to the taxa config used to guide class inclusion (and, if applicable, augmentation decisions).

- `--model-path` (default: `models/fossil_resnet18.pt`)
  - Where to save trained weights.

- `--index-path` (default: `models/train_index.pt`)
  - Where to save the optional “training embedding index” (features + labels + file paths).

- `--model` (default: `resnet18`)
  - Choices: `resnet18`, `resnet34`, `resnet50`, `vgg16`, `densenet121`

- `--threshold` (integer)
  - If set, cap each class at this image count (drops classes below the cap; downsamples classes above it).

- `--exclude-classes` (flag)
  - Remove classes marked with `-` in `taxa-config.txt`.

- `--include-config-classes-only` (flag)
  - Include only classes marked with `+` in `taxa-config.txt`.



## Outputs

After preprocessing and training, you should have some of the following (depending on which model you train):

- `models/class_names.json` — list of class labels in model output order

Model weights:
- `models/fossil_resnet18.pt`
- `models/fossil_resnet34.pt`
- `models/fossil_resnet50.pt`
- `models/fossil_vgg16.pt`
- `models/fossil_densenet121.pt`

Optional index:
- `models/train_index.pt` — embedding index of training images (feature vectors + labels + file paths)

### Seeds and reproducibility

All scripts accept a `--seed` argument:

```bash
python augment_images.py --seed 123
python train_model.py --seed 123
```

The seed affects:
- Dataset shuffling and validation split
- Random augmentations (rotations, flips, zoom, etc.)
- Torch initialization and any other seeded randomness



## Predict on new images or folders

After training, classify new fossils using `predict_image.py`:

```bash
python predict_image.py --example-dir example_images
```

### Arguments

- `--example-dir` (required)
  - Folder with images to predict (processed recursively).

- `--console-print` (flag)
  - Print extra details.

- `--top-predictions` (default: `3`)
  - How many top classes to record per image.

- `--neighbors` (default: `3`)
  - How many nearest training images to record **if** an index is provided.

- `--model-path` (default: `models/fossil_resnet18.pt`)
  - Path to trained weights.

- `--class-names` (default: `models/class_names.json`)
  - Path to class label list.

- `--index-path` (default: `blank.file`)
  - Path to a saved training embedding index.
  - If this file does not exist (or is intentionally set to a dummy path), neighbor search is skipped and only top-k predictions are produced.

- `--output-dir` (default: `output`)
  - Directory where the timestamped CSV will be written.

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



## Recommended workflow

1. Convert HEIC or PNG photos (if any) to JPG > `convert_heics.py`
2. Generate `taxa-config.txt` > `taxa_for_config.py`
3. Augment dataset + build validation split > `augment_images.py`
4. Train model > `train_model.py`
5. Predict on new images > `predict_image.py`



## Run all scripts with a single command

A helper exists at `util/run_with_summary.py` to run:
- augmentation (`augment_images.py`)
- training (`train_model.py`)
- prediction (`predict_image.py`)

The script writes a summary file that captures:
- script arguments used for each step
- basic system information
- taxa information discovered/used for the run
- the output CSV name produced by prediction

Recommended bookkeeping per run:
- the exact `taxa-config.txt` used
- `data/augmented/owner-combined/` (augmented training images)
- `data/val/owner-combined/` (validation images)
- the example/prediction folder used
- the prediction CSV and its corresponding summary file
