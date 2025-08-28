Using Deep Learning to Identify Fossils of the Atlantic Coastal Plain:
In this project, we will train a neural network to classify fossil images from across the Eastern United States, spanning the rich Cretaceous beds of New Jersey to the diverse coastal deposits of Maryland, the Carolinas, and Florida. We will work collaboratively to develop a dynamic, Python-based toolkit for fossil image augmentation, convolutional neural network training, and model evaluation across a range of variables including dataset size, class count, and class balance. The result will be a powerful deep learning framework tailored to one of paleontology’s more understudied regions, offering valuable tools for both professional researchers and fossil enthusiasts alike. Sitting at the intersection of biology, computer science, and geology, this project is ideal for students interested in applying bioinformatics to distinctive, real-world scientific research, with the potential to contribute to publishable findings.

# Fossil Image Preprocessing & Training

This repository contains scripts to preprocess fossil image datasets (cropping, padding, augmentation), train a ResNet-18 classifier, and run predictions on new images.

## Installation

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install required libraries
pip install torch torchvision tqdm pillow
```

All other modules used are from Python’s standard library (`os`, `shutil`, `argparse`, `json`, `time`, `random`).

# Usage

## Preprocess and Augment Images
Prepare your dataset under this structure:

```
data/train/owner-*/<taxon>/
```

Each `owner-*` folder should contain taxon folders, each with `.png`, `.jpg`, or `.jpeg` images.  

Run the augmentation script:

```bash
python augment_images.py
```

This will:
- Rebuild `data/val/owner-combined` (validation set)  
- Create `data/augmented/owner-combined` with 20 augmented variants per training image  

## Train the Model

Use the training script (`train_model.py`) to train a ResNet-18 classifier.  

```bash
# Train on raw data
python train_model.py

# Train on augmented data
python train_model.py --use-augmented
```

### What the training script does
- Merges `owner-*` folders into a single `owner-combined` training folder (if using raw data)  
- Splits a validation set into `data/val/owner-combined` if missing  
- Applies augmentation and normalization  
- Loads a pretrained ResNet-18, updates the final layer for your classes, and trains for 5 epochs  
- Saves model weights and class names in the `models/` folder  

## Predict on New Images

After training, use the prediction script (`predict.py`) to classify new fossil images.  
Place images in `example_images/` and run:

```bash
python predict.py
```

For each image, the script prints the **top 3 predicted classes** with confidence scores.

Example output:

```
shark_tooth.png:
  1. Lamnidae (87.32% confidence)
  2. Carcharhinidae (10.15% confidence)
  3. Odontaspididae (2.53% confidence)
```

# Outputs

After preprocessing and training, the following files are saved under `models/`:

- `class_names.json` — class labels  
- `fossil_resnet18.pt` — trained ResNet-18 model weights  
