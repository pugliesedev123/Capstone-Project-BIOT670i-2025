# evaluate_model.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ----- Set device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load transforms -----
val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----- Load validation data -----
val_data = datasets.ImageFolder('data/val', transform=val_transforms)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=16, shuffle=False)
class_names = val_data.classes

# ----- Load model -----
num_classes = len(class_names)
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/fossil_resnet18.pt'))
model = model.to(device)
model.eval()

# ----- Evaluate -----
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# ----- Metrics -----
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {acc * 100:.2f}%")

# ----- Confusion Matrix -----
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
