import torch.nn.functional as F
import torch
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn as nn
import json

# Define device  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names  
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Load model  
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust dynamically
model.load_state_dict(torch.load('models/fossil_resnet18.pt', map_location=device))
model = model.to(device)
model.eval()

# Define transforms  
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Predict all images in folder  
example_dir = 'example_images'
image_files = [f for f in os.listdir(example_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

top_n = 3  # Number of predictions to show

for file in image_files:
    img_path = os.path.join(example_dir, file)
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)

        # Get top N predictions
        top_probs, top_indices = torch.topk(probabilities, top_n)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()

    print(f"\n{file}:")
    
    for i in range(top_n):
        predicted_class = class_names[top_indices[i]]
        confidence_pct = top_probs[i] * 100
        print(f"  {i+1}. {predicted_class} ({confidence_pct:.2f}% confidence)")