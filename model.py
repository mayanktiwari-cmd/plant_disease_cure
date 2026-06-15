import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# Load class names
with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

# CORRECTED FOR INFERENCE: No random augmentations, deterministic sizing only
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standardize directly to the model's expected input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 38)
    )
    # map_location='cpu' ensures this runs fine even on a server/machine without a GPU
    model.load_state_dict(
        torch.load('plant_disease_model.pth', map_location='cpu')
    )
    model.eval() # Crucial: locks BatchNorm and Dropout layers
    return model

def predict(image, model):
    # Ensure image is in RGB mode (handles PNGs with alpha channels or grayscale uploads)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad(): # Crucial: Disables gradient calculation to save memory and speed up
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    class_name = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    # Clean up class name string
    # "Apple___Apple_scab" → "Apple — Apple Scab"
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').title()
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Healthy"

    return plant, disease, confidence_score