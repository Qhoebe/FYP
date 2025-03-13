import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTModel
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initiate_model():
    
    model = ViTObjectCounter().to(DEVICE)
    checkpoint = torch.load('model/model_weights/vit_checkpoint_10.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]

def count_objects(model, img):
    image = preprocess_image(img).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(image)
    return int(output.item())  # Assuming model outputs a count

class ViTObjectCounter(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=1):
        super(ViTObjectCounter, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.vit.config.hidden_size, num_classes)  # Regression Head

    def forward(self, images):
        outputs = self.vit(images).last_hidden_state[:, 0, :]  # Use CLS token
        count_pred = self.regressor(outputs)
        return count_pred
