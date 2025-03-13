import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def count_objects(model, img):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
    return int(output.item())  # Assuming model outputs a count
