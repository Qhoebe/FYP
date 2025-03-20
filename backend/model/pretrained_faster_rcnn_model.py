import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(): 
    def __init__(self):
        self.model = PretrainedFasterRCNN(threshold=0.8).to(DEVICE)

    def preprocess_image(self,img):
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        img_tensor = transform(img)
        return img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]

    def count_objects(self, img):
        img_tensor = self.preprocess_image(img).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor, [1])
        return int(output.item())  # Assuming model outputs a count

class PretrainedFasterRCNN(nn.Module):
    def __init__(self, threshold=0.5): 
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        self.threshold = threshold

    def forward(self,images,target_classes):
        detections = self.model(images)
    
        detection = detections[0]
        counts = np.zeros(len(target_classes), dtype=np.float32)
        # Extract bounding boxes, labels, and scores from the prediction
        labels = detection['labels']
        scores = detection['scores']

        # filter out only specific categories
        target_indices = [i for i, label in enumerate(labels) if label in target_classes ]
        filtered_labels = labels[target_indices]
        filtered_scores = scores[target_indices]

        filtered_labels = filtered_labels[filtered_scores > self.threshold]

        for label in filtered_labels: 
            counts[target_classes.index(label)] += 1

        # Store filtered results
        return counts
