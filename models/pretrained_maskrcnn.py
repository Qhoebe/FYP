import os
import numpy as np 
import json
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# project root dir
ROOT_DIR = os.path.abspath(os.curdir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_file, target_classes, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing images.
            annotation_file (str): Path to the COCO annotation file (JSON format).
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())  # Get all image IDs
        self.transform = transform
        self.target_classes = target_classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_dir, image_info["file_name"])

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load captions
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        category_counts = np.zeros(len(self.target_classes), dtype=np.float32)

        for ann in annotations:
            if ann['category_id'] in self.target_classes: 
                category_counts[self.target_classes.index(ann['category_id'])] += 1
        
        # Convert to tensor
        category_counts = torch.tensor(category_counts, dtype=torch.float32) 

        return torch.tensor(image_id) , image, category_counts
    

# dataset information 
def get_COCO_dataset(target_classes, is_blurred): 
    # trasformer
    if is_blurred: 
        transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
    else: 
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # load val data
    print("Loading Validation Data...")
    val_dataset = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/val2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_val2017.json',
        target_classes,
        transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # sanity check
    print("val_dataset size:", len(val_dataset))
    print("val_loader size:", len(val_dataloader))

    return val_dataset,val_dataloader
    
class PretrainedMaskRCNN(nn.Module):
    def __init__(self, threshold=0.5): 
        super().__init__()
        self.model = models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT')
        self.threshold = threshold

    def forward(self, image_ids, images, target_classes):
        detections = self.model(images)  # Convert batch tensor to list of images

        detection = detections[0]
        counts = np.zeros(len(target_classes), dtype=np.float32)

        # Extract bounding boxes, labels, and scores from the prediction
        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']
        masks = detection['masks']

        # filter out only specific categories
        target_indices = [i for i, label in enumerate(labels) if label in target_classes]
        filtered_boxes = boxes[target_indices]
        filtered_labels = labels[target_indices]
        filtered_scores = scores[target_indices]
        
        filtered_boxes = filtered_boxes[filtered_scores > self.threshold]
        filtered_labels = filtered_labels[filtered_scores > self.threshold]
        filtered_scores = filtered_scores[filtered_scores > self.threshold]


        for label in filtered_labels:
            counts[target_classes.index(label)] += 1

        return {
            "image_id": image_ids,
            "boxes": filtered_boxes,
            "labels": filtered_labels,
            "scores": filtered_scores,
            "prediction": counts
        }

    
def pretrained_evaluation(is_blurred = False):
    # Run Evaluation on COCO Validation Data 
    target_classes = [1]
    val_dataset,val_dataloader = get_COCO_dataset(target_classes, is_blurred)

    print("Running Mask RCNN model...")
    model = PretrainedMaskRCNN(threshold=0.8).to(DEVICE)

    model.eval()

    results = []

    for image_ids, images, counts in val_dataloader:
        image_ids = image_ids.to(DEVICE) 
        images = images.to(DEVICE) 
        counts = counts.to(DEVICE) 

        detection = model(image_ids, images, target_classes)
        detection["ground_truths"] = counts.cpu().tolist()  # Move to CPU

        # Move all detection dictionary tensors to CPU before storing
        # Convert tensors and NumPy arrays to lists before storing in results
        detection_json = {
            key: (
                value.cpu().tolist() if isinstance(value, torch.Tensor) else
                value.tolist() if isinstance(value, np.ndarray) else
                value
            ) for key, value in detection.items()
        }

        results.append(detection_json)

        # Free GPU memory manually
        del image_ids, images, counts, detection
        torch.cuda.empty_cache()  # Clears unused memory

    # Save results to JSON file
    if is_blurred:
        with open(f'{ROOT_DIR}/results/pretrained_maskrcnn_results_on_blurred_images.json', "w") as f:
            json.dump(results, f, indent=4)
    else: 
        with open(f'{ROOT_DIR}/results/pretrained_maskrcnn_results.json.json', "w") as f:
            json.dump(results, f, indent=4)
    




    




                

    

