import os
import numpy as np 
import json
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
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
def get_COCO_dataset(target_classes): 
    # trasformer
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # def collate_fn(batch):
    #     image_ids = [item[0] for item in batch]  # Stack image IDs into a tensor
    #     images = [item[1] for item in batch]  # Stack image tensors
    #     counts = [item[2] for item in batch]  # Stack count tensors
    #     return image_ids, images, counts
    
    # load train data
    print("Loading Training Data...")
    train_dataset = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/train2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_train2017.json',
        target_classes,
        transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

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
    print("train_dataset size:", len(train_dataset))
    print("train_loader size:", len(train_dataloader))
    print("val_dataset size:", len(val_dataset))
    print("val_loader size:", len(val_dataloader))

    return train_dataset,train_dataloader,val_dataset,val_dataloader
    
class PretrainedFasterRCNN(nn.Module):
    def __init__(self, threshold=0.5): 
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        self.threshold = threshold

    def forward(self,image_ids,images,target_classes):
        detections = self.model(images)
    
        detection = detections[0]
        counts = np.zeros(len(target_classes), dtype=np.float32)
        # Extract bounding boxes, labels, and scores from the prediction
        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']

        # filter out only specific categories
        target_indices = [i for i, label in enumerate(labels) if label in target_classes ]
        filtered_boxes = boxes[target_indices]
        filtered_labels = labels[target_indices]
        filtered_scores = scores[target_indices]

        filtered_boxes = filtered_boxes[filtered_scores > self.threshold]
        filtered_labels = filtered_labels[filtered_scores > self.threshold]
        filtered_scores = filtered_scores[filtered_scores > self.threshold]

        for label in filtered_labels: 
            counts[target_classes.index(label)] += 1

        # Store filtered results
        return ({
            "image_id":image_ids,
            "boxes": filtered_boxes,
            "labels": filtered_labels,
            "scores": filtered_scores,
            "counts": counts
        })
    
def pretrained_evaluation():
    # Run Evaluation on COCO Validation Data 
    target_classes = [1]
    train_dataset,train_dataloader,val_dataset,val_dataloader = get_COCO_dataset(target_classes)

    print("Running Faster RCNN model...")
    model = PretrainedFasterRCNN(threshold=0.8).to(DEVICE)

    model.eval()

    se_scores,ae_scores, em_scores = [],[],[]
    results = []

    for image_ids, images, counts in tqdm(val_dataloader, desc="Evaluating model", unit="sample"):
        image_ids = image_ids.to(DEVICE) 
        images = images.to(DEVICE) 
        counts = counts.to(DEVICE) 

        detection = model(image_ids, images, target_classes)
        detection["actual_counts"] = counts.cpu().tolist()  # Move to CPU
        detection["square_error"] = (np.array(detection["actual_counts"]) - np.array(detection["counts"]))**2
        detection["absolute_error"] = abs(np.array(detection["actual_counts"]) - np.array(detection["counts"]))
        detection["exact_match"] = np.where(detection["square_error"] == 0, 1, 0)

        ae_scores.append(detection["absolute_error"])
        se_scores.append(detection["square_error"])
        em_scores.append(detection["exact_match"])

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

    # Compute Metric Scores
    print("Evaluation Results")
    print("Mean Square Error:", np.mean(np.array(se_scores), axis=0))
    print("Mean absolute Error:", np.mean(np.array(ae_scores), axis=0))
    print("Exact Match Ratio:", np.mean(np.array(em_scores), axis=0))

    # Save results to JSON file
    with open(f'{ROOT_DIR}/results/pretrained_fasterrcnn_results.json', "w") as f:
        json.dump(results, f, indent=4)

    




                

    

