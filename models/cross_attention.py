import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel
from pycocotools.coco import COCO
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm

# project root dir
ROOT_DIR = os.path.abspath(os.curdir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_exemplar_features(image, bbox, transform):
    """
    Extracts the exemplar region based on bounding box.
    """
    x, y, w, h = bbox
    exemplar = image.crop((x, y, x + w, y + h))  # Crop the bounding box region
    if transform:
        exemplar = transform(exemplar)
    return exemplar

class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_file, target_classes, transform=None, mode ="train"):
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())  
        self.transform = transform
        self.target_classes = target_classes
        self.mode = mode  # "train" or "test"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_dir, image_info["file_name"])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        category_counts = np.zeros(len(self.target_classes), dtype=np.float32)
        exemplar = torch.zeros((3, 224, 224))  # Default blank tensor if no valid exemplar

        for ann in annotations:
            if ann['category_id'] in self.target_classes: 
                category_counts[self.target_classes.index(ann['category_id'])] += 1
                if self.mode == "train" and category_counts[self.target_classes.index(ann['category_id'])] == 1:  
                    exemplar_bbox = ann["bbox"]  
                    exemplar = extract_exemplar_features(image, exemplar_bbox, self.transform)

        category_counts = torch.tensor(category_counts, dtype=torch.float32) 

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image_id), image, exemplar, category_counts

def get_COCO_dataset(target_classes, is_train_blurred=False): 
    # trasformer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_blur = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ViT
        transforms.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 2.0)),  # Apply Gaussian blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if is_train_blurred: transform_train = transform_blur
    else: transform_train = transform
    
    # load train data
    print("Loading Training Data...")
    train_dataset = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/train2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_train2017.json',
        target_classes,
        transform_train
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # load val data
    print("Loading Validation Data...")
    val_dataset = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/val2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_val2017.json',
        target_classes,
        transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_dataset_blur = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/val2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_val2017.json',
        target_classes,
        transform_blur
    )
    val_dataloader_blur = DataLoader(val_dataset_blur, batch_size=16, shuffle=False)

    # sanity check
    print("train_dataset size:", len(train_dataset))
    print("train_loader size:", len(train_dataloader))
    print("val_dataset size:", len(val_dataset))
    print("val_loader size:", len(val_dataloader))
    print("val_dataset_blur size:", len(val_dataset_blur))
    print("val_loader_blur size:", len(val_dataloader_blur))

    return train_dataset,train_dataloader,val_dataset,val_dataloader,val_dataset_blur,val_dataloader_blur
 


# Model Definition
class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, exemplar_features, image_features):
        """
        Args:
            exemplar_features: [B, 1, D] - Query (Exemplar Features)
            image_features: [B, N, D] - Key & Value (Image Features)

        Returns:
            Attended Features [B, 1, D] -> Squeezed to [B, D]
        """
        attended_features, _ = self.cross_attention(query=exemplar_features, key=image_features, value=image_features)
        return attended_features.squeeze(1)  # Remove sequence dimension


class ExemplarObjectCounter(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=1):
        super(ExemplarObjectCounter, self).__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove FC layers

        self.exemplar_extractor = models.resnet18(pretrained=True)
        self.exemplar_extractor = nn.Sequential(*list(self.exemplar_extractor.children())[:-1])  
        self.exemplar_fc = nn.Linear(512, feature_dim)  

        self.cross_attention = CrossAttention(feature_dim, num_heads=8)  # Use Cross Attention

        self.regressor = nn.Linear(feature_dim, num_classes)

    def forward(self, images, exemplars):
        """
        Args:
            images: Input images [B, 3, H, W].
            exemplars: Exemplar regions from bounding boxes [B, 3, 224, 224].

        Returns:
            Object count predictions.
        """
        features = self.backbone(images)  # [B, C, H', W']
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

        # Extract exemplar features
        exemplar_features = self.exemplar_extractor(exemplars).view(B, -1)  # [B, 512]
        exemplar_features = self.exemplar_fc(exemplar_features).unsqueeze(1)  # [B, 1, 2048]

        attended_features = self.cross_attention(exemplar_features, features)  # [B, 2048]
        count_pred = self.regressor(attended_features)  # [B, 1]
        return count_pred


# Training
def train(model, dataloader, optimizer, loss_fn, device, cur_epoch=0, epochs=5, is_train_blur=False):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for image_ids, images, exemplars, counts in  tqdm(dataloader, desc=f'Epoch {cur_epoch+epoch+1} Training model', unit="batch"):
        # for image_ids, images, exemplars, counts in dataloader:
            images, exemplars, counts = images.to(device), exemplars.to(device), counts.to(device)

            optimizer.zero_grad()
            preds = model(images, exemplars)
            loss = loss_fn(preds, counts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{cur_epoch+epoch+1}/{epochs+cur_epoch}], Loss: {avg_loss:.4f}")

        if is_train_blur:
            torch.save({
                'epoch': cur_epoch+epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"cross_attention_image_blur_checkpoint_{cur_epoch+epoch+1}.pth")
            print(f"Checkpoint {cur_epoch+epoch+1} saved!")
        else: 
            torch.save({
                'epoch': cur_epoch+epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"cross_attention_checkpoint_{cur_epoch+epoch+1}.pth")
            print(f"Checkpoint {cur_epoch+epoch+1} saved!")

def evaluate(model, dataloader, device):
    results = []

    model.eval()
    total_error = 0
    with torch.no_grad():
        for image_ids, images, exemplars, counts in dataloader:
            image_ids, images, exemplars, counts = image_ids.to(device), images.to(device), exemplars.to(device), counts.to(device)
            preds = model(images, exemplars)
            total_error += torch.abs(preds - counts).sum().item()

            for item, image_id in enumerate(image_ids):  # Fix the iteration
                results_json = {
                    "image_id": int(image_id.item()),  # Convert tensor to integer
                    "prediction": preds[item].cpu().tolist(),
                    "ground_truths": counts[item].cpu().tolist()
                }
                results.append(results_json)
            
            # Free GPU memory manually
            del image_ids, images, exemplars, counts, preds
            torch.cuda.empty_cache()  # Clears unused memory

    avg_error = total_error / len(dataloader.dataset)
    print(f"Mean Absolute Error (MAE): {avg_error:.2f}")

    return results

def save_results(results, is_train_blur=False, is_test_blur = False, epoch = None): 
    file_name = "cross_attention" 
    if is_train_blur: file_name += "_image_blur"
    if epoch: file_name += f"_{epoch}"
    file_name += "_results"
    if is_test_blur: file_name+= "_on_blurred_images"

    # Save results to JSON file
    with open(f'{ROOT_DIR}/results/{file_name}.json', "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Saved results to {file_name}.json")


# Initialize and train
target_classes = [1]
is_train_blur = False 
train_dataset,train_dataloader,val_dataset,val_dataloader,val_dataset_blur,val_dataloader_blur = get_COCO_dataset(target_classes,is_train_blur)

model = ExemplarObjectCounter().to(DEVICE)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# # Load Checkpoint
# checkpoint = torch.load(f'cross_attention_checkpoint_8.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# cur_epoch = checkpoint['epoch']  # Resume from the correct epoch
cur_epoch = 0

for i in range(6):

    # Train model
    train(model, train_dataloader, optimizer, loss_fn, DEVICE,cur_epoch=(cur_epoch+i), epochs=1,is_train_blur = is_train_blur)

    # Evaluate Model
    results = evaluate(model, val_dataloader, DEVICE)
    save_results(results, is_train_blur, False, epoch=(cur_epoch+i+1))
    
    results = evaluate(model, val_dataloader_blur, DEVICE)
    save_results(results, is_train_blur, True, epoch=(cur_epoch+i+1))


