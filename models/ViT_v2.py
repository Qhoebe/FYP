import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel
from pycocotools.coco import COCO
import os
import json
from PIL import Image
import numpy as np
import time

# project root dir
ROOT_DIR = os.path.abspath(os.curdir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
 
class ViTObjectCounter(nn.Module):
    def __init__(self):
        super(ViTObjectCounter, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Density Map Decoder (Same as Before)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Output density map
        )

        # Count Regression Head
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Convert [B, 1, H, W] → [B, 1, 1, 1]
            nn.Flatten(),             # [B, 1, 1, 1] → [B, 1]
            nn.Linear(1, 1)           # Final count prediction
        )

    def forward(self, x):
        features = self.vit(x).last_hidden_state
        features = features[:, 1:, :]  # Remove class token
        b, n, d = features.shape
        h, w = int(n**0.5), int(n**0.5)  # 14x14
        features = features.permute(0, 2, 1).contiguous().view(b, d, h, w)

        # Get density map
        density_map = self.decoder(features)

        # Get count prediction from density map
        count_pred = self.count_regressor(density_map)

        return count_pred  # Return both outputs


def train(model, dataloader, optimizer, loss_fn, device, cur_epoch=0, epochs=5, is_train_blur=False):
    print("Training Model")
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        # for image_ids, images, counts in  tqdm(dataloader, desc=f'Epoch {cur_epoch+epoch+1} Training model', unit="batch"):
        for image_ids, images, counts in  dataloader:
            image_ids, images, counts = image_ids.to(device), images.to(device), counts.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, counts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Epoch [{cur_epoch+epoch+1}/{epochs+cur_epoch}], Loss: {avg_loss:.4f}, Time: {duration} secs")
        
        if is_train_blur: 
            torch.save({
                'epoch': cur_epoch+epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"vit_image_blur_checkpoint_{cur_epoch+epoch+1}.pth")
            print(f"Checkpoint {cur_epoch+epoch+1} saved!")
        else: 
            torch.save({
                'epoch': cur_epoch+epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"vit_checkpoint_{cur_epoch+epoch+1}.pth")
            print(f"Checkpoint {cur_epoch+epoch+1} saved!")


def evaluate(model, dataloader, device):
    results = []

    model.eval()
    total_error = 0
    with torch.no_grad():
        for image_ids, images, counts in  dataloader:
            image_ids, images, counts = image_ids.to(device), images.to(device), counts.to(device)
            preds = model(images)
            total_error += torch.abs(preds - counts).sum().item()

            for item, image_id in enumerate(image_ids):  # Fix the iteration
                results_json = {
                    "image_id": int(image_id.item()),  # Convert tensor to integer
                    "prediction": preds[item].cpu().tolist(),
                    "ground_truths": counts[item].cpu().tolist()
                }
                results.append(results_json)
            
            # Free GPU memory manually
            del image_ids, images, counts, preds
            torch.cuda.empty_cache()  # Clears unused memory

    avg_error = total_error / len(dataloader.dataset)
    print(f"Mean Absolute Error (MAE): {avg_error:.2f}")

    return results

def save_results(results, is_train_blur=False, is_test_blur = False, epoch = None): 
    file_name = "ViT" 
    if is_train_blur: file_name += "_image_blur"
    if epoch: file_name += f"_{epoch}"
    file_name += "_results"
    if is_test_blur: file_name+= "_on_blurred_images"

    # Save results to JSON file
    with open(f'{ROOT_DIR}/results/{file_name}.json', "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {file_name}.json")


# Main function: 
target_classes = [1]
is_train_blur = False # determine if model should be trained with blurred images 
train_dataset,train_dataloader,val_dataset,val_dataloader,val_dataset_blur,val_dataloader_blur = get_COCO_dataset(target_classes,is_train_blur)

model = ViTObjectCounter().to(DEVICE)

# Define Loss Function & Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# # Load checkpoint
# checkpoint = torch.load('vit_checkpoint_5.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# cur_epoch = checkpoint['epoch']  # Resume from the correct epoch


for i in range(5):

   # Train the model
    train(model, train_dataloader, optimizer, loss_fn, DEVICE,cur_epoch=i, epochs=1,is_train_blur=is_train_blur)

    # Evaluate Model
    results = evaluate(model, val_dataloader, DEVICE)
    save_results(results, is_train_blur, False, epoch=(i+1))
    
    results = evaluate(model, val_dataloader_blur, DEVICE)
    save_results(results, is_train_blur, True, epoch=(i+1))

   

