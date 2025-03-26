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
import cv2
#from tqdm import tqdm

# project root dir
ROOT_DIR = os.path.abspath(os.curdir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print(DEVICE)


class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_file, target_classes,transform_density=None, transform=None, filter_target=False):
        """
        Args:
            root_dir (str): Path to the directory containing images.
            annotation_file (str): Path to the COCO annotation file (JSON format).
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.transform_density = transform_density
        self.target_classes = target_classes

        # Load all image IDs
        all_image_ids = list(self.coco.imgs.keys())

        if filter_target:
            # **Filter images that contain at least one target class**
            self.image_ids = [
                image_id for image_id in all_image_ids
                if any(ann['category_id'] in self.target_classes for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)))
            ]
            print(f"Filtered dataset: {len(self.image_ids)} images contain target classes.")
        else:
            # Use all images (for evaluation)
            self.image_ids = all_image_ids

    def __len__(self):
        return len(self.image_ids)
    
    def generate_density_map(self, image_shape, centers, sigma=10):
        """Convert object centers to a density map using Gaussian kernels."""
        height, width = image_shape[:2]
        density_map = np.zeros((height, width), dtype=np.float32)
        
        if len(centers) == 0:  # No persons in image
            return density_map  # Return zero map

        for x, y in centers:
            density_map[y, x] += 1  # Mark object location
        
        # Apply Gaussian blur to create a smooth density map
        # density_map = cv2.GaussianBlur(density_map, (15, 15), sigma)
        return density_map


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

    
        # get count of object
        category_counts = np.zeros(len(self.target_classes), dtype=np.float32)
        centers = []

        for ann in annotations:
            if ann['category_id'] in self.target_classes: 
                # counting number of object
                category_counts[self.target_classes.index(ann['category_id'])] += 1
                # get center for density map
                x, y, w, h = ann['bbox']
                center_x, center_y = int(x + w / 2), int(y + h / 2)  # Use bounding box center
                centers.append((center_x, center_y))

        # get density map
        image2 = cv2.imread(image_path)
        density_map = self.generate_density_map(image2.shape, centers)
        if self.transform_density:
            density_map = self.transform_density(density_map)     
        
        # Convert to tensor
        category_counts = torch.tensor(category_counts, dtype=torch.float32) 
        
        return torch.tensor(image_id) , image, density_map, category_counts

def get_COCO_dataset(target_classes): 
    # trasformer
    transform_density = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),  # Resize for ViT
        transforms.ToTensor()
    ])

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
    
    # load train data
    print("Loading Training Data...")
    train_dataset = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/train2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_train2017.json',
        target_classes,
        transform_density,
        transform,
        True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # load val data
    print("Loading Validation Data...")
    val_dataset = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/val2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_val2017.json',
        target_classes,
        transform_density,
        transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_dataset_blur = COCODataset(
        f'{ROOT_DIR}/data/COCO/images/val2017',
        f'{ROOT_DIR}/data/COCO/annotations/instances_val2017.json',
        target_classes,
        transform_density,
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
 
class ViTDensityModel(nn.Module):
    def __init__(self):
        super(ViTDensityModel, self).__init__()
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

        return density_map, count_pred  # Return both outputs



def train(model, dataloader, optimizer, loss_fn_density, loss_fn_count, device, cur_epoch=0, epochs=5):
    print("Training Model")
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        # for image_ids, images, density_maps, counts in tqdm(dataloader, desc=f'Epoch {cur_epoch+epoch+1}', unit="batch"):
        for image_ids, images, density_maps, counts in dataloader:
            images, density_maps, counts = images.to(device), density_maps.to(device), counts.to(device)

            optimizer.zero_grad()
            preds_density, preds_count = model(images)

            loss_density = loss_fn_density(preds_density, density_maps)
            loss_count = loss_fn_count(preds_count, counts)  # Match dimensions

            loss = loss_density + 0.1 * loss_count  # Weight count loss less
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        duration = time.time() - start_time
        print(f"Epoch [{cur_epoch+epoch+1}/{epochs+cur_epoch}], Loss: {avg_loss:.4f}, Time: {duration:.2f} secs")

        torch.save({
            'epoch': cur_epoch+epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"vit_density_v2_checkpoint_{cur_epoch+epoch+1}.pth")
        print(f"Checkpoint {cur_epoch+epoch+1} saved!")

        
def density_map_to_count(density_map):
    count = torch.sum(density_map).item()  # Sum pixels to get count
    return count

def evaluate(model, dataloader, device):
    results = []

    model.eval()
    with torch.no_grad():
        for image_ids, images, density_maps, counts in dataloader:
            images = images.to(device)
            preds_density, preds_count = model(images)

            for item, image_id in enumerate(image_ids):
                results_json = {
                    "image_id": int(image_id.item()),  # Convert tensor to integer
                    # "density_map": preds_density[item].cpu().numpy().tolist(),  # Move to CPU and get value
                    "prediction": preds_count[item].cpu().item(),  # Move to CPU and get value
                    "ground_truths": counts[item].cpu().item()  # Move to CPU and get value
                }
                results.append(results_json)

            
            del image_ids, images, counts, density_maps, preds_density, preds_count
            torch.cuda.empty_cache()  # Free GPU memory

    return results

def save_results(results, is_test_blur = False, epoch = None): 
    file_name = "ViT_density_v2" 
    if epoch: file_name += f"_{epoch}"
    file_name += "_results"
    if is_test_blur: file_name+= "_on_blurred_images"

    # Save results to JSON file
    with open(f'{ROOT_DIR}/results/{file_name}.json', "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {file_name}.json")


# Main function: 
target_classes = [1]
train_dataset,train_dataloader,val_dataset,val_dataloader,val_dataset_blur,val_dataloader_blur = get_COCO_dataset(target_classes)

model = ViTDensityModel().to(DEVICE)

# Define Loss Function & Optimizer
loss_fn_density = nn.MSELoss()
loss_fn_count = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Load checkpoint
# checkpoint = torch.load('vit_density_v2_checkpoint_8.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# cur_epoch = checkpoint['epoch']  # Resume from the correct epoch
cur_epoch = 0

for i in range(8):

    # Train the model
    train(model, train_dataloader, optimizer, loss_fn_density,loss_fn_count, DEVICE,cur_epoch=(cur_epoch+i), epochs=1)

    # Evaluate Model
    results = evaluate(model, val_dataloader, DEVICE)
    save_results(results, False, epoch=(cur_epoch+i+1))
    
    results = evaluate(model, val_dataloader_blur, DEVICE)
    save_results(results, True, epoch=(cur_epoch+i+1))

   

