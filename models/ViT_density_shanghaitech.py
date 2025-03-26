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
from tqdm import tqdm

# project root dir
ROOT_DIR = os.path.abspath(os.curdir)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(DEVICE)


class ShanghaiTechDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.image_files = sorted(os.listdir(image_dir))  # List of image files
        self.gt_files = sorted(os.listdir(gt_dir))       # List of ground truth files (density maps)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file name (without extension if needed)
        image_name = self.image_files[idx]  # Full filename (e.g., 'IMG_1.jpg')

        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB

        # Load ground truth density map
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        gt = np.load(gt_path)  # Assuming the GT is stored as a .npy file with density map

        if self.transform:
            image = self.transform(image)

        # Convert ground truth density map to tensor
        gt = torch.tensor(gt, dtype=torch.float32)

        return image_name, image, gt   # Return image name along with image and ground truth


def get_ShanghaiTech_Dataset(): 
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
    train_dataset = ShanghaiTechDataset(
        f'{ROOT_DIR}/data/ShanghaiTech/part_A_final/train_data/images',
        f'{ROOT_DIR}/data/ShanghaiTech/part_A_final/train_data/ground_truth',
        transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # load test data
    print("Loading Test Data...")
    test_dataset = ShanghaiTechDataset(
        f'{ROOT_DIR}/data/ShanghaiTech/part_A_final/test_data/images',
        f'{ROOT_DIR}/data/ShanghaiTech/part_A_final/test_data/ground_truth',
        transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    test_dataset_blur = ShanghaiTechDataset(
        f'{ROOT_DIR}/data/ShanghaiTech/part_A_final/test_data/images',
        f'{ROOT_DIR}/data/ShanghaiTech/part_A_final/test_data/ground_truth',
        transform_blur
    )
    test_dataloader_blur = DataLoader(test_dataset_blur, batch_size=16, shuffle=False)

    # sanity check
    print("train_dataset size:", len(train_dataset))
    print("train_loader size:", len(train_dataloader))
    print("test_dataset size:", len(test_dataset))
    print("test_loader size:", len(test_dataloader))
    print("test_dataset_blur size:", len(test_dataset_blur))
    print("test_loader_blur size:", len(test_dataloader_blur))

    return train_dataset,train_dataloader,test_dataset,test_dataloader,test_dataset_blur,test_dataloader_blur
 
import torch
import torch.nn as nn
from transformers import ViTModel

class ViT_Counter(nn.Module):
    def __init__(self, pretrained_model="google/vit-base-patch16-224"):
        super(ViT_Counter, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model)
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # Output 1-channel density map
        )

    def forward(self, x):
        vit_out = self.vit(x).last_hidden_state  # Shape: [batch, num_patches, 768]
        b, n, c = vit_out.shape
        h = w = int(n**0.5)  # Assuming square patch layout
        vit_out = vit_out.permute(0, 2, 1).reshape(b, c, h, w)  # Convert to feature map
        density_map = self.decoder(vit_out)  # Predict density map
        return density_map


def train(model, dataloader, optimizer, loss_fn, device, cur_epoch=0, epochs=5):
    print("Training Model")
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for image_ids, images, density_maps in  tqdm(dataloader, desc=f'Epoch {cur_epoch+epoch+1} Training model', unit="batch"):
        #for image_ids, images, density_maps in dataloader:
            images, density_maps = images.to(device),density_maps.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, density_maps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Epoch [{cur_epoch+epoch+1}/{epochs+cur_epoch}], Loss: {avg_loss:.4f}, Time: {duration} secs")
        
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
        for image_ids, images, density_maps in  dataloader:
            image_ids, images, density_maps = image_ids.to(device), images.to(device), density_maps.to(device)
            preds = model(images)
            preds_count = [density_map_to_count(density_map) for density_map in preds]
            ground_truths = [density_map_to_count(density_map) for density_map in density_maps]

            for item, image_id in enumerate(image_ids):  # Fix the iteration
                results_json = {
                    "image_id": int(image_id.item()),  # Convert tensor to integer
                    "prediction": preds_count[item].cpu(),
                    "ground_truths": ground_truths[item].cpu()
                }
                results.append(results_json)
            
            # Free GPU memory manually
            del image_ids, images, density_maps, preds
            torch.cuda.empty_cache()  # Clears unused memory


    return results

def save_results(results, is_test_blur = False, epoch = None): 
    file_name = "ViT_density" 
    if epoch: file_name += f"_{epoch}"
    file_name += "_results"
    if is_test_blur: file_name+= "_on_blurred_images"

    # Save results to JSON file
    with open(f'{ROOT_DIR}/results/{file_name}.json', "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {file_name}.json")


# Main function: 
target_classes = [1]
train_dataset,train_dataloader,test_dataset,test_dataloader,test_dataset_blur,test_dataloader_blur = get_ShanghaiTech_Dataset()

model = ViT_Counter().to(DEVICE)

# Define Loss Function & Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Load checkpoint
# checkpoint = torch.load('model_weights/vit_density_checkpoint_1.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# cur_epoch = checkpoint['epoch']  # Resume from the correct epoch
cur_epoch = 0

for i in range(1):

    # Train the model
    train(model, train_dataloader, optimizer, loss_fn, DEVICE,cur_epoch=(cur_epoch+i), epochs=1)

    # Etestuate Model
    results = evaluate(model, test_dataloader, DEVICE)
    save_results(results, False, epoch=(cur_epoch+i+1))
    
    results = evaluate(model, test_dataloader_blur, DEVICE)
    save_results(results, True, epoch=(cur_epoch+i+1))

   

