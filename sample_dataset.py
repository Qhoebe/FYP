import os
import random
import shutil

# Step 1: Identify the source folder location
source_folder = 'data/COCO/images/train2017'

# Step 2: Create a new folder
destination_folder = 'data/COCO/images/sample_train2017'
os.makedirs(destination_folder, exist_ok=True)

# Step 3: Randomly select 20% of the images
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

num_images_to_select = 50000
selected_images = random.sample(image_files, num_images_to_select)

# Step 4: Copy the selected images to the new folder
for image in selected_images:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.copy(source_path, destination_path)

print(f'{len(selected_images)} images have been copied to {destination_folder}')
