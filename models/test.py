import numpy as np
import os
from PIL import Image

ROOT_DIR = os.path.abspath(os.curdir)
print('root dir:', ROOT_DIR)

image_id = 7141  # Use the image file name or its ID

image_path = f'{ROOT_DIR}/data/FSC147/images_384_VarV2/{image_id}.jpg'
image = Image.open(image_path)

import json
with open(f'{ROOT_DIR}/data/FSC147/annotation_FSC147_384.json') as f:
    annotations = json.load(f)

annotation = annotations[f"{image_id}.jpg"]

print(annotation.keys())

img = Image.open(image_path)
img.show()
# print(annotation['bbox'])  # Bounding box coordinates (x, y, width, height)
# print(annotation['count'])  # Object count