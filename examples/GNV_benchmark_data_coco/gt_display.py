import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO

# Directory containing your images and the JSON file
data_dir = './'
# Path to the COCO format JSON file
json_file = os.path.join(data_dir, 'annotations.json')
# Directory to save the masked images
output_dir = os.path.join(data_dir, 'masked_images')
# Directory to save the original images with masks
original_with_masks_dir = os.path.join(data_dir, 'original_with_masks')

# Create the output directories if they don't exist
for directory in [output_dir, original_with_masks_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize COCO api for instance annotations
coco = COCO(json_file)

# Get all image ids
img_ids = coco.getImgIds()

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(data_dir, img_info['file_name'])
    image = cv2.imread(img_path)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Create a blank mask image
    mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
    for ann in anns:
        if 'segmentation' in ann:
            # Generate mask for the current annotation
            mask = coco.annToMask(ann)
            # Combine with existing mask
            mask_img |= mask.astype(np.uint8) * 255

    # Resize the mask image to match the size of the original image
    mask_img = cv2.resize(mask_img, (image.shape[1], image.shape[0]))

    # Save the original image with masks
    masked_original = image.copy()
    masked_original[mask_img > 0] = [0, 255, 0]  # Greem
    # Make the masks more transparent
    alpha = 0.5
    masked_original = cv2.addWeighted(image, 1 - alpha, masked_original, alpha, 0)
    cv2.imwrite(os.path.join(original_with_masks_dir, img_info['file_name']), masked_original)

    # Convert mask image to binary (black and white)
    _, binary_mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)

    # Save the binary masked image
    cv2.imwrite(os.path.join(output_dir, img_info['file_name']), binary_mask)

print('Binary masked images have been saved to:', output_dir)
print('Original images with transparent masks have been saved to:', original_with_masks_dir)
