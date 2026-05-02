"""
This script processes images in a folder using YOLOv8 for object detection and three different segmentation methods after detection.
"""

from ultralytics import YOLO  # Assuming YOLOv5 is installed as 'ultralytics'
import cv2
import numpy as np
from skimage import io
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte
from datetime import datetime
import os
import argparse



def apply_segmentation(image, method):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'Edge-Based+YOLO':
        edge_img = sobel(gray_img) > 0.01 # best
        return edge_img.astype(np.uint8) * 255  # Binary mask multiplied to match uint8 format

    elif method == 'Thresholding+YOLO':
        # Convert to HSV and use adaptive thresholding on the hue channel
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_img[:, :, 0]
        thresh_img = cv2.adaptiveThreshold(hue_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
        return thresh_img

    elif method == 'Watershed+YOLO':
        # Apply color filtering before Watershed segmentation
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(img_hsv, (36, 25, 25), (70, 255,255))

        # Simple morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)

        # Background (sure background area)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Foreground (sure foreground area)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Unknown region (boundary between foreground and background)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed algorithm
        markers = cv2.watershed(image, markers)

        # Create the mask where the regions inside the boundaries will be white
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers > 1] = 255  # Ignore background and boundaries

        # Optionally, if you want to remove small noise or small regions, you can perform an opening operation
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    else:
        # Return an empty mask if method is unknown
        return np.zeros_like(gray_img, dtype=np.uint8)


def invert_mask(mask):
    return cv2.bitwise_not(mask)

def load_yolo_bboxes(bbox_file, image_shape):
    """
    Load bounding boxes from a YOLO format file.
    """
    bboxes = []
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Convert from YOLO format to absolute coordinates
                x_center, y_center, width, height = map(float, parts[1:5])
                img_height, img_width = image_shape[:2]
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                bboxes.append([x1, y1, x2, y2])
    return bboxes


def save_images(image, mask, method, base_dir, file_name, timestamp):

    # Include timestamp in directory paths
    original_with_masks_dir = os.path.join(base_dir, "original_with_masks", method, timestamp)
    mask_visualizations_dir = os.path.join(base_dir, "mask_visualizations", method, timestamp)
    mask_files_dir = os.path.join(base_dir, "mask_files", method, timestamp)

    # Create directories if they don't exist
    os.makedirs(original_with_masks_dir, exist_ok=True)
    os.makedirs(mask_visualizations_dir, exist_ok=True)
    os.makedirs(mask_files_dir, exist_ok=True)

    # Prepare file paths
    original_path = os.path.join(original_with_masks_dir, file_name)
    mask_visualization_path = os.path.join(mask_visualizations_dir, os.path.splitext(file_name)[0] + '_mask.png')
    mask_file_path = os.path.join(mask_files_dir, os.path.splitext(file_name)[0] + '.npy')

    # Save original with masks (for visualization, boundaries are highlighted)
    original_with_mask = mark_boundaries(image, mask, color=(0, 1, 0))
    io.imsave(original_path, img_as_ubyte(original_with_mask))

    # Save mask visualization (binary mask)
    mask_visualization = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    io.imsave(mask_visualization_path, mask_visualization, check_contrast=False)

    # Save mask file (.npy format for further processing if needed)
    np.save(mask_file_path, mask)


def process_image_with_bboxes(image_path, bbox_folder, base_dir, timestamp):
    img = io.imread(image_path)
    bbox_file_path = os.path.join(bbox_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    if not os.path.exists(bbox_file_path):
        print(f"Bounding box file not found for {image_path}: {bbox_file_path}. Skipping.")
        return

    combined_masks = {}
    methods = ['Edge-Based+YOLO', 'Thresholding+YOLO', 'Watershed+YOLO']

    for method in methods:
        combined_masks[method] = np.zeros(img.shape[:2], dtype=np.uint8)

    bboxes = load_yolo_bboxes(bbox_file_path, img.shape)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox  # Extract bounding box coordinates
        cropped_img = img[y1:y2, x1:x2]

        # Apply segmentation methods to the cropped image
        for method in methods:
            mask = apply_segmentation(cropped_img, method)
            combined_masks[method][y1:y2, x1:x2] = np.maximum(combined_masks[method][y1:y2, x1:x2], mask)

    # Save the combined masks and other outputs
    for method, mask in combined_masks.items():
        save_images(img, mask, method, base_dir, os.path.basename(image_path), timestamp)

def process_folder(folder_path, bbox_folder, base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            process_image_with_bboxes(image_path, bbox_folder, base_dir, timestamp)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classical segmentation baselines inside YOLO boxes.")
    parser.add_argument("--images_folder", default="examples/GNV_benchmark_data_coco")
    parser.add_argument("--bbox_folder", default="outputs/GNV_benchmark_results/yolo_bboxes")
    parser.add_argument("--base_dir", default="outputs/GNV_benchmark_results/other_methods")
    args = parser.parse_args()

    images_folder = args.images_folder
    bbox_folder = args.bbox_folder
    base_dir = args.base_dir
    ensure_directory_exists(images_folder)
    ensure_directory_exists(bbox_folder)
    ensure_directory_exists(base_dir)


    process_folder(images_folder, bbox_folder, base_dir)
