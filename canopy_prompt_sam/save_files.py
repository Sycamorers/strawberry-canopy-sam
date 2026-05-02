from ultralytics import YOLO  # YOLO model for object detection
import numpy as np
import torch
import cv2
import os
from datetime import datetime
import json
import argparse
import time

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pycocotools.coco import COCO
from skimage.draw import polygon
from segment_anything import sam_model_registry, SamPredictor
# EfficientSAM
from SAMs.EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# MobileSAM
from SAMs.MobileSAM.MobileSAMv2.mobilesamv2 import sam_model_registry as mobile_sam_model_registryv2
from SAMs.MobileSAM.MobileSAMv2.mobilesamv2 import SamPredictor as mobile_sam_model_SamPredictorv2



def save_combined_binary_masks(masks, img_path, mask_img_output_dir):
    """Saves all binary masks for a single image into one combined mask image."""
    if len(masks) == 0:
        return
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    base_name = os.path.basename(img_path)
    mask_filename = os.path.join(mask_img_output_dir, base_name)
    cv2.imwrite(mask_filename, combined_mask)


def save_image_with_masks(image, img_path, output_dir, case_name):
    """Saves the image with masks applied, including the case name in the folder path."""
    output_dir_with_case = os.path.join(output_dir, case_name)
    os.makedirs(output_dir_with_case, exist_ok=True)
    base_name = os.path.basename(img_path)
    output_filename = os.path.join(output_dir_with_case, base_name)
    cv2.imwrite(output_filename, image)

def save_masks_as_npy(all_refined_masks, img_path, mask_output_dir):
    """Saves all refined masks as a single .npy file, replacing the original image extension with .npy."""
    if not all_refined_masks:
        return

    npy_mask_filename = os.path.join(mask_output_dir, os.path.basename(img_path).rsplit('.', 1)[0] + '.npy')
    all_refined_masks_np = np.stack(all_refined_masks, axis=0)
    np.save(npy_mask_filename, all_refined_masks_np)

def save_yolo_format_boxes(boxes, img_dims, output_dir, img_path):
    """
    Saves bounding boxes in YOLO format. Each line in the output file corresponds to one bounding box,
    with the format: class_id x_center y_center width height, all normalized to [0, 1].
    """
    img_name = os.path.basename(img_path)
    txt_filename = img_name.rsplit('.', 1)[0] + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)

    h, w = img_dims  # Image dimensions
    with open(txt_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
