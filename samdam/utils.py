from ultralytics import YOLO  # YOLO model for object detection
import numpy as np
import torch
import cv2
import os
import sys
from datetime import datetime
import json
import argparse
import time


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pycocotools.coco import COCO
from skimage.draw import polygon
from segment_anything import sam_model_registry, SamPredictor

from .paths import MOBILE_SAM_V2_ROOT, require_file, weight_path

if str(MOBILE_SAM_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILE_SAM_V2_ROOT))

# EfficientSAM
from SAMs.EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# MobileSAM
from SAMs.MobileSAM.MobileSAMv2.mobilesamv2 import sam_model_registry as mobile_sam_model_registryv2
from SAMs.MobileSAM.MobileSAMv2.mobilesamv2 import SamPredictor as mobile_sam_model_SamPredictorv2

def save_combined_binary_masks(masks, img_path, output_dir):
    """Saves all binary masks for a single image into one combined mask image."""
    if len(masks) == 0:
        return
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    base_name = os.path.basename(img_path)
    mask_filename = f"{base_name.split('.')[0]}.png"
    mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_path, combined_mask)


def draw_points_on_image(image, coords, labels, newmethod, point_size=3):
    if newmethod:
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        for point in pos_points:
            cv2.circle(image, (int(point[0]), int(point[1])), point_size, (0, 255, 0), -1)  # Green for positive points

        for point in neg_points:
            cv2.circle(image, (int(point[0]), int(point[1])), point_size, (0, 0, 255), -1)  # Red for negative points
    else:
        for point in coords:
            cv2.circle(image, (int(point[0]), int(point[1])), point_size, (0, 0, 255), -1)


def is_inside(box1, box2):
    """Checks if box1 is inside box2."""
    return all([box1[0] >= box2[0], box1[1] >= box2[1], box1[2] <= box2[2], box1[3] <= box2[3]])

def find_relative_position(frt_box, plt_box):
    """Finds the relative position of a fruit box within a plant box."""
    plt_center_x, plt_center_y = (plt_box[0] + plt_box[2]) / 2, (plt_box[1] + plt_box[3]) / 2
    frt_center_x, frt_center_y = (frt_box[0] + frt_box[2]) / 2, (frt_box[1] + frt_box[3]) / 2

    if frt_center_x < plt_center_x and frt_center_y > plt_center_y:
        return 'bottom_left'
    elif frt_center_x >= plt_center_x and frt_center_y > plt_center_y:
        return 'bottom_right'
    elif frt_center_x < plt_center_x and frt_center_y <= plt_center_y:
        return 'top_left'
    else:
        return 'top_right'

def apply_masks_to_image(image, masks):
    """Applies segmentation masks to the original image with different colors based on labels."""
    # Create colored images for each possible label
    green_img = np.zeros(image.shape, image.dtype)
    green_img[:, :] = [0, 255, 0]  # Green for label 1

    light_blue_img = np.zeros(image.shape, image.dtype)
    light_blue_img[:, :] = [255, 255, 0]  # Light Blue for label 2

    purple_img = np.zeros(image.shape, image.dtype)
    purple_img[:, :] = [255, 0, 255]  # Purple for label 3

    # Apply masks in reverse order of priority
    for mask in masks:
        # Apply colors to the image based on the mask label
        mask_label_3 = (mask == 3)
        if np.any(mask_label_3):
            image[mask_label_3] = cv2.addWeighted(image, 0.5, purple_img, 0.5, 0)[mask_label_3]

        mask_label_2 = (mask == 2)
        if np.any(mask_label_2):
            image[mask_label_2] = cv2.addWeighted(image, 0.5, light_blue_img, 0.5, 0)[mask_label_2]

        mask_label_1 = (mask == 1)
        if np.any(mask_label_1):
            image[mask_label_1] = cv2.addWeighted(image, 0.5, green_img, 0.5, 0)[mask_label_1]

    return image


def load_models(which_sam="SAM"):
    """Loads the YOLO and SAM models."""
    # sb_yolo_model = YOLO('./models_wts/sb_new.pt')
    # plt_yolo_model = YOLO('./models_wts/plt.pt')
    sb_yolo_model = YOLO(require_file(weight_path("SAMDAM_fruit.pt"), "fruit YOLO weight"))
    plt_yolo_model = YOLO(require_file(weight_path("SAMDAM_plant.pt"), "plant YOLO weight"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_predictor = None
    if which_sam == "SAM":
        sam_checkpoint = require_file(weight_path("sam_vit_h_4b8939.pth"), "SAM ViT-H checkpoint")
        model_type = "vit_h"
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        sam_predictor = SamPredictor(sam_model)
    mobilesamv2 = None
    if which_sam == "MobileSAM":
        prompt_guided_path = require_file(
            MOBILE_SAM_V2_ROOT / "PromptGuidedDecoder" / "Prompt_guided_Mask_Decoder.pt",
            "MobileSAMv2 prompt-guided decoder",
        )
        prompt_guided_decoder = mobile_sam_model_registryv2["PromptGuidedDecoder"](prompt_guided_path)
        mobilesamv2 = sam_model_registry["vit_h"]()
        mobilesamv2.prompt_encoder = prompt_guided_decoder["PromtEncoder"]
        mobilesamv2.mask_decoder = prompt_guided_decoder["MaskDecoder"]
    damv2 = "depth-anything/Depth-Anything-V2-Small-hf"


    return sb_yolo_model, plt_yolo_model, sam_predictor, mobilesamv2, damv2


def setup_directories(base_dir, timestamp, pointcase, which_sam, num_points,strategy, new_method):
    """
    Sets up directories for storing outputs, including a new directory for YOLO format bounding boxes.
    """
    if new_method:
        base_dir = os.path.join(base_dir, which_sam, pointcase, timestamp,"new")
    else:
        base_dir = os.path.join(base_dir, which_sam, pointcase, timestamp,'old')
    point_case = str(num_points) + 'p'
    base_dir = os.path.join(base_dir, point_case, strategy)
    output_dir = os.path.join(base_dir, 'original_with_masks')
    mask_output_dir = os.path.join(base_dir, 'mask_files')
    mask_img_output_dir = os.path.join(base_dir, 'mask_visualizations')
    combined_visualization_dir = os.path.join(base_dir, 'combined_visualizations')
    bbox_output_dir = os.path.join(base_dir, 'yolo_bboxes')
    threeD_output_dir = os.path.join(base_dir, '3d_plots')


    for dir in [output_dir, mask_output_dir, mask_img_output_dir, combined_visualization_dir, bbox_output_dir]:
        print(f"Creating directory: {dir}")
        os.makedirs(dir, exist_ok=True)

    return output_dir, mask_output_dir, mask_img_output_dir, combined_visualization_dir, bbox_output_dir,threeD_output_dir


def coco_mask_to_binary(ann, coco, image_shape):
    """Converts COCO annotations to binary mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if isinstance(ann['segmentation'], list):
        for segmentation in ann['segmentation']:
            poly = np.array(segmentation).reshape((-1, 2))
            rr, cc = polygon(poly[:, 1], poly[:, 0], shape=image_shape)
            mask[rr, cc] = 1
    else:
        mask = coco.annToMask(ann)
    return mask


def calculate_iou(predicted_mask, ground_truth_mask):
    """Calculates Intersection over Union (IoU) between two masks."""
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    return intersection / union if union != 0 else 0

def load_and_process_predicted_mask(predicted_mask_path, image_shape):
    """Loads and processes predicted mask."""
    predicted_mask = np.load(predicted_mask_path)
    if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)
    elif predicted_mask.ndim == 3:
        predicted_mask = np.any(predicted_mask, axis=0).astype(np.uint8)

    if predicted_mask.dtype == bool:
        predicted_mask = predicted_mask.astype(np.uint8)

    predicted_mask = cv2.resize(predicted_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return predicted_mask



def calculate_and_save_iou(annotations_path, predicted_masks_dir, output_dir, method_name, use_preliminary_masks, num_points, strategy, new_method, box_type, point_case):
    """Calculates IoU and saves results."""
    coco = COCO(annotations_path)
    iou_scores = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    premask = 'premask'
    nopremask = 'nopremask'
    if use_preliminary_masks:
        method_case_suffix = f'{point_case}_{method_name}_{premask}_{box_type}'
    else:
        method_case_suffix = f'{point_case}_{method_name}_{nopremask}_{box_type}'

    if new_method:
        output_file_name = f'iou_{method_case_suffix}_{num_points}_{strategy}_{timestamp}_new.txt'
    else:
        output_file_name = f'iou_{method_case_suffix}_{num_points}_{strategy}_{timestamp}_old.txt'
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, 'w') as output_file:
        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            predicted_mask_filename = img_info['file_name'].rsplit('.jpg', 1)[0] + '.npy'
            predicted_mask_path = os.path.join(predicted_masks_dir, predicted_mask_filename)

            print(f"Checking for file: {predicted_mask_path}")
            if os.path.exists(predicted_mask_path):
                image_shape = (img_info['height'], img_info['width'])
                predicted_mask = load_and_process_predicted_mask(predicted_mask_path, image_shape)

                combined_gt_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                for ann in anns:
                    ground_truth_mask = coco_mask_to_binary(ann, coco, image_shape)
                    combined_gt_mask = np.logical_or(combined_gt_mask, ground_truth_mask).astype(np.uint8)

                iou = calculate_iou(predicted_mask, combined_gt_mask)
                iou_scores.append(iou)
                output_file.write(f'IoU for image {img_info["file_name"]}: {iou}\n')
            else:
                output_file.write(f'Predicted mask for image {img_info["file_name"]} not found.\n')

        average_iou = np.mean(iou_scores) if iou_scores else 0
        output_file.write(f'Average IoU: {average_iou}\n')
    return average_iou
