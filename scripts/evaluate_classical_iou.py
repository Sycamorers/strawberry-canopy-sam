"""
This script calculates the Intersection over Union (IoU) between the predicted masks and the ground truth masks for a given set of images.
"""
from pycocotools.coco import COCO
import numpy as np
import os
from skimage.draw import polygon
import cv2
import json
import argparse


def coco_mask_to_binary(ann, coco, image_shape):
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
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    return intersection / union if union != 0 else 0

def load_and_process_predicted_mask(predicted_mask_path, image_shape):
    predicted_mask = np.load(predicted_mask_path)
    if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)
    elif predicted_mask.ndim == 3:
        predicted_mask = np.any(predicted_mask, axis=0).astype(np.uint8)

    # Convert predicted_mask from bool to uint8 if it is boolean
    if predicted_mask.dtype == bool:
        predicted_mask = predicted_mask.astype(np.uint8)

    predicted_mask = cv2.resize(predicted_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return predicted_mask




def main(annotations_path, images_dir, timestamp, predicted_base_dir, output_dir_base):
    """
    Run benchmarks for all specified segmentation methods on a COCO-style dataset with a single execution.

    Parameters:
    - annotations_path: str, path to the COCO-style annotation file
    - images_dir: str, directory where the original images are stored
    - timestamp: str, unique identifier for the test run, provided manually by the user
    """
    methods = ["Edge-Based+YOLO", "Thresholding+YOLO", "Watershed+YOLO"]

    for method_name in methods:

        predicted_masks_dir = os.path.join(predicted_base_dir, method_name, timestamp)
        output_dir = os.path.join(output_dir_base, method_name)

        # Ensure the predicted masks directory exists
        if not os.path.exists(predicted_masks_dir):
            print(f"Predicted masks directory {predicted_masks_dir} does not exist for {method_name}. Skipping.")
            continue

        # Load COCO annotations
        coco = COCO(annotations_path)
        iou_scores = []

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_name = f'iou_{method_name}_{timestamp}.txt'
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, 'w') as output_file:
            for img_id in coco.getImgIds():
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)

                predicted_mask_filename = img_info['file_name'].replace('.jpg', '.npy')
                predicted_mask_path = os.path.join(predicted_masks_dir, predicted_mask_filename)

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

        print(f"Benchmark for {method_name} completed. Results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IoU for classical segmentation baselines.")
    parser.add_argument("--annotations_path", default="examples/GNV_benchmark_data_coco/annotations.json")
    parser.add_argument("--images_dir", default="examples/GNV_benchmark_data_coco")
    parser.add_argument("--timestamp", required=True, help="Timestamp folder produced by run_classical_segmentation.py.")
    parser.add_argument(
        "--predicted_base_dir",
        default="outputs/GNV_benchmark_results/other_methods/mask_files",
        help="Directory containing method/timestamp mask folders.",
    )
    parser.add_argument("--output_dir", default="outputs/iou_results_other")
    args = parser.parse_args()

    main(
        args.annotations_path,
        args.images_dir,
        args.timestamp,
        args.predicted_base_dir,
        args.output_dir,
    )
