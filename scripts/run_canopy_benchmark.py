"""
This script combines the two Python scripts, which is calculating Intersection over Union (IoU) and
segmenting fruits using various models like YOLO, SAM, and their derivatives. The combined script
will facilitate a workflow that segments fruits from images and then evaluates the segmentation
performance using IoU.
"""
from collections import Counter
from ultralytics import YOLO  # YOLO model for object detection
import numpy as np
import torch
import cv2
import os
import sys
from pathlib import Path
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILE_SAM_V2_ROOT = PROJECT_ROOT / "SAMs" / "MobileSAM" / "MobileSAMv2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MOBILE_SAM_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILE_SAM_V2_ROOT))

# SAM
from segment_anything import sam_model_registry, SamPredictor
# EfficientSAM
from SAMs.EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# MobileSAM
from SAMs.MobileSAM.MobileSAMv2.mobilesamv2 import sam_model_registry as mobile_sam_model_registryv2
from SAMs.MobileSAM.MobileSAMv2.mobilesamv2 import SamPredictor as mobile_sam_model_SamPredictorv2


# import funcs
from canopy_prompt_sam.plot3d import get_3d_rst
from canopy_prompt_sam.box_selection import filter_boxes_at_edges
from canopy_prompt_sam.point_selection import select_points,get_points_based_on_strategy
from canopy_prompt_sam.overlap import is_overlap_bm, calculate_overlap_area_bm,calculate_overlap_area_bb
from canopy_prompt_sam.save_files import save_combined_binary_masks,save_masks_as_npy, save_image_with_masks,save_yolo_format_boxes
from canopy_prompt_sam.utils import draw_points_on_image,is_inside,find_relative_position,apply_masks_to_image,load_models,setup_directories,coco_mask_to_binary,calculate_iou,load_and_process_predicted_mask,calculate_and_save_iou
from canopy_prompt_sam.paths import require_file, weight_path



def process_image(img_path, output_dir, mask_output_dir, mask_img_output_dir, combined_visualization_dir, bbox_output_dir,
                  sb_yolo_model, plt_yolo_model, which_sam, which_efficient_sam, sam_predictor,
                  mobilesamv2, mobv2_encoder_type,
                  pointcase, method, box_type,
                  new_method= True,
                  strategy = "uniform_grid", num_points = 1,
                  use_exclusive_points=True, use_preliminary_masks=True):
    """Processes each image for segmentation and mask generation, and saves images with masks."""
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # SAM
    # input the image to SAM
    if which_sam == 'SAM':
        sam_predictor.set_image(image_rgb)

    # EfficientSAM
    # no image input
    if which_sam == 'EfficientSAM':
        if which_efficient_sam == 'EfficientSAM_t':
            EfficientSAM_t = build_efficient_sam_vitt()
        elif which_efficient_sam == 'EfficientSAM_s':
            EfficientSAM_s = build_efficient_sam_vits()

    # MobileSAM
    mobv2_predictor = None
    if which_sam == 'MobileSAM':
        encoder_path = {
            'efficientvit_l2': require_file(weight_path('l2.pt'), 'MobileSAM efficientvit_l2 encoder'),
            'tiny_vit': require_file(weight_path('mobile_sam.pt'), 'MobileSAM tiny_vit encoder'),
            'sam_vit_h': require_file(weight_path('sam_vit_h.pt'), 'MobileSAM SAM ViT-H encoder'),
        }
        image_encoder = mobile_sam_model_registryv2[mobv2_encoder_type](encoder_path[mobv2_encoder_type])
        mobilesamv2.image_encoder = image_encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobilesamv2.to(device=device)
        mobilesamv2.eval()
        mobv2_predictor = mobile_sam_model_SamPredictorv2(mobilesamv2)
        mobv2_predictor.set_image(image)

    sb_yolo_result = sb_yolo_model([img_path])[0]
    plt_yolo_result = plt_yolo_model([img_path])[0]

    sb_boxes = sb_yolo_result.boxes.xyxy.cpu().numpy()
    sb_labels = sb_yolo_result.boxes.cls.cpu().numpy()

    # get plant
    plt_boxes = plt_yolo_result.boxes.xyxy.cpu().numpy() # including weeds
    plt_labels = plt_yolo_result.boxes.cls.cpu().numpy()
    plt_boxes = plt_boxes[np.isin(plt_labels, [0])]

    if pointcase == "G":
        frt_boxes = sb_boxes[np.isin(sb_labels, [1, 2, 3, 4])]
    elif pointcase == "no_G":
        frt_boxes = sb_boxes[np.isin(sb_labels, [2, 3, 4])]
    elif pointcase == "no_GW":
        frt_boxes = sb_boxes[np.isin(sb_labels, [3, 4])]

    # Save plant bounding boxes in YOLO format
    img_dims = image.shape[:2]
    save_yolo_format_boxes(plt_boxes, img_dims, bbox_output_dir, img_path)

    all_refined_masks = []
    print(f'img_path: {img_path}')

    plot_background_points = []
    plot_background_lables = []
    selected_overlaps = []
    selected_relative_overlaps = []
    unselected_boxes_list = []
    selected_boxes_list = []
    old_overlap_boxes = []
    hollow_area_list = []


    for plt_index, plt_box in enumerate(plt_boxes):
        background_points = []

        # only in this condition for new method since everything is needed
        if new_method and use_preliminary_masks and use_exclusive_points:
            background_points = []
            start_time = time.time()
            # SAM
            if which_sam == 'SAM':
                # mask is first output
                # the third is feature
                preliminary_masks, _, preliminary_masks_input = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=plt_box[None, :],
                    multimask_output=False
                )
                for pre_mask in preliminary_masks:
                    pre_mask_binary = (pre_mask > 0).astype(np.uint8)
                    selected_boxes, unselected_boxes, hollow_area= filter_boxes_at_edges(pre_mask_binary, frt_boxes)
                    unselected_boxes_list.extend(unselected_boxes)
                    selected_boxes_list.extend(selected_boxes)
                    hollow_area_list.append(hollow_area)
                    for box in selected_boxes:
                        if is_overlap_bm(box,pre_mask):
                            overlap_coords, overlap_mask = calculate_overlap_area_bm(box, pre_mask)
                            selected_overlaps.append(overlap_mask)
                            relative_position = find_relative_position(box, plt_box)
                            points,selected_relative_mask = get_points_based_on_strategy(overlap_coords, overlap_mask, num_points, strategy, relative_position)
                            # print("points:", points)
                            selected_relative_overlaps.append(selected_relative_mask)
                            if not points:
                                continue
                            background_points.extend(points)
                        else:
                            continue

                if not background_points:
                    background_points = None
                    background_labels = None
                else:
                    background_labels = np.zeros(len(background_points))
                    background_points = np.array(background_points)

                refined_masks, _, _ = sam_predictor.predict(
                    point_coords=background_points,
                    point_labels=background_labels,
                    mask_input=preliminary_masks_input, # feature
                    box=plt_box[None, :],
                    multimask_output=False
                )







            # EfficientSAM
            elif which_sam == 'EfficientSAM':
                return print("EfficientSAM does not support preliminary masks")
            # MobileSAM
            else:


                preliminary_masks, _, preliminary_masks_input = mobv2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=plt_box[None, :],
                    multimask_output=False
                )

                for pre_mask in preliminary_masks:
                    pre_mask_binary = (pre_mask > 0).astype(np.uint8)
                    selected_boxes, unselected_boxes, hollow_area= filter_boxes_at_edges(pre_mask_binary, frt_boxes)
                    unselected_boxes_list.extend(unselected_boxes)
                    selected_boxes_list.extend(selected_boxes)

                    hollow_area_list.append(hollow_area)

                    for box in selected_boxes:
                        if is_overlap_bm(box,pre_mask):
                            overlap_coords, overlap_mask = calculate_overlap_area_bm(box, pre_mask)
                            selected_overlaps.append(overlap_mask)
                            relative_position = find_relative_position(box, plt_box)
                            points,selected_relative_mask = get_points_based_on_strategy(overlap_coords, overlap_mask, num_points, strategy, relative_position)
                            # print("points:", points)
                            selected_relative_overlaps.append(selected_relative_mask)

                            if not points:
                                continue

                            background_points.extend(points)

                        else:
                            continue

                if not background_points:
                    background_points = None
                    background_labels = None
                else:
                    background_labels = np.zeros(len(background_points))
                    background_points = np.array(background_points)


                refined_masks, _, _ = mobv2_predictor.predict(
                    point_coords=background_points,
                    point_labels=background_labels,
                    mask_input=preliminary_masks_input,
                    box=plt_box[None, :],
                    multimask_output=False
                )

        # original method
        else:

            if use_exclusive_points:
                # using overlapping area
                if box_type == 'overlapbox':
                    for frt_box in frt_boxes:
                        overlap_box = calculate_overlap_area_bb(frt_box, plt_box)
                        if overlap_box:
                            position = find_relative_position(overlap_box, plt_box)
                            points = select_points(overlap_box, position, method)
                            points = list(points)
                            background_points.extend(points)
                            old_overlap_boxes.append(overlap_box)

                else:
                    for frt_box in frt_boxes:
                        if is_inside(frt_box, plt_box):
                            position = find_relative_position(frt_box, plt_box)
                            points = select_points(frt_box, position, method)
                            points = list(points)
                            background_points.extend(points)
                            old_overlap_boxes.append(frt_box)
                background_labels = np.zeros(len(background_points))
                background_points = np.array(background_points)
                if isinstance(background_points, np.ndarray) and background_points.size == 0:
                    background_points = None
                    background_labels = None
            else:
                background_points = None # for without points
                background_labels = None

        # with preliminary masks
            if use_preliminary_masks:
                start_time = time.time()
                # SAM
                if which_sam == 'SAM':
                    _, _, preliminary_masks = sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=plt_box[None, :],
                        multimask_output=False
                    )

                    refined_masks, _, _ = sam_predictor.predict(
                        point_coords=background_points,
                        point_labels=background_labels,
                        mask_input=preliminary_masks,
                        box=plt_box[None, :],
                        multimask_output=False
                    )
                # EfficientSAM
                elif which_sam == 'EfficientSAM':
                    return print("EfficientSAM does not support preliminary masks")
            # MobileSAM
                else:
                    _, _, preliminary_masks = mobv2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=plt_box[None, :],
                        multimask_output=False
                    )

                    refined_masks, _, _ = mobv2_predictor.predict(
                        point_coords=background_points,
                        point_labels=background_labels,
                        mask_input=preliminary_masks,
                        box=plt_box[None, :],
                        multimask_output=False
                    )

            else:
                # without preliminary masks
                # SAM
                start_time = time.time()

                if which_sam == 'SAM':
                    refined_masks, _, _ = sam_predictor.predict(
                        point_coords=background_points,
                        point_labels=background_labels,
                        mask_input=None,
                        box=plt_box[None, :],
                        multimask_output=False
                    )
                # EfficientSAM
                elif which_sam == 'EfficientSAM':
                    if which_efficient_sam == 'EfficientSAM_t':
                        refined_masks, _ = EfficientSAM_t(
                            image_rgb[None, ...],
                            background_points[None, ...],
                            background_labels[None, ...],
                        )
                    elif which_efficient_sam == 'EfficientSAM_s':
                        refined_masks, _ = EfficientSAM_s(
                            image_rgb[None, ...],
                            background_points[None, ...],
                            background_labels[None, ...],
                        )
                # MobileSAM
                else:
                    refined_masks, _, _ = mobv2_predictor.predict(
                        point_coords=background_points,
                        point_labels=background_labels,
                        mask_input=None,
                        box=plt_box[None, :],
                        multimask_output=False
                    )

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(which_sam + f" processing time: {elapsed_time_ms} ms") # here it includes all the searching time




        if refined_masks.size > 0:
            all_refined_masks.append(refined_masks[0])
        if background_points is not None:
            plot_background_points.extend(background_points)
        if background_labels is not None:
            plot_background_lables.extend(background_labels)


    # get all the masks: SAM, overlap, relative selected
        # get all the masks: SAM, overlap, relative selected
    all_mask = []
    if all_refined_masks:
        all_mask.extend(all_refined_masks)
    if selected_overlaps:
        all_mask.extend(selected_overlaps)  # selected_overlaps is labeled as 2
    if selected_relative_overlaps:
        all_mask.extend(selected_relative_overlaps)  # selected_relative_overlaps is labeled as 3

    if all_refined_masks:
        # mask saving and visualization
        save_combined_binary_masks(all_refined_masks, img_path, mask_img_output_dir)  # only save SAM mask for IoU
        image_with_sam_masks = apply_masks_to_image(image.copy(), all_refined_masks)
        image_with_all_masks = apply_masks_to_image(image.copy(), all_mask)
        image_with_hollow_area = apply_masks_to_image(image.copy(), hollow_area_list)
        save_image_with_masks(image_with_sam_masks, img_path, output_dir, "SAM_mask")  # SAM masks
        save_image_with_masks(image_with_all_masks, img_path, output_dir, "ALL_mask")  # all masks
        save_image_with_masks(image_with_hollow_area, img_path, output_dir, "HOLLOW_mask")  # original image
        save_masks_as_npy(all_refined_masks, img_path, mask_output_dir)  # only save SAM mask for IoU

        # include everything in the visualization
        image_with_combined_visualization = image.copy()
        hollow_boxes_visualization = image.copy()
        image_with_combined_visualization = apply_masks_to_image(image_with_combined_visualization, all_mask)
        if new_method:
            box_counter = Counter(tuple(box) for box in selected_boxes_list)
            if unselected_boxes_list:
                for box in unselected_boxes_list:
                    if not isinstance(box, list):
                        box = box.tolist()
                    cv2.rectangle(image_with_combined_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)  # color: white
                    cv2.rectangle(hollow_boxes_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
            if selected_boxes_list:
                for box in selected_boxes_list:
                    if not isinstance(box, list):
                        box = box.tolist()
                    box_tuple = tuple(box)
                    if box_counter[box_tuple] > 1:
                        color = (128, 0, 128)  # color: purple
                    else:
                        color = (0, 0, 255)  # color: red
                    cv2.rectangle(image_with_combined_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.rectangle(hollow_boxes_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        else:
            for box in old_overlap_boxes:
                cv2.rectangle(image_with_combined_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)


        for box in plt_boxes:
            if not isinstance(box, list):
                box = box.tolist()
            cv2.rectangle(image_with_combined_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # color: blue
            cv2.rectangle(hollow_boxes_visualization, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # color: blue


        # image_with_combined_visualization = apply_masks_to_image(image_with_combined_visualization, all_mask)
        hollow_boxes_visualization = apply_masks_to_image(hollow_boxes_visualization, hollow_area_list)


        # plot all points on the image
        if new_method:
            if plot_background_points is not None and plot_background_lables is not None:
                if len(plot_background_points) == len(plot_background_lables):
                    for points, labels in zip(plot_background_points, plot_background_lables):
                        draw_points_on_image(image_with_combined_visualization, points, labels, new_method)
                else:
                    print("Error: Length of points and labels arrays do not match.")
            else:
                print("Error: plot_background_points or plot_background_lables is None.")
        else:
            # print(plot_background_points)
            draw_points_on_image(image_with_combined_visualization, plot_background_points, plot_background_lables, new_method)


        save_image_with_masks(image_with_combined_visualization, img_path, combined_visualization_dir, "everything")
        save_image_with_masks(hollow_boxes_visualization, img_path, combined_visualization_dir, "hollow_boxes")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarking with configurations.')
    parser.add_argument('--pointcase', type=str, required=True, help='Point case')
    parser.add_argument('--method', type=str, required=True, help='Method for processing')
    parser.add_argument('--pre_mask', type=str, required=True, choices=['with_pre_mask', 'no_pre_mask'], help='Pre-mask configuration')
    parser.add_argument('--box_type', type=str, required=True, choices=['wholebox', 'overlapbox'], help='Type of bounding box')
    parser.add_argument('--use_preliminary_masks', type=lambda x: (str(x).lower() == 'true'), required=True, help='Use preliminary masks')
    parser.add_argument('--use_exclusive_points', type=lambda x: (str(x).lower() == 'true'), required=True, help='Use exclusive points for method other than 0p')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for benchmark results')
    parser.add_argument('--image_folder', type=str, required=True, help='Image folder')
    parser.add_argument('--annotations_path', type=str, required=True, help='Path to the annotations JSON file.')
    parser.add_argument('--iou_output_dir', type=str, required=True, help='Output directory for IoU results.')
    parser.add_argument('--method_name', type=str, required=True, help='Method name for identification.')

    # newly added
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument("--mobv2_encoder_type", default='efficientvit_l2', choices=['tiny_vit', 'sam_vit_h', 'efficientvit_l2'], help="choose the MobileSAM encoder")
    parser.add_argument("--which_sam", choices=['SAM', 'EfficientSAM', 'MobileSAM'], default='SAM', help="choose the segmentation backbone")
    parser.add_argument("--which_efficient_sam", choices=['EfficientSAM_t', 'EfficientSAM_s'], default='EfficientSAM_t', help="choose the EfficientSAM variant")
    parser.add_argument("--newmethod", type=lambda x: (str(x).lower() == 'true'), required=True, help='Use new method')



    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "_" + args.method + "_" + args.pre_mask
    suffix = timestamp + method_name + "_" + args.box_type



    image_paths = [os.path.join(args.image_folder, file) for file in os.listdir(args.image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths.sort()

    sb_yolo_model, plt_yolo_model, sam_predictor, mobilesamv2, damv2 = load_models(args.which_sam)

    best_strategy = None
    best_iou = 0
    avg_iou_lst = []


    for num_points in range(2, 3):
        # for strategy in ['random', 'stratified', 'density_based']:  # Define your strategies here
        for strategy in ['random']:  # Define your strategies here
            output_dir, mask_output_dir, mask_img_output_dir, combined_visualization_dir, bbox_output_dir, threeD_output_dir = setup_directories(args.base_dir, suffix, args.pointcase, args.which_sam, num_points, strategy, args.newmethod)

            iou_sum = 0
            for img_path in image_paths:
                # Assuming a function get_points_based_on_strategy exists that returns points based on strategy
                process_image(img_path, output_dir, mask_output_dir, mask_img_output_dir, combined_visualization_dir, bbox_output_dir,
                              sb_yolo_model, plt_yolo_model, args.which_sam, args.which_efficient_sam, sam_predictor, mobilesamv2,
                              args.mobv2_encoder_type,
                              args.pointcase, args.method, args.box_type,
                              args.newmethod,
                              strategy=strategy, num_points=num_points,
                              use_exclusive_points=args.use_exclusive_points, use_preliminary_masks=args.use_preliminary_masks)

            avg_iou = calculate_and_save_iou(
            args.annotations_path, mask_output_dir, args.iou_output_dir, args.method_name, args.use_preliminary_masks, num_points, strategy, new_method=args.newmethod,box_type=args.box_type,point_case= args.pointcase
        )
            avg_iou_lst.append((num_points, strategy, avg_iou))

            if avg_iou > best_iou:
                best_iou = avg_iou
                best_strategy = (num_points, strategy)

    # avg_iou_folder = "avg_ious"
    # if not os.path.exists(avg_iou_folder):
    #     os.makedirs(avg_iou_folder)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{avg_iou_folder}/iou_results_{timestamp}.txt"

    # with open(filename, 'w') as f:
    #     for num_points, strategy, avg_iou in avg_iou_lst:
    #         f.write(f"Num Points: {num_points}, Strategy: {strategy}, Avg IoU: {avg_iou}, New:{str(args.newmethod)}\n")
    #     f.write(f"\nBest strategy: {best_strategy} with IoU: {best_iou}\n")


    best_num_points, best_strategy_name = best_strategy
    output_dir, mask_output_dir, mask_img_output_dir, combined_visualization_dir, bbox_output_dir, threeD_output_dir = setup_directories(args.base_dir, suffix, args.pointcase, args.which_sam, best_num_points, best_strategy_name,args.newmethod)

    for img_path in image_paths:

        img_basename = os.path.basename(img_path)
        masks_filename = os.path.splitext(img_basename)[0] + ".npy"
        masks_path = os.path.join(mask_output_dir, masks_filename)
        get_3d_rst(img_path, masks_path, damv2, threeD_output_dir)
