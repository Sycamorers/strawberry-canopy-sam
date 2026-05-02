from ultralytics import YOLO  # YOLO model for object detection
import numpy as np
import torch
import cv2
import os
from datetime import datetime
import json
import argparse
import time
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pycocotools.coco import COCO
from skimage.draw import polygon

from .overlap import extract_relative_area


def calculate_center_point(box):
    """Calculates the center point of a bounding box."""
    center_x = box[0] + (box[2] - box[0]) / 2
    center_y = box[1] + (box[3] - box[1]) / 2
    return center_x, center_y


def select_points(frt_box, position, method):
    """Selects background points based on the fruit box and its position relative to the plant box."""
    if method == '2p_6':
        offset_x = (frt_box[2] - frt_box[0]) / 6
        offset_y = (frt_box[3] - frt_box[1]) / 6
        if position == 'top_right':
            return [(frt_box[2] - offset_x, frt_box[1] + offset_y), (frt_box[2] - 2 * offset_x, frt_box[1] + 2 * offset_y)]
        elif position == 'top_left':
            return [(frt_box[0] + offset_x, frt_box[1] + offset_y), (frt_box[0] + 2 * offset_x, frt_box[1] + 2 * offset_y)]
        elif position == 'bottom_right':
            return [(frt_box[2] - offset_x, frt_box[3] - offset_y), (frt_box[2] - 2 * offset_x, frt_box[3] - 2 * offset_y)]
        else:  # bottom_left
            return [(frt_box[0] + offset_x, frt_box[3] - offset_y), (frt_box[0] + 2 * offset_x, frt_box[3] - 2 * offset_y)]

    if method == '2p_8':
        offset_x = (frt_box[2] - frt_box[0]) / 8
        offset_y = (frt_box[3] - frt_box[1]) / 8
        if position == 'top_right':
            return [(frt_box[2] - 2 * offset_x, frt_box[1] + 2 * offset_y), (frt_box[2] - 3 * offset_x, frt_box[1] + 3 * offset_y)]
        elif position == 'top_left':
            return [(frt_box[0] + 2 * offset_x, frt_box[1] + 2 * offset_y), (frt_box[0] + 3 * offset_x, frt_box[1] + 3 * offset_y)]
        elif position == 'bottom_right':
            return [(frt_box[2] - 2 * offset_x, frt_box[3] - 2 * offset_y), (frt_box[2] - 3 * offset_x, frt_box[3] - 3 * offset_y)]
        else:  # bottom_left
            return [(frt_box[0] + 2 * offset_x, frt_box[3] - 2 * offset_y), (frt_box[0] + 3 * offset_x, frt_box[3] - 3 * offset_y)]

    if method == '1p_6':
        offset_x = (frt_box[2] - frt_box[0]) / 6
        offset_y = (frt_box[3] - frt_box[1]) / 6
        if position == 'top_right':
            return [(frt_box[2] - 2 * offset_x, frt_box[1] + 2 * offset_y)]
        elif position == 'top_left':
            return [(frt_box[0] + 2 * offset_x, frt_box[1] + 2 * offset_y)]
        elif position == 'bottom_right':
            return [(frt_box[2] - 2 * offset_x, frt_box[3] - 2 * offset_y)]
        else:  # bottom_left
            return [(frt_box[0] + 2 * offset_x, frt_box[3] - 2 * offset_y)]

    if method == '1p_8':
        offset_x = (frt_box[2] - frt_box[0]) / 8
        offset_y = (frt_box[3] - frt_box[1]) / 8
        if position == 'top_right':
            return [(frt_box[2] - offset_x, frt_box[1] + offset_y)]
        elif position == 'top_left':
            return [(frt_box[0] + offset_x, frt_box[1] + offset_y)]
        elif position == 'bottom_right':
            return [(frt_box[2] - offset_x, frt_box[3] - offset_y)]
        else:  # bottom_left
            return [(frt_box[0] + offset_x, frt_box[3] - offset_y)]

    if method == '1p_center':
        center_point = calculate_center_point(frt_box)
        return [center_point]

    if method == '4p_corners':
        width = frt_box[2] - frt_box[0]
        height = frt_box[3] - frt_box[1]
        offset_x = width / 8
        offset_y = height / 8
        top_left = (frt_box[0] + offset_x, frt_box[1] + offset_y)
        top_right = (frt_box[2] - offset_x, frt_box[1] + offset_y)
        bottom_left = (frt_box[0] + offset_x, frt_box[3] - offset_y)
        bottom_right = (frt_box[2] - offset_x, frt_box[3] - offset_y)
        return [top_left, top_right, bottom_left, bottom_right]

    if method == '0p':
        return []




def extract_relative_area(overlap_coords, mask_shape, relative_position):
    overlap_array = np.array(overlap_coords)
    x_min, y_min = overlap_array.min(axis=0)
    x_max, y_max = overlap_array.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    relative_mask = np.zeros(mask_shape, dtype=np.uint8)

    if relative_position == 'top_left':
        selected_area = [[x, y] for [x, y] in overlap_coords if x < x_min + width // 2 and y < y_min + height // 2]
    elif relative_position == 'top_right':
        selected_area = [[x, y] for [x, y] in overlap_coords if x >= x_min + width // 2 and y < y_min + height // 2]
    elif relative_position == 'bottom_left':
        selected_area = [[x, y] for [x, y] in overlap_coords if x < x_min + width // 2 and y >= y_min + height // 2]
    else:  # bottom_right
        selected_area = [[x, y] for [x, y] in overlap_coords if x >= x_min + width // 2 and y >= y_min + height // 2]

    for [x, y] in selected_area:
        relative_mask[y, x] = 3

    return selected_area, relative_mask



def random_sampling(overlap_coords, num_points, relative_position, mask_shape):
    relative_coords, relative_mask = extract_relative_area(overlap_coords, mask_shape, relative_position)

    # if len(relative_coords) <= num_points:
    #     return relative_coords, relative_mask

    indices = np.random.choice(len(relative_coords), num_points, replace=False)
    points = [relative_coords[i] for i in indices]
    return points, relative_mask

def uniform_grid_sampling(overlap_coords, num_points, relative_position, mask_shape):
    relative_coords, relative_mask = extract_relative_area(overlap_coords, mask_shape, relative_position)
    relative_array = np.array(relative_coords)
    x_min, y_min = relative_array.min(axis=0)
    x_max, y_max = relative_array.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    grid_size = int(np.ceil(np.sqrt(num_points)))
    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)

    points = []
    for gx in grid_x:
        for gy in grid_y:
            gx = int(round(gx))
            gy = int(round(gy))
            if (gx, gy) in relative_coords:
                points.append((gx, gy))
                if len(points) >= num_points:
                    return points, relative_mask

    return points[:num_points], relative_mask




def stratified_sampling(overlap_coords, num_points, relative_position, mask_shape):
    relative_coords, relative_mask = extract_relative_area(overlap_coords, mask_shape, relative_position)
    relative_array = np.array(relative_coords)
    x_min, y_min = relative_array.min(axis=0)
    x_max, y_max = relative_array.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    num_regions = int(np.ceil(np.sqrt(num_points)))
    region_width = max(1, width // num_regions)
    region_height = max(1, height // num_regions)

    points = []
    for i in range(num_regions):
        for j in range(num_regions):
            region_coords = [(x, y) for (x, y) in relative_coords
                             if i * region_width <= x - x_min < (i + 1) * region_width
                             and j * region_height <= y - y_min < (j + 1) * region_height]
            if region_coords:
                point = random.choice(region_coords)
                points.append(point)
                if len(points) >= num_points:
                    return points, relative_mask

    return points[:num_points], relative_mask



def edge_based_sampling(overlap_coords, num_points, relative_position, mask_shape):
    relative_coords, relative_mask = extract_relative_area(overlap_coords, mask_shape, relative_position)
    relative_array = np.array(relative_coords)
    x_min, y_min = relative_array.min(axis=0)
    x_max, y_max = relative_array.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    overlap_array = np.zeros((height, width), dtype=np.uint8)
    for (x, y) in relative_coords:
        overlap_array[y - y_min, x - x_min] = 1

    edges = cv2.Canny(overlap_array, 100, 200)
    edge_points = np.argwhere(edges > 0)
    edge_points = [(pt[1] + x_min, pt[0] + y_min) for pt in edge_points]

    if len(edge_points) >= num_points:
        indices = np.random.choice(len(edge_points), num_points, replace=False)
        points = [edge_points[i] for i in indices]
    else:
        points = edge_points

    return points, relative_mask


def density_based_sampling(overlap_coords, num_points, relative_position, mask_shape):
    relative_coords, relative_mask = extract_relative_area(overlap_coords, mask_shape, relative_position)
    relative_array = np.array(relative_coords)
    x_min, y_min = relative_array.min(axis=0)
    x_max, y_max = relative_array.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    density_map = np.zeros((height, width), dtype=int)
    for (x, y) in relative_coords:
        density_map[y - y_min, x - x_min] += 1

    flat_density = density_map.flatten()
    prob_density = flat_density / np.sum(flat_density)
    indices = np.random.choice(np.arange(len(flat_density)), num_points, p=prob_density)
    points = [(idx % width + x_min, idx // width + y_min) for idx in indices]

    return points, relative_mask



def get_points_based_on_strategy(overlap_coords, overlap_mask, num_points, strategy, relative_position):
    """
    Select points based on the given strategy within the overlapping area.

    Parameters:
    overlap_coords (list): List of overlapping pixel coordinates.
    overlap_mask (np.ndarray): Binary mask representing the overlapping area.
    num_points (int): Number of points to select.
    strategy (str): Strategy for selecting points.
    relative_position (str): Relative position to focus the point selection.

    Returns:
    tuple: List of selected points and the relative area mask.
    """
    mask_shape = overlap_mask.shape
    if strategy == 'uniform_grid':
        points, relative_mask = uniform_grid_sampling(overlap_coords, num_points, relative_position, mask_shape)
    elif strategy == 'random':
        points, relative_mask = random_sampling(overlap_coords, num_points, relative_position, mask_shape)
    elif strategy == 'stratified':
        points, relative_mask = stratified_sampling(overlap_coords, num_points, relative_position, mask_shape)
    elif strategy == 'edge_based':
        points, relative_mask = edge_based_sampling(overlap_coords, num_points, relative_position, mask_shape)
    elif strategy == 'density_based':
        points, relative_mask = density_based_sampling(overlap_coords, num_points, relative_position, mask_shape)

    return points, relative_mask
