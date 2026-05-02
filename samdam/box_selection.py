import numpy as np
import math
from scipy.ndimage import binary_erosion

def calculate_average_diagonal_length(boxes):
    """
    Calculates the average diagonal length of a list of bounding boxes.

    Parameters:
    boxes (list): List of bounding boxes (each box is a tuple of (x1, y1, x2, y2)).

    Returns:
    float: Average diagonal length of the bounding boxes.
    """
    diagonals = [math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in boxes]
    return np.mean(diagonals)

# def define_hollow_edge_area(mask, avg_diagonal):
#     """
#     Defines the hollow edge area of the mask based on the average diagonal length.

#     Parameters:
#     mask (np.ndarray): A 2D numpy array representing the segmentation mask.
#     avg_diagonal (float): Average diagonal length of the bounding boxes.

#     Returns:
#     np.ndarray: A binary mask representing the hollow edge area.
#     """
#     hollow_mask = np.zeros_like(mask, dtype=np.uint8)
#     y_indices, x_indices = np.indices(mask.shape)

#     # Find the boundary of the mask
#     mask_boundary = np.argwhere(mask == 1)
#     y_min, x_min = mask_boundary.min(axis=0)
#     y_max, x_max = mask_boundary.max(axis=0)

#     # Create the outer and inner boundaries of the hollow area
#     outer_boundary = (x_indices >= x_min) & (x_indices <= x_max) & (y_indices >= y_min) & (y_indices <= y_max)
#     inner_boundary = (x_indices >= x_min + avg_diagonal) & (x_indices <= x_max - avg_diagonal) & (y_indices >= y_min + avg_diagonal) & (y_indices <= y_max - avg_diagonal)

#     # Define the hollow area by subtracting the inner boundary from the outer boundary
#     hollow_area = outer_boundary & ~inner_boundary
#     hollow_mask[hollow_area] = 1

#     return hollow_mask


def define_hollow_edge_area(mask, avg_diagonal):
    """
    Defines the hollow edge area of the mask based on the average diagonal length.

    Parameters:
    mask (np.ndarray): A 2D numpy array representing the segmentation mask.
    avg_diagonal (float): Average diagonal length of the bounding boxes.

    Returns:
    np.ndarray: A binary mask representing the hollow edge area.
    """
    # Ensure the mask is binary
    binary_mask = mask > 0

    # Calculate the structuring element size for erosion
    structure_size = int(avg_diagonal)
    structuring_element = np.ones((structure_size, structure_size), dtype=np.uint8)

    # Perform binary erosion to create the inner boundary
    eroded_mask = binary_erosion(binary_mask, structure=structuring_element)

    # Create the hollow area by subtracting the eroded mask from the original mask
    hollow_area = binary_mask & ~eroded_mask

    return hollow_area.astype(np.uint8)


def filter_boxes_at_edges(mask, boxes):
    """
    Filters out the bounding boxes that are at the edges of the mask.

    Parameters:
    mask (np.ndarray): A 2D numpy array representing the segmentation mask.
    boxes (list): List of bounding boxes (each box is a tuple of (x1, y1, x2, y2)).

    Returns:
    list: List of bounding boxes that are not at the edges.
    """
    avg_diagonal = calculate_average_diagonal_length(boxes)
    hollow_edge_area = define_hollow_edge_area(mask, avg_diagonal)

    selected_boxes = []
    unselected_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        if hollow_edge_area[int(box_center_y), int(box_center_x)] == 1:
            selected_boxes.append(box)
        else:
            unselected_boxes.append(box)

    return selected_boxes, unselected_boxes, hollow_edge_area
