import numpy as np


def calculate_overlap_area_bb(box1, box2):
    """Calculates the overlapping area between two boxes."""
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))

    if x_overlap > 0 and y_overlap > 0:
        return [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    else:
        return None  # No overlap



def is_overlap_bm(box, mask):
    """
    Check whether the box overlaps with the mask.

    Parameters:
    box (tuple): A tuple of four coordinates representing the rectangle (x1, y1, x2, y2).
    mask (np.ndarray): A 2D numpy array representing the segmentation mask.

    Returns:
    bool: True if the box overlaps with the mask, False otherwise.
    """
    x1, y1, x2, y2 = map(int, box)

    # Ensure the coordinates are within the bounds of the mask
    x1 = max(0, min(x1, mask.shape[1]))
    x2 = max(0, min(x2, mask.shape[1]))
    y1 = max(0, min(y1, mask.shape[0]))
    y2 = max(0, min(y2, mask.shape[0]))

    # Check if any pixel in the defined rectangle is part of the segmentation mask
    for x in range(x1, x2):
        for y in range(y1, y2):
            if mask[y, x] == 1:
                return True
    return False

def calculate_overlap_area_bm(box, mask):
    """
    Calculates the overlapping area between a box and a segmentation mask.

    Parameters:
    box (tuple): A tuple of four coordinates representing the rectangle (x1, y1, x2, y2).
    mask (np.ndarray): A 2D numpy array representing the segmentation mask.

    Returns:
    tuple: A tuple containing the list of overlapping pixel coordinates and the overlap mask.
    """
    x1, y1, x2, y2 = map(int, box)

    # Ensure the coordinates are within the bounds of the mask
    x1 = max(0, min(x1, mask.shape[1]))
    x2 = max(0, min(x2, mask.shape[1]))
    y1 = max(0, min(y1, mask.shape[0]))
    y2 = max(0, min(y2, mask.shape[0]))

    overlap_coords = []
    overlap_mask = np.zeros_like(mask, dtype=np.uint8)

    # Check if any pixel in the defined rectangle is part of the segmentation mask
    for x in range(x1, x2):
        for y in range(y1, y2):
            if mask[y, x] == 1:
                overlap_coords.append([x, y])
                overlap_mask[y, x] = 2

    return overlap_coords, overlap_mask


def extract_relative_area(overlap_coords, mask_shape, relative_position):
    overlap_array = np.array(overlap_coords)
    x_min, y_min = overlap_array.min(axis=0)
    x_max, y_max = overlap_array.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    relative_mask = np.zeros(mask_shape, dtype=np.uint8)

    if relative_position == 'top_left':
        selected_area = [(x, y) for (x, y) in overlap_coords if x < x_min + width // 2 and y < y_min + height // 2]
    elif relative_position == 'top_right':
        selected_area = [(x, y) for (x, y) in overlap_coords if x >= x_min + width // 2 and y < y_min + height // 2]
    elif relative_position == 'bottom_left':
        selected_area = [(x, y) for (x, y) in overlap_coords if x < x_min + width // 2 and y >= y_min + height // 2]
    else:  # bottom_right
        selected_area = [(x, y) for (x, y) in overlap_coords if x >= x_min + width // 2 and y >= y_min + height // 2]

    for (x, y) in selected_area:
        relative_mask[y, x] = 3

    return selected_area, relative_mask