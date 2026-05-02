import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# DAMv2 for 3D representations
def get_depth(image, image_processor, model):
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth_map

def load_image_depth(image_path):
    return Image.open(image_path).convert("RGB")

def load_depth_model(model_name):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    return image_processor, model

def get_volume(depth_map, mask):
    mask = mask.astype(bool)
    masked_depth = depth_map * mask
    volume = np.sum(masked_depth)
    return volume

def save_volumes(volumes, original_image_name, saving_dir):
    os.makedirs(saving_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_image_name))[0]
    save_path = os.path.join(saving_dir, f"{base_name}_volumes.txt")
    with open(save_path, 'w') as f:
        for i, volume in enumerate(volumes):
            f.write(f"Volume for Mask {i + 1}: {volume}\n")

def plot_3d_all_masks(depth_map, masks, original_image_name, saving_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    z = depth_map

    colormap = colormaps['hsv']

    for i in range(masks.shape[0]):
        mask = masks[i]
        mask_flat = mask.flatten()
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        color = colormap(i / masks.shape[0])
        ax.scatter(x_flat[mask_flat == 1], y_flat[mask_flat == 1], z_flat[mask_flat == 1], c=[color], label=f'Mask {i + 1}', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.legend()

    base_name = os.path.splitext(os.path.basename(original_image_name))[0]
    save_path = os.path.join(saving_dir, f"{base_name}_all_masks_3d_plot.jpg")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_3d_single_mask(depth_map, mask, mask_index, masks_total, original_image_name, saving_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    z = depth_map

    colormap = colormaps['hsv']
    color = colormap(mask_index / masks_total)

    mask_flat = mask.flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    ax.scatter(x_flat[mask_flat == 1], y_flat[mask_flat == 1], z_flat[mask_flat == 1], c=[color], label=f'Mask {mask_index + 1}', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.legend()

    base_name = os.path.splitext(os.path.basename(original_image_name))[0]
    save_path = os.path.join(saving_dir, f"{base_name}_mask_{mask_index + 1}_3d_plot.jpg")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def get_3d_rst(image_path, masks_path, model_name, saving_dir):
    image = load_image_depth(image_path)
    masks = np.load(masks_path)

    image_processor, model = load_depth_model(model_name)
    depth_map = get_depth(image, image_processor, model)

    # Format the depth map and save the depth image
    formatted = (depth_map * 255 / np.max(depth_map)).astype("uint8")
    depth = Image.fromarray(formatted)

    # Ensure the saving directory exists
    os.makedirs(saving_dir, exist_ok=True)

    # Save the depth image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    depth_image_save_path = os.path.join(saving_dir, f"{base_name}_depth_image.jpg")
    depth.save(depth_image_save_path)

    volumes = [get_volume(depth_map, masks[i]) for i in range(masks.shape[0])]
    save_volumes(volumes, image_path, saving_dir)

    plot_3d_all_masks(depth_map, masks, image_path, saving_dir)

    for i in range(masks.shape[0]):
        plot_3d_single_mask(depth_map, masks[i], i, masks.shape[0], image_path, saving_dir)
