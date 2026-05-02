"""Generate monocular depth images for a folder of RGB images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import GLPNForDepthEstimation, GLPNImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("examples/GNV_benchmark_data_coco"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/depth/GLPN"),
        help="Directory where depth images will be written.",
    )
    parser.add_argument(
        "--model_name",
        default="vinvino02/glpn-nyu",
        help="Hugging Face model id for GLPN depth estimation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = GLPNImageProcessor.from_pretrained(args.model_name)
    model = GLPNForDepthEstimation.from_pretrained(args.model_name)

    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        depth.save(args.output_dir / f"depth_{image_path.name}")

    print(f"Depth images saved in {args.output_dir}")


if __name__ == "__main__":
    main()
