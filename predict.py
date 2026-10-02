"""Predict masks and WWR from M1/M2/M5 or a paired M3/M4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from wwr_prediction.models import build_model


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    task = checkpoint["task"]
    classes = 2 if task in {"building_binary", "opening"} else 3
    model = build_model(config["architecture"], classes, False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, task


def infer(model, image_rgb: np.ndarray, image_size: int, device) -> np.ndarray:
    resized = cv2.resize(image_rgb, (image_size, image_size))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        prediction = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(prediction, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--building-checkpoint", type=Path)
    parser.add_argument("--opening-checkpoint", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f"Cannot read {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.checkpoint:
        model, config, task = load_checkpoint(args.checkpoint, device)
        if task != "single_stage":
            raise ValueError("Use paired checkpoints for a two-stage model")
        mask = infer(model, image, config["image_size"], device)
    elif args.building_checkpoint and args.opening_checkpoint:
        building_model, building_config, _ = load_checkpoint(args.building_checkpoint, device)
        opening_model, opening_config, _ = load_checkpoint(args.opening_checkpoint, device)
        building = infer(building_model, image, building_config["image_size"], device) == 1
        opening = infer(opening_model, image, opening_config["image_size"], device) == 1
        mask = np.zeros(building.shape, np.uint8)
        mask[building] = 1
        mask[building & opening] = 2
    else:
        raise ValueError("Provide --checkpoint or both paired checkpoints")

    wall = int((mask == 1).sum())
    opening = int((mask == 2).sum())
    wwr = opening / (wall + opening) if wall + opening else 0.0
    print(json.dumps({"image": args.image.name, "wall_pixels": wall, "opening_pixels": opening, "wwr": wwr}, indent=2))


if __name__ == "__main__":
    main()

