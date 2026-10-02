"""Labelme dataset reader shared by M1-M5."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = 255
SUPPORTED_LABELS = {"building", "window", "door", "road", "ground"}


def discover_samples(data_dir: Path, sample_limit: int | None = None) -> list[tuple[Path, Path]]:
    samples = []
    for annotation in sorted(data_dir.glob("*.json")):
        image = next(
            (annotation.with_suffix(ext) for ext in (".png", ".jpg", ".jpeg") if annotation.with_suffix(ext).exists()),
            None,
        )
        if image is not None:
            samples.append((image, annotation))
    return samples[:sample_limit] if sample_limit else samples


def transforms(image_size: int, training: bool) -> A.Compose:
    operations = []
    if training:
        operations.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
                A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            ]
        )
    operations.append(A.Resize(image_size, image_size))
    return A.Compose(operations, seed=42)


def read_polygons(annotation_path: Path) -> tuple[int, int, dict[str, list]]:
    record = json.loads(annotation_path.read_text(encoding="utf-8"))
    polygons = {label: [] for label in SUPPORTED_LABELS}
    for shape in record.get("shapes", []):
        label = str(shape.get("label", "")).lower().strip()
        points = shape.get("points", [])
        if label in polygons and len(points) >= 3:
            polygons[label].append(points)
    return int(record["imageHeight"]), int(record["imageWidth"]), polygons


def fill(mask: np.ndarray, polygons: list, value: int) -> None:
    for points in polygons:
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], value)


def make_mask(height: int, width: int, polygons: dict[str, list], task: str) -> np.ndarray:
    if task == "single_stage":
        mask = np.zeros((height, width), np.uint8)
        fill(mask, polygons["building"], 1)
        fill(mask, polygons["window"], 2)
        fill(mask, polygons["door"], 2)
        return mask

    building = np.zeros((height, width), np.uint8)
    for label in ("building", "window", "door"):
        fill(building, polygons[label], 1)

    if task == "building_binary":
        return building

    if task == "building_hard_negative":
        mask = np.zeros((height, width), np.uint8)
        fill(mask, polygons["road"], 2)
        fill(mask, polygons["ground"], 2)
        mask[building == 1] = 1
        return mask

    if task == "opening":
        mask = np.full((height, width), IGNORE_INDEX, np.uint8)
        mask[building == 1] = 0
        fill(mask, polygons["window"], 1)
        fill(mask, polygons["door"], 1)
        return mask

    raise ValueError(f"Unknown task: {task}")


class FacadeDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], task: str, image_size: int, training: bool):
        self.samples = samples
        self.task = task
        self.transform = transforms(image_size, training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, annotation_path = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, polygons = read_polygons(annotation_path)
        if image.shape[:2] != (height, width):
            raise ValueError(f"Image/annotation size mismatch: {image_path.name}")
        mask = make_mask(height, width, polygons, self.task)
        augmented = self.transform(image=image, mask=mask)
        image_tensor = torch.from_numpy(augmented["image"].transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(augmented["mask"].astype(np.int64))
        return image_tensor, mask_tensor, image_path.name


def wwr_from_mask(mask: np.ndarray) -> float:
    wall = int((mask == 1).sum())
    opening = int((mask == 2).sum())
    denominator = wall + opening
    return opening / denominator if denominator else 0.0

