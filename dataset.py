"""
dataset.py — YOLO format dataset loader for RF-DETR training.

Expected directory layout (standard YOLO / Roboflow export):
    dataset_dir/
        data.yaml          ← class names + split paths
        train/
            images/        ← *.jpg / *.png / *.bmp
            labels/        ← *.txt (YOLO format: class cx cy w h, normalised)
        valid/
            images/
            labels/
        test/              ← optional
            images/
            labels/

Usage:
    train_ds, val_ds, class_names = build_datasets(
        dataset_dir = "/path/to/my_dataset",
        resolution  = 560,
    )
    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn, ...)
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

RFDETR_SRC = Path(__file__).resolve().parent.parent / "rf-detr" / "src"
if RFDETR_SRC.exists() and str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

# RF-DETR dataset utilities
from rfdetr.datasets.aug_config import (
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONFIG,
    AUG_CONSERVATIVE,
    AUG_INDUSTRIAL,
)
from rfdetr.datasets.coco import make_coco_transforms
from rfdetr.datasets.yolo import YoloDetection, is_valid_yolo_dataset

REQUIRED_YAML_FILES = ["data.yaml", "data.yml"]

AUGMENTATION_PRESETS = {
    "default": AUG_CONFIG,
    "conservative": AUG_CONSERVATIVE,
    "aggressive": AUG_AGGRESSIVE,
    "aerial": AUG_AERIAL,
    "industrial": AUG_INDUSTRIAL,
    "none": {},
}


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader workers for reproducible augmentation / sampling."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────── Dataset builder ─────────────────────────────── #

def build_datasets(
    dataset_dir: str,
    resolution: int = 560,
    patch_size: int = 14,
    num_windows: int = 4,
    train_aug_config=None,
) -> Tuple[YoloDetection, YoloDetection, List[str]]:
    """
    Build train and validation datasets from a YOLO-format directory.

    Args:
        dataset_dir: Root of the YOLO dataset (must contain data.yaml and
                     train/ + valid/ subdirectories).
        resolution:  Target square image size in pixels.  Must be divisible by
                     ``patch_size * num_windows`` (default: 56).
        patch_size:  DINOv2 patch size — should match the model (default: 14).
        num_windows: Windowed attention window count (default: 4).
        train_aug_config: Optional RF-DETR/Albumentations config for the train split.

    Returns:
        (train_dataset, val_dataset, class_names)
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not is_valid_yolo_dataset(str(root)):
        raise ValueError(
            f"'{root}' does not look like a valid YOLO dataset.\n"
            "Expected: data.yaml, train/images/, train/labels/, valid/images/, valid/labels/"
        )

    block = patch_size * num_windows
    if resolution % block != 0:
        raise ValueError(
            f"resolution={resolution} must be divisible by patch_size*num_windows={block}. "
            f"Try {round(resolution / block) * block}."
        )

    data_file = next(
        (root / f for f in REQUIRED_YAML_FILES if (root / f).exists()), root / "data.yaml"
    )

    train_ds = YoloDetection(
        img_folder=str(root / "train" / "images"),
        lb_folder=str(root / "train" / "labels"),
        data_file=str(data_file),
        transforms=make_coco_transforms(
            image_set="train",
            resolution=resolution,
            multi_scale=False,           # no random multi-scale, keep it simple
            skip_random_resize=True,     # resize directly to `resolution`
            patch_size=patch_size,
            num_windows=num_windows,
            aug_config=train_aug_config,
        ),
    )

    val_ds = YoloDetection(
        img_folder=str(root / "valid" / "images"),
        lb_folder=str(root / "valid" / "labels"),
        data_file=str(data_file),
        transforms=make_coco_transforms(
            image_set="val",
            resolution=resolution,
            multi_scale=False,
            skip_random_resize=True,
            patch_size=patch_size,
            num_windows=num_windows,
        ),
    )

    class_names: List[str] = train_ds.classes
    return train_ds, val_ds, class_names


# ─────────────────────────────────── Collation ───────────────────────────────── #

def collate_fn(batch):
    """
    Stack images into a single tensor; keep targets as a list of dicts.

    RF-DETR's LWDETR.forward accepts either a plain tensor or a NestedTensor.
    When all images in the batch have the same resolution (which they do after
    the fixed-size resize transform), stacking into a tensor is fine.
    """
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_loaders(
    dataset_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    resolution: int = 560,
    pin_memory: bool = True,
    patch_size: int = 14,
    num_windows: int = 4,
    train_aug_config=None,
    seed: int | None = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Convenience wrapper: builds datasets and loaders in one call.

    Returns:
        (train_loader, val_loader, class_names)
    """
    train_ds, val_ds, class_names = build_datasets(
        dataset_dir,
        resolution=resolution,
        patch_size=patch_size,
        num_windows=num_windows,
        train_aug_config=train_aug_config,
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return train_loader, val_loader, class_names
