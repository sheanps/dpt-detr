import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class YOLODetectionDataset(Dataset):
    """
    Dataset loader for YOLO-format object detection labels.

    YOLO label format per line: <class_id> <cx> <cy> <w> <h>
    All values are normalized [0, 1] relative to image dimensions.

    This format maps directly to the decoder's output format [cx, cy, w, h]
    so no coordinate conversion is needed at any point in the pipeline.

    Directory structure expected:
        img_dir/
            image1.jpg
            image2.jpg
        label_dir/
            image1.txt
            image2.txt

    Images without a corresponding label file are silently skipped.
    """

    def __init__(
        self,
        img_dir,
        label_dir,
        image_size=(476, 630),  # (H, W) — matches DINOv2 ViT-L patch grid
        augment=False,
    ):
        """
        Args:
            img_dir:    path to directory of images
            label_dir:  path to directory of YOLO .txt label files
            image_size: (H, W) to resize all images to
            augment:    whether to apply training augmentations
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.augment = augment

        self.samples = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(label_dir, stem + '.txt')
            if os.path.exists(label_path):
                self.samples.append((
                    os.path.join(img_dir, fname),
                    label_path,
                ))

        # ImageNet normalization — required for DINOv2 backbone
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, label_path):
        """
        Parse a YOLO .txt label file.

        Returns:
            boxes:  (N, 4) float32 tensor [cx, cy, w, h] normalized
            labels: (N,)   int64  tensor of class indices (0-indexed)
        """
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                labels.append(cls)
                boxes.append([cx, cy, w, h])

        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def _augment(self, image, boxes):
        """
        Simple training augmentations that are safe with normalized box coords.

        Only horizontal flip is applied — avoids any augmentation that would
        require re-normalizing box coordinates (e.g. random crop, rotation).
        ColorJitter is applied to the image only.

        Args:
            image: PIL Image
            boxes: (N, 4) float32 [cx, cy, w, h] normalized
        Returns:
            image: PIL Image
            boxes: (N, 4) float32
        """
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            if boxes.shape[0] > 0:
                # cx flips to 1 - cx, w unchanged
                boxes[:, 0] = 1.0 - boxes[:, 0]

        # Color jitter — image only, no box changes needed
        color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        image = color_jitter(image)

        return image, boxes

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        image = image.resize(
            (self.image_size[1], self.image_size[0]),  # PIL takes (W, H)
            Image.BILINEAR,
        )

        boxes, labels = self._load_labels(label_path)

        if self.augment:
            image, boxes = self._augment(image, boxes)

        # To tensor + normalize for DINOv2
        image = TF.to_tensor(image)
        image = self.normalize(image)

        return image, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    """
    Custom collate for variable-length target lists.

    Images are stacked into a single tensor.
    Targets stay as a list of dicts since each image has a different
    number of ground truth boxes — cannot be naively batched.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)       # (B, 3, H, W)
    return images, list(targets)       # targets: list of B dicts