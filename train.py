#!/usr/bin/env python3
"""
train.py — Train RF-DETR on a YOLO-format dataset from (near) scratch.

No Roboflow pretrained weights are required.
By default this uses the public DINOv2 backbone from HuggingFace, but you can
also point it at a local Hugging Face vision checkpoint directory (for example
your frozen DINOv3 export with config.json + safetensors).

Quick start:
    python train.py \
        --dataset  /path/to/my_dataset \
        --epochs   100 \
        --batch    4 \
        --output   runs/my_run

See --help for all options.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

RFDETR_SRC = Path(__file__).resolve().parent.parent / "rf-detr" / "src"
if RFDETR_SRC.exists() and str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

from dataset import AUGMENTATION_PRESETS, build_loaders
from model import build_rfdetr, count_parameters, inspect_hf_backbone


# ─────────────────────────────── Arg parsing ─────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RF-DETR (DINOv2 + DETR) on a YOLO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--dataset",  required=True, help="Root directory of the YOLO dataset")
    p.add_argument("--output",   default="runs/exp",  help="Directory to save checkpoints and logs")
    # Model
    p.add_argument("--preset",   default="base",      choices=["base", "large"],
                   help="Model size. 'base'=DINOv2-Small backbone; 'large'=DINOv2-Base")
    p.add_argument("--backbone-path", default=None,
                   help="Optional local Hugging Face backbone directory containing config.json and safetensors")
    p.add_argument("--backbone-out-indexes", nargs="+", type=int, default=None,
                   help="Optional zero-based transformer block indexes to expose as RF-DETR feature maps (e.g. 5 11 17 23)")
    p.add_argument("--backbone-prefix-tokens", type=int, default=None,
                   help="Optional override for non-patch tokens to strip from the local backbone hidden states")
    p.add_argument("--resolution", type=int, default=560,
                   help="Input image resolution (square). Must be divisible by 56 for base/large")
    p.add_argument("--multi-scale", action="store_true", default=True,
                   help="Enable RF-DETR-style random multi-scale resizing during training")
    p.add_argument("--no-multi-scale", action="store_false", dest="multi_scale",
                   help="Disable random multi-scale resizing during training")
    p.add_argument("--expanded-scales", action="store_true", default=True,
                   help="Use the wider RF-DETR scale range for multi-scale training")
    p.add_argument("--no-expanded-scales", action="store_false", dest="expanded_scales",
                   help="Use the narrower RF-DETR scale range for multi-scale training")
    p.add_argument("--aug-preset", default="aerial", choices=sorted(AUGMENTATION_PRESETS.keys()),
                   help="Training augmentation preset. Use 'aerial' for satellite / overhead imagery")
    p.add_argument("--no-dinov2-weights", action="store_true",
                   help="Skip HuggingFace DINOv2 backbone weights (use random init). Ignored when --backbone-path is set.")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze backbone encoder weights during training")
    # Training
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=4,    dest="batch_size")
    p.add_argument("--workers",  type=int,   default=4,    dest="num_workers")
    p.add_argument("--lr",       type=float, default=1e-4, help="Base learning rate")
    p.add_argument("--lr-backbone-scale", type=float, default=0.1,
                   help="Multiplier applied to LR for the DINOv2 encoder (e.g. 0.1 → 1e-5)")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int,   default=5,
                   help="Linear LR warmup for this many epochs before cosine decay")
    p.add_argument("--clip-grad", type=float, default=0.1,
                   help="Max gradient norm (0 = disabled)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for Python, NumPy, PyTorch, and dataloader workers")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision (float16)")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--ema-decay", type=float, default=0.993,
                   help="EMA decay. Set <= 0 to disable EMA")
    p.add_argument("--ema-tau", type=int, default=100,
                   help="EMA warmup time constant in optimizer steps")
    p.add_argument("--lr-vit-layer-decay", type=float, default=0.8,
                   help="Layer-wise LR decay factor across backbone transformer blocks")
    p.add_argument("--lr-component-decay", type=float, default=0.7,
                   help="Component LR decay applied to decoder/backbone groups")
    # Checkpointing
    p.add_argument("--resume",   default=None, help="Path to a checkpoint .pth to resume from")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save a periodic checkpoint every N epochs (in addition to best)")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch for more reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEMA:
    """Exponential moving average with RF-DETR-style tau warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.993, tau: int = 100):
        self.decay = decay
        self.tau = tau
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    def _effective_decay(self) -> float:
        updates = self.updates + 1
        if self.tau > 0:
            return self.decay * (1.0 - math.exp(-updates / self.tau))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._effective_decay()

        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
                continue
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "tau": self.tau,
            "updates": self.updates,
            "model": self.module.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.tau = state_dict.get("tau", self.tau)
        self.updates = state_dict.get("updates", 0)
        self.module.load_state_dict(state_dict["model"])


def compute_multi_scale_scales(
    resolution: int,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> list[int]:
    """Match RF-DETR multi-scale size generation for square inputs."""
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    return [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]


# ─────────────────────────────── LR schedule ─────────────────────────────────── #

def get_lr_scale(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup then cosine decay, returning a multiplicative factor [0, 1]."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(warmup_epochs, 1)
    t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def set_lr(optimizer: AdamW, epoch: int, base_lrs: list[float],
           warmup_epochs: int, total_epochs: int) -> float:
    scale = get_lr_scale(epoch, warmup_epochs, total_epochs)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * scale
    return optimizer.param_groups[-1]["lr"]  # return "main" lr for logging


def _extract_layer_index(name: str) -> int | None:
    markers = (
        ".blocks.",
        ".layer.",
        ".layers.",
        ".encoder.layer.",
        ".transformer.layer.",
        ".h.",
    )
    for marker in markers:
        if marker not in name:
            continue
        tail = name.split(marker, 1)[1]
        piece = tail.split(".", 1)[0]
        if piece.isdigit():
            return int(piece)
    return None


def _get_backbone_lr_decay(name: str, lr_decay_rate: float, num_layers: int) -> float:
    layer_id = num_layers + 1
    if any(token in name for token in ("embeddings", "patch_embed", "pos_embed")):
        layer_id = 0
    else:
        layer_index = _extract_layer_index(name)
        if layer_index is not None:
            layer_id = layer_index + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _get_backbone_weight_decay(name: str, base_weight_decay: float) -> float:
    if any(token in name.lower() for token in ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")):
        return 0.0
    return base_weight_decay


def build_optimizer_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    lr_backbone_scale: float,
    lr_vit_layer_decay: float,
    lr_component_decay: float,
    num_encoder_layers: int,
) -> tuple[list[dict], list[float]]:
    """Approximate RF-DETR optimizer grouping with layer-wise backbone decay."""
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    order: list[tuple[float, float]] = []
    encoder_lr = lr * lr_backbone_scale

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        group_lr = lr
        group_wd = weight_decay
        if "backbone.0.encoder" in name:
            group_lr = (
                encoder_lr
                * _get_backbone_lr_decay(name, lr_vit_layer_decay, num_encoder_layers)
                * (lr_component_decay**2)
            )
            group_wd = _get_backbone_weight_decay(name, weight_decay)
        elif "transformer.decoder" in name:
            group_lr = lr * lr_component_decay

        key = (group_lr, group_wd)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(param)

    param_groups = [{"params": grouped[(group_lr, group_wd)], "lr": group_lr, "weight_decay": group_wd}
                    for group_lr, group_wd in order]
    base_lrs = [group["lr"] for group in param_groups]
    return param_groups, base_lrs


# ─────────────────────────────── Device helpers ──────────────────────────────── #

def to_device(targets: list[dict], device: torch.device) -> list[dict]:
    """Move all tensors in a list of target dicts to the given device."""
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]


# ─────────────────────────────── Train / val loops ───────────────────────────── #

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    scaler, use_amp, clip_grad, ema=None, multi_scales=None, print_freq=50):
    model.train()
    criterion.train()

    running_loss = 0.0
    num_batches = len(data_loader)
    t0 = time.time()

    for i, (images, targets) in enumerate(data_loader):
        if multi_scales:
            scale = random.choice(multi_scales)
            if images.shape[-1] != scale or images.shape[-2] != scale:
                images = F.interpolate(images, size=(scale, scale), mode="bilinear", align_corners=False)
        images  = images.to(device)
        targets = to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs    = model(images)
            loss_dict  = criterion(outputs, targets)
            # Weighted sum across all loss terms (incl. aux outputs)
            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict
                if k in criterion.weight_dict
            )

        scaler.scale(losses).backward()

        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        running_loss += losses.item()

        if (i + 1) % print_freq == 0 or (i + 1) == num_batches:
            avg = running_loss / (i + 1)
            lr  = optimizer.param_groups[-1]["lr"]
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}  [{i+1:5d}/{num_batches}]  "
                  f"loss={avg:.4f}  lr={lr:.2e}  {elapsed:.1f}s")

    return running_loss / num_batches


@torch.no_grad()
def validate(model, criterion, postprocessors, data_loader, device, use_amp):
    model.eval()
    criterion.eval()

    total_loss = 0.0
    coco_evaluator_cls = None
    try:
        coco_evaluator_cls = importlib.import_module("rfdetr.evaluation.coco_eval").CocoEvaluator
    except ImportError:
        coco_evaluator_cls = None

    coco_evaluator = None
    if coco_evaluator_cls is not None and hasattr(data_loader.dataset, "coco"):
        coco_evaluator = coco_evaluator_cls(data_loader.dataset.coco, ["bbox"])

    for images, targets in data_loader:
        images  = images.to(device)
        targets = to_device(targets, device)

        with autocast(enabled=use_amp):
            outputs   = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict
                if k in criterion.weight_dict
            )
        total_loss += losses.item()

        if coco_evaluator is not None:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets])
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            predictions = {int(target["image_id"].item()): result for target, result in zip(targets, results)}
            coco_evaluator.update(predictions)

    metrics = None
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
        metrics = {
            "map_50_95": stats[0],
            "map_50": stats[1],
            "map_75": stats[2],
            "mar_100": stats[8],
        }

    return total_loss / max(len(data_loader), 1), metrics


# ─────────────────────────────── Checkpoint I/O ──────────────────────────────── #

def save_checkpoint(path, model, optimizer, scaler, epoch, val_loss, meta=None, ema=None):
    ckpt = {
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "meta":      meta or {},
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ─────────────────────────────────── Main ────────────────────────────────────── #

def main():
    args = parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ── #
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    backbone_path = str(Path(args.backbone_path).expanduser().resolve()) if args.backbone_path else None
    backbone_meta = inspect_hf_backbone(backbone_path) if backbone_path else None
    backbone_patch_size = backbone_meta["patch_size"] if backbone_meta else 14
    backbone_num_windows = 1 if backbone_meta else 4
    backbone_out_indexes = args.backbone_out_indexes
    train_aug_config = AUGMENTATION_PRESETS[args.aug_preset]
    multi_scales = None
    if args.multi_scale:
        multi_scales = compute_multi_scale_scales(
            args.resolution,
            expanded_scales=args.expanded_scales,
            patch_size=backbone_patch_size,
            num_windows=backbone_num_windows,
        )

    if backbone_meta:
        inferred_indexes = backbone_out_indexes or "auto"
        print("\nUsing local backbone:")
        print(f"  path         : {backbone_meta['path']}")
        print(f"  model_type   : {backbone_meta['model_type']}")
        print(f"  hidden_size  : {backbone_meta['hidden_size']}")
        print(f"  layers       : {backbone_meta['num_hidden_layers']}")
        print(f"  patch_size   : {backbone_patch_size}")
        print(f"  prefix_tokens: {args.backbone_prefix_tokens if args.backbone_prefix_tokens is not None else backbone_meta['num_prefix_tokens']}")
        print(f"  feature_idxs : {inferred_indexes}")

    # ── Data ── #
    print(f"\nLoading dataset from: {args.dataset}")
    print(f"Using augmentation preset: {args.aug_preset}")
    train_loader, val_loader, class_names = build_loaders(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        pin_memory=device.type == "cuda",
        patch_size=backbone_patch_size,
        num_windows=backbone_num_windows,
        train_aug_config=train_aug_config,
        seed=args.seed,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train  : {len(train_loader.dataset):,} images")
    print(f"Val    : {len(val_loader.dataset):,}  images")
    if multi_scales:
        print(f"Multi-scale train sizes: {multi_scales}")

    # ── Model ── #
    print(f"\nBuilding RF-DETR ({args.preset}) ...")
    load_dinov2 = not args.no_dinov2_weights
    if backbone_meta:
        print("  → Loading local backbone weights from disk")
    elif not load_dinov2:
        print("  → Random initialisation (no DINOv2 backbone weights)")
    else:
        print("  → Loading DINOv2 backbone weights from HuggingFace")

    model, criterion, postprocessors = build_rfdetr(
        num_classes=num_classes,
        preset=args.preset,
        load_dinov2_weights=load_dinov2,
        device=str(device),
        freeze_encoder=args.freeze_encoder,
        backbone_path=backbone_path,
        backbone_out_feature_indexes=backbone_out_indexes,
        backbone_num_prefix_tokens=args.backbone_prefix_tokens,
    )
    model.to(device)
    criterion.to(device)
    ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau) if args.ema_decay > 0 else None
    if ema is not None:
        ema.module.to(device)

    stats = count_parameters(model)
    print(f"  Parameters: {stats['total']:,} total / {stats['trainable']:,} trainable")

    num_encoder_layers = (
        max(backbone_out_indexes) + 1
        if backbone_out_indexes
        else (backbone_meta["num_hidden_layers"] if backbone_meta else 12)
    )
    param_groups, base_lrs = build_optimizer_param_groups(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_backbone_scale=args.lr_backbone_scale,
        lr_vit_layer_decay=args.lr_vit_layer_decay,
        lr_component_decay=args.lr_component_decay,
        num_encoder_layers=num_encoder_layers,
    )
    if not param_groups:
        raise ValueError("No trainable parameters found. Did you freeze the entire model?")

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # ── Mixed precision scaler (no-op on CPU/MPS) ── #
    use_amp = args.amp and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    # ── Resume ── #
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scaler, ema=ema)
        start_epoch += 1
        print(f"  Resuming at epoch {start_epoch}, best val loss was {best_val_loss:.4f}")

    # ── Training loop ── #
    print(f"\nTraining for {args.epochs} epochs ...\n")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_map_50_95": [],
        "val_map_50": [],
        "val_map_75": [],
        "ema_val_loss": [],
        "ema_val_map_50_95": [],
        "ema_val_map_50": [],
        "ema_val_map_75": [],
        "lr": [],
    }

    for epoch in range(start_epoch, args.epochs):
        current_lr = set_lr(optimizer, epoch, base_lrs, args.warmup_epochs, args.epochs)

        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, scaler, use_amp, args.clip_grad, ema=ema, multi_scales=multi_scales,
        )

        val_loss, val_metrics = validate(model, criterion, postprocessors, val_loader, device, use_amp)
        ema_val_loss, ema_val_metrics = (None, None)
        if ema is not None:
            ema_val_loss, ema_val_metrics = validate(ema.module, criterion, postprocessors, val_loader, device, use_amp)

        # Logging
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map_50_95"].append(None if val_metrics is None else val_metrics["map_50_95"])
        history["val_map_50"].append(None if val_metrics is None else val_metrics["map_50"])
        history["val_map_75"].append(None if val_metrics is None else val_metrics["map_75"])
        history["ema_val_loss"].append(ema_val_loss)
        history["ema_val_map_50_95"].append(None if ema_val_metrics is None else ema_val_metrics["map_50_95"])
        history["ema_val_map_50"].append(None if ema_val_metrics is None else ema_val_metrics["map_50"])
        history["ema_val_map_75"].append(None if ema_val_metrics is None else ema_val_metrics["map_75"])
        history["lr"].append(current_lr)

        if val_metrics is None:
            print(f"epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")
        else:
            print(
                f"epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"mAP50-95={val_metrics['map_50_95']:.4f}  mAP50={val_metrics['map_50']:.4f}  lr={current_lr:.2e}"
            )
        if ema_val_metrics is not None:
            print(
                f"           ema_val={ema_val_loss:.4f}  "
                f"ema_mAP50-95={ema_val_metrics['map_50_95']:.4f}  ema_mAP50={ema_val_metrics['map_50']:.4f}"
            )
        elif ema_val_loss is not None:
            print(f"           ema_val={ema_val_loss:.4f}")

        # ── Checkpointing ── #
        meta = {"class_names": class_names, "num_classes": num_classes,
            "preset": args.preset, "resolution": args.resolution,
            "aug_preset": args.aug_preset,
            "backbone_path": backbone_path,
            "backbone_out_indexes": backbone_out_indexes,
            "backbone_prefix_tokens": args.backbone_prefix_tokens,
            "seed": args.seed,
            "ema_decay": args.ema_decay,
            "ema_tau": args.ema_tau,
            "lr_vit_layer_decay": args.lr_vit_layer_decay,
            "lr_component_decay": args.lr_component_decay}

        selection_val_loss = ema_val_loss if ema_val_loss is not None else val_loss

        # Always save latest
        save_checkpoint(out_dir / "last.pth", model, optimizer, scaler, epoch, selection_val_loss, meta, ema=ema)

        # Save best
        if selection_val_loss < best_val_loss:
            best_val_loss = selection_val_loss
            save_checkpoint(out_dir / "best.pth", model, optimizer, scaler, epoch, selection_val_loss, meta, ema=ema)
            print(f"  ✓ New best  val={best_val_loss:.4f}  → {out_dir / 'best.pth'}")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            path = out_dir / f"epoch_{epoch:04d}.pth"
            save_checkpoint(path, model, optimizer, scaler, epoch, selection_val_loss, meta, ema=ema)

        # Save history
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
