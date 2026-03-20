import os
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import YOLODetectionDataset, collate_fn
from .loss import DETRLoss, HungarianMatcher
from .model import DINOv3DETR


# ── Utilities ──────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running mean of a scalar value across steps."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def log(epoch, step, total_steps, losses, prefix="train"):
    parts = [f"[{prefix}] epoch {epoch:03d}  step {step:04d}/{total_steps:04d}"]
    for k, v in losses.items():
        parts.append(f"{k}: {v:.4f}")
    print("  ".join(parts))


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Run one full pass over the validation set.
    Returns dict of average losses.
    """
    model.eval()
    meters = defaultdict(AverageMeter)

    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        logits, boxes, aux_outputs = model(images)
        outputs = {"logits": logits, "boxes": boxes, "aux_outputs": aux_outputs}

        loss_dict = criterion(outputs, targets)

        B = images.shape[0]
        for k, v in loss_dict.items():
            meters[k].update(v.item(), B)

    return {k: m.avg for k, m in meters.items()}


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = YOLODetectionDataset(
        img_dir=os.path.join(args.data_dir, "train", "images"),
        label_dir=os.path.join(args.data_dir, "train", "labels"),
        image_size=(args.image_height, args.image_width),
        augment=True,
    )
    val_dataset = YOLODetectionDataset(
        img_dir=os.path.join(args.data_dir, "val", "images"),
        label_dir=os.path.join(args.data_dir, "val", "labels"),
        image_size=(args.image_height, args.image_width),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = DINOv3DETR(
        num_classes=args.num_classes,
        image_height=args.image_height,
        image_width=args.image_width,
        features=args.features,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        backbone_weights_path=args.backbone_weights,
        head_weights_path=args.head_weights,
        device=str(device),
    )

    # Only head parameters are trained — backbone is frozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
    criterion = DETRLoss(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_class=args.weight_class,
        weight_bbox=args.weight_bbox,
        weight_giou=args.weight_giou,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        aux_loss_weight=args.aux_loss_weight,
    ).to(device)

    # ── Optimizer + Scheduler ──────────────────────────────────────────────────
    # AdamW with weight decay on non-bias/norm params only
    # Ref: DETR training recipe (Carion et al.)
    param_groups = [
        {
            "params": [
                p for n, p in model.head.named_parameters()
                if p.requires_grad and "bias" not in n
                and not isinstance(model.head.get_submodule(n.rsplit(".", 1)[0])
                                   if "." in n else model.head, nn.LayerNorm)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.head.named_parameters()
                if p.requires_grad and ("bias" in n)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # Linear warmup then cosine decay
    # Ref: RT-DETR training schedule (Zhao et al.)
    def lr_lambda(step):
        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Checkpoint Resume ──────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_last.pth")

    if args.resume and os.path.isfile(checkpoint_path):
        print(f"Resuming from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}")

    # ── Train ──────────────────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Keep backbone in eval mode — it is frozen and has BatchNorm layers
        model.backbone.eval()

        meters = defaultdict(AverageMeter)

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            logits, boxes, aux_outputs = model(images)
            outputs = {"logits": logits, "boxes": boxes, "aux_outputs": aux_outputs}

            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — important for transformer decoder stability
            # Ref: DETR training (Carion et al.)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=args.clip_grad)

            optimizer.step()
            scheduler.step()
            global_step += 1

            B = images.shape[0]
            for k, v in loss_dict.items():
                meters[k].update(v.item() if isinstance(v, torch.Tensor) else v, B)

            if step % args.log_every == 0:
                log(epoch, step, len(train_loader),
                    {k: m.avg for k, m in meters.items()}, prefix="train")

        # ── Validation ─────────────────────────────────────────────────────────
        val_losses = validate(model, val_loader, criterion, device)
        log(epoch, len(train_loader), len(train_loader), val_losses, prefix="val")

        val_loss = val_losses["loss"]

        # ── Checkpointing ──────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "head": model.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
            torch.save(ckpt, best_path)
            print(f"[ckpt] New best val loss {best_val_loss:.4f} — saved to {best_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 DETR detector")

    # Data
    parser.add_argument("--data_dir",       type=str,   required=True,  help="Root dir with train/val subdirs in YOLO format")
    parser.add_argument("--image_height",   type=int,   default=476)
    parser.add_argument("--image_width",    type=int,   default=630)
    parser.add_argument("--num_classes",    type=int,   default=2)

    # Model
    parser.add_argument("--features",           type=int,   default=256)
    parser.add_argument("--num_queries",         type=int,   default=300)
    parser.add_argument("--num_decoder_layers",  type=int,   default=6)
    parser.add_argument("--backbone_weights",    type=str,   default=None,   help="Path to DINOv3 ViT-L weights")
    parser.add_argument("--head_weights",        type=str,   default=None,   help="Path to DETRHead weights (optional resume)")

    # Loss
    parser.add_argument("--cost_class",      type=float, default=1.0)
    parser.add_argument("--cost_bbox",       type=float, default=5.0)
    parser.add_argument("--cost_giou",       type=float, default=2.0)
    parser.add_argument("--weight_class",    type=float, default=1.0)
    parser.add_argument("--weight_bbox",     type=float, default=5.0)
    parser.add_argument("--weight_giou",     type=float, default=2.0)
    parser.add_argument("--focal_alpha",     type=float, default=0.5)
    parser.add_argument("--focal_gamma",     type=float, default=2.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.5)

    # Training
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--warmup_epochs", type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--clip_grad",     type=float, default=0.1)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--log_every",     type=int,   default=10)
    parser.add_argument("--device",        type=str,   default="cuda")

    # Output
    parser.add_argument("--output_dir",    type=str,   default="checkpoints")
    parser.add_argument("--resume",        action="store_true", help="Resume from checkpoint_last.pth")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)