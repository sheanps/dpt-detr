import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ── Box Utilities ──────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes):
    """
    Convert [cx, cy, w, h] → [x1, y1, x2, y2].
    YOLO format is already [cx, cy, w, h] normalized so no scaling needed.
    """
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU between all pairs of boxes.

    Args:
        boxes1: (N, 4) xyxy
        boxes2: (M, 4) xyxy
    Returns:
        (N, M) GIoU matrix

    Ref: Generalized Intersection over Union (Rezatofighi et al.)
    Used in both matcher cost and regression loss following DETR paper.
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)

    return giou


# ── Hungarian Matcher ──────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Solves the bipartite matching between predicted queries and ground truth
    objects using the Hungarian algorithm.

    Cost is a weighted sum of:
      - Classification cost (softmax probability of correct class)
      - L1 box cost
      - GIoU box cost

    Note: matching uses softmax-based class cost (not focal) for stability
    during assignment. Focal loss is only applied in the actual training loss.

    Ref: DETR (End-to-End Object Detection with Transformers, Carion et al.)
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs:
                logits: (B, Q, num_classes)
                boxes:  (B, Q, 4) normalized [cx, cy, w, h]
            targets: list of dicts, each with:
                labels: (N,)
                boxes:  (N, 4) normalized [cx, cy, w, h] — YOLO format, no conversion needed
        Returns:
            list of (pred_indices, tgt_indices) tuples per batch item
        """
        B, Q, _ = outputs["logits"].shape

        # Flatten batch dim for cost computation
        pred_logits = outputs["logits"].flatten(0, 1)   # (B*Q, num_classes)
        pred_boxes = outputs["boxes"].flatten(0, 1)     # (B*Q, 4)

        tgt_labels = torch.cat([t["labels"] for t in targets])  # (sum_N,)
        tgt_boxes = torch.cat([t["boxes"] for t in targets])    # (sum_N, 4)

        # Classification cost — softmax probability of the correct class
        pred_probs = pred_logits.softmax(-1)                     # (B*Q, num_classes)
        cost_class = -pred_probs[:, tgt_labels]                  # (B*Q, sum_N)

        # L1 box cost — YOLO format [cx, cy, w, h] matches decoder output directly
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)      # (B*Q, sum_N)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )                                                         # (B*Q, sum_N)

        C = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )
        C = C.view(B, Q, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


# ── Loss ───────────────────────────────────────────────────────────────────────

class DETRLoss(nn.Module):
    """
    DETR loss combining:
      - Focal loss for classification (no explicit background class needed)
      - L1 loss for box regression
      - GIoU loss for box regression

    Auxiliary losses are applied to each intermediate decoder layer output
    to accelerate convergence — especially important with sparse datasets.

    Classification uses sigmoid focal loss rather than softmax cross-entropy:
      - No num_classes + 1 needed (background is implicit)
      - Down-weights easy negatives automatically via (1 - p_t)^gamma
      - alpha=0.5 used instead of 0.25 (COCO default) since we only have 2 classes

    Ref: Focal Loss (RetinaNet, Lin et al.)
    Ref: DETR loss formulation (Carion et al.)
    Ref: RT-DETR auxiliary loss per decoder layer (Zhao et al.)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_class=1.0,
        weight_bbox=5.0,
        weight_giou=2.0,
        focal_alpha=0.5,    # 0.5 for 2-class, 0.25 for COCO-scale imbalance
        focal_gamma=2.0,
        aux_loss_weight=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss_weight = aux_loss_weight

    def loss_labels(self, outputs, targets, indices):
        """
        Sigmoid focal loss over all queries.
        Matched queries get their ground truth class as a one-hot target.
        Unmatched queries get all-zero targets (background, implicitly suppressed).

        Ref: Focal Loss (RetinaNet, Lin et al.)
        """
        logits = outputs["logits"]                          # (B, Q, num_classes)
        B, Q, num_classes = logits.shape

        # One-hot targets — unmatched queries are all zeros (background)
        target_classes = torch.zeros(B, Q, num_classes, device=logits.device)
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            target_classes[i, pred_idx, targets[i]["labels"][tgt_idx]] = 1.0

        # Focal loss
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none")
        p_t = prob * target_classes + (1 - prob) * (1 - target_classes)
        alpha_t = self.focal_alpha * target_classes + (1 - self.focal_alpha) * (1 - target_classes)
        focal = alpha_t * (1 - p_t) ** self.focal_gamma * ce

        num_boxes = max(sum(len(t["boxes"]) for t in targets), 1)
        return focal.sum() / num_boxes

    def loss_boxes(self, outputs, targets, indices):
        """
        L1 + GIoU loss on matched predictions only.
        Box format is [cx, cy, w, h] normalized — matches YOLO label format directly.

        Ref: DETR box loss (Carion et al.)
        """
        pred_boxes = outputs["boxes"]                       # (B, Q, 4)

        pred_matched = torch.cat([
            pred_boxes[i][pred_idx]
            for i, (pred_idx, _) in enumerate(indices)
        ])                                                  # (total_matched, 4)

        tgt_matched = torch.cat([
            targets[i]["boxes"][tgt_idx]
            for i, (_, tgt_idx) in enumerate(indices)
        ])                                                  # (total_matched, 4)

        num_boxes = max(pred_matched.shape[0], 1)

        loss_bbox = F.l1_loss(pred_matched, tgt_matched, reduction="sum") / num_boxes

        loss_giou = (1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(pred_matched),
                box_cxcywh_to_xyxy(tgt_matched),
            )
        )).sum() / num_boxes

        return loss_bbox, loss_giou

    def _compute_losses(self, outputs, targets, indices):
        loss_class = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)

        total = (
            self.weight_class * loss_class +
            self.weight_bbox * loss_bbox +
            self.weight_giou * loss_giou
        )

        return total, loss_class, loss_bbox, loss_giou

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with
                logits:      (B, Q, num_classes)
                boxes:       (B, Q, 4)
                aux_outputs: list of dicts with logits + boxes per intermediate layer
            targets: list of dicts with
                labels: (N,) — 0-indexed class ids
                boxes:  (N, 4) — normalized [cx, cy, w, h]
        Returns:
            dict of losses
        """
        # Match final layer predictions to targets
        indices = self.matcher(outputs, targets)

        total, loss_class, loss_bbox, loss_giou = self._compute_losses(
            outputs, targets, indices
        )

        loss_dict = {
            "loss": total,
            "loss_class": loss_class.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }

        # Auxiliary losses on intermediate decoder layers
        # Ref: RT-DETR — per-layer supervision for faster convergence
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux, targets)
            aux_total, aux_class, aux_bbox, aux_giou = self._compute_losses(
                aux, targets, aux_indices
            )
            total = total + self.aux_loss_weight * aux_total

            loss_dict[f"aux_{i}_loss_class"] = aux_class.detach()
            loss_dict[f"aux_{i}_loss_bbox"] = aux_bbox.detach()
            loss_dict[f"aux_{i}_loss_giou"] = aux_giou.detach()

        # Update total loss to include auxiliary contributions
        loss_dict["loss"] = total

        return loss_dict