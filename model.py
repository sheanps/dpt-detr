import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from backbone import DINOv3
from detr import DETRHead


class DINOv3DETR(nn.Module):
    """
    Object detection model combining a frozen DINOv3 ViT-L backbone
    with a DPT-style FPN fusion head and DETR transformer decoder.

    Backbone:
        DINOv3 ViT-L (patch size 14) — fully frozen during training.
        Intermediate features extracted from layers [6, 12, 18, 24].
        These are 0-indexed as [5, 11, 17, 23] in get_intermediate_layers.

    Head:
        DETRHead — per-scale projection + FPN fusion (refinenets) + DETR decoder.
        See detr_head.py for full pipeline description.

    Ref: DINOv3 (Oquab et al.) — backbone architecture (DINOv3 follows same API)
    Ref: RF-DETR (Roboflow) — ViT backbone + single scale DETR decoder design
    Ref: DPT (Ranftl et al.) — intermediate layer extraction + FPN fusion
    """

 
    INTERMEDIATE_LAYERS = [5, 11, 17, 23]

    def __init__(
        self,
        num_classes=2,
        image_height=476,
        image_width=630,
        features=256,             
        out_channels=[256, 512, 1024, 1024], 
        num_queries=300,
        num_decoder_layers=6,
        ffn_dim=1024,
        num_heads=8,
        dropout=0.0,
        use_bn=False,
        backbone_weights_path=None, 
        head_weights_path=None,    
        device="cuda",
    ):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.device = device

  
        self.backbone = DINOv3(model_name="vitl")

        if backbone_weights_path and os.path.isfile(backbone_weights_path):
            print(f"[Backbone] Loading weights from: {backbone_weights_path}")
            state_dict = torch.load(backbone_weights_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Backbone] Missing keys:    {missing}")
            if unexpected:
                print(f"[Backbone] Unexpected keys: {unexpected}")
        else:
            print("[Backbone] No weights loaded — backbone randomly initialized.")

        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

   
        self.head = DETRHead(
            in_channels=1024,    
            features=features,
            out_channels=out_channels,
            num_classes=num_classes,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_bn=use_bn,
        )

        if head_weights_path and os.path.isfile(head_weights_path):
            print(f"[Head] Loading weights from: {head_weights_path}")
            state_dict = torch.load(head_weights_path, map_location="cpu")
            missing, unexpected = self.head.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Head] Missing keys:    {missing}")
                # Reinitialize submodules with missing weights
                modules_to_init = set(k.split(".")[0] for k in missing)
                for name, module in self.head.named_children():
                    if name in modules_to_init:
                        print(f"[Head] Reinitializing: head.{name}")
                        self._init_module(module)
            if unexpected:
                print(f"[Head] Unexpected keys: {unexpected}")
        else:
            print("[Head] No weights loaded — head randomly initialized.")
            self._init_module(self.head)

        self.backbone.to(device)
        self.head.to(device)

    def _init_module(self, module):
        """
        Weight initialization per layer type.
        Conv2d: Kaiming normal (suited for ReLU activations)
        Linear: Xavier uniform (suited for attention / FFN layers)
        BatchNorm: constant 1/0
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) normalized image tensor
        Returns:
            logits:      (B, num_queries, num_classes)
            boxes:       (B, num_queries, 4) normalized [cx, cy, w, h]
            aux_outputs: list of dicts with logits + boxes per intermediate
                         decoder layer — consumed by DETRLoss auxiliary losses
        """
        patch_h = x.shape[-2] // 14
        patch_w = x.shape[-1] // 14

        # Extract intermediate ViT-L features — backbone is frozen
        # Layers 6, 12, 18, 24 (1-indexed) → [5, 11, 17, 23] (0-indexed)
        # Each entry is (tokens, cls_token): tokens shape (B, patch_h*patch_w, 1024)
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                x,
                self.INTERMEDIATE_LAYERS,
                return_class_token=True,
            )

        logits, boxes, aux_outputs = self.head(features, patch_h, patch_w)

        return logits, boxes, aux_outputs

    def save_head(self, path):
        """Save detection head weights only — backbone is frozen so no need to save it."""
        torch.save(self.head.state_dict(), path)
        print(f"[Head] Saved to: {path}")

    @torch.no_grad()
    def predict(self, x, confidence_threshold=0.5):
        """
        Inference with confidence thresholding per image in the batch.

        Args:
            x:                    (B, 3, H, W) normalized image tensor
            confidence_threshold: float — minimum score to keep a detection
        Returns:
            list of B dicts, each with:
                scores:  (K,)    confidence scores of kept detections
                labels:  (K,)    predicted class indices
                boxes:   (K, 4)  normalized [cx, cy, w, h]
        """
        self.eval()
        logits, boxes, _ = self.forward(x)

        # Sigmoid scores — one score per class per query (focal loss convention)
        probs = logits.sigmoid()                            # (B, Q, num_classes)
        scores, labels = probs.max(dim=-1)                  # (B, Q)

        results = []
        for i in range(x.shape[0]):
            keep = scores[i] > confidence_threshold
            results.append({
                "scores": scores[i][keep],
                "labels": labels[i][keep],
                "boxes":  boxes[i][keep],
            })

        return results