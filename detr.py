import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import _make_scratch, FeatureFusionBlock, SimpleFPN
from .decoder import DETRDecoder


class DETRHead(nn.Module):
    """
    Detection head built on top of DINOv2 ViT-L intermediate features.

    Pipeline:
        ViT intermediate layers [6, 12, 18, 24]
            → per-scale channel projection (projects)
            → per-scale spatial resize (resize_layers)
            → scratch layer norm convs (layer_rn)
            → top-down FPN fusion via FeatureFusionBlocks (refinenets)
            → flatten path_1 → DETR decoder memory
            → transformer decoder (N layers)
            → class logits + box predictions

    Design decisions:
        - Reuses DPT-style projection + refinenet fusion from segmentation head.
          For detection we stop fusion at path_1 spatial resolution rather than
          upsampling to pixel resolution. FeatureFusionBlock `size` arg is used
          to hold resolution steady instead of doubling.
          Ref: DPT (Vision Transformers for Dense Prediction, Ranftl et al.)

        - Single scale memory (path_1) passed to decoder rather than multi-scale
          concat. DINOv2 self-attention + refinenet fusion means the memory is
          already rich — no encoder (AIFI/CCFM) needed before the decoder.
          Ref: RF-DETR (Roboflow) — single fused scale as decoder memory.

        - No background class in classifier. Focal loss handles background
          implicitly via down-weighting of easy negatives.
          Ref: RT-DETR (DETRs Beat YOLOs on Real-time Object Detection, Zhao et al.)
    """

    def __init__(
        self,
        in_channels=1024,           # DINOv2 ViT-L hidden dim
        features=256,               # common FPN feature dim
        out_channels=[256, 512, 1024, 1024],  # per-scale projection dims
        num_classes=2,
        num_queries=300,
        num_decoder_layers=6,
        ffn_dim=1024,
        num_heads=8,
        dropout=0.0,
        use_bn=False,
    ):
        super().__init__()

        # ── Per-scale channel projection ───────────────────────────────────────
        # Projects each ViT intermediate layer from in_channels → out_channels[i]
        # Ref: DPT reassemble blocks
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1)
            for out_ch in out_channels
        ])

        # ── Per-scale spatial resize ───────────────────────────────────────────
        # Layer 0 (earliest) gets 4x upsample — coarsest spatial resolution
        # Layer 3 (latest)   gets 2x downsample — finest semantic features
        # Ref: DPT resize layers
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # ── Scratch: per-scale norm convs before FPN fusion ────────────────────
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)

        # ── DETR Decoder ───────────────────────────────────────────────────────
        self.decoder = DETRDecoder(
            num_queries=num_queries,
            dim=features,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            num_classes=num_classes,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, out_features, patch_h, patch_w):
        """
        Args:
            out_features: list of 4 tuples (tokens, cls_token) from
                          DINOv2 get_intermediate_layers with return_class_token=True
                          tokens shape: (B, patch_h*patch_w, in_channels)
            patch_h: int — number of patches along height (H // 14 for ViT-L)
            patch_w: int — number of patches along width  (W // 14 for ViT-L)
        Returns:
            logits:      (B, num_queries, num_classes)
            boxes:       (B, num_queries, 4) normalized [cx, cy, w, h]
            aux_outputs: list of dicts with logits + boxes per intermediate
                         decoder layer — used for auxiliary losses in training
        """
        out = []
        for i, x in enumerate(out_features):
            x = x[0]  # (B, N, C) — drop cls token, use patch tokens only

            # Reshape sequence → spatial feature map
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

            # Project channels and resize spatially
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # ── Scratch norm convs ─────────────────────────────────────────────────
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # ── Top-down FPN fusion via refinenets ─────────────────────────────────
        # size= is passed explicitly to hold spatial resolution steady.
        # In the segmentation head this path upsamples to pixel resolution —
        # for detection we stop at layer_1 spatial scale and feed to decoder.
        # Ref: DPT fusion adapted for detection (no pixel-level upsample)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size=layer_1_rn.shape[2:])

        # ── Flatten fused features → decoder memory ────────────────────────────
        # Single scale memory following RF-DETR design — richer than RT-DETR
        # multi-scale concat because DINOv2 + FPN fusion already covers context.
        # Ref: RF-DETR (Roboflow)
        B, C, H, W = path_1.shape
        memory = path_1.flatten(2).permute(2, 0, 1)    # (HW, B, features)

        logits, boxes, aux_outputs = self.decoder(memory)

        return logits, boxes, aux_outputs