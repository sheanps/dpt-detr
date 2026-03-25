"""
model.py — Build RF-DETR with either the stock DINOv2 windowed backbone or a
local Hugging Face vision backbone directory.

Architecture: RFDETRBase config
    backbone : DINOv2-Small windowed, DINOv2-Base windowed, or a local HF ViT
    decoder  : 3-layer conditional DETR decoder with deformable cross-attention
    criterion: Focal loss + L1 / GIoU bbox loss, Hungarian matching

Usage:
        model, criterion, postprocessors = build_rfdetr(num_classes=10)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

RFDETR_SRC = Path(__file__).resolve().parent.parent / "rf-detr" / "src"
if RFDETR_SRC.exists() and str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

# ── RF-DETR internal imports (install with: pip install -e /path/to/rf-detr[train]) ──
from rfdetr.models.backbone import Joiner, build_backbone
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.postprocess import PostProcess
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.models.transformer import Transformer
from rfdetr.utilities.tensors import NestedTensor

# ─────────────────────────────────── Presets ─────────────────────────────────── #

# RF-DETR Base — matches RFDETRBaseConfig exactly
BASE_CFG = dict(
    encoder="dinov2_windowed_small",   # ViT-Small/14, no registers, windowed attn
    hidden_dim=256,
    out_feature_indexes=[2, 5, 8, 11], # 4 feature stages from the ViT
    projector_scale=["P4"],            # single-scale P4 → num_feature_levels=1
    patch_size=14,
    num_windows=4,
    positional_encoding_size=37,       # 37 * 14 = 518 (matches DINOv2 default)
    resolution=560,                    # input image size (must be divisible by 56)
    # Decoder
    dec_layers=3,
    sa_nheads=8,
    ca_nheads=16,
    dec_n_points=2,
    num_queries=300,
    group_detr=13,
    two_stage=True,
    bbox_reparam=True,
    lite_refpoint_refine=True,
    # Misc
    layer_norm=True,
    dim_feedforward=2048,
    dropout=0.0,
    vit_encoder_num_layers=12,
)

# RF-DETR Large — swap in DINOv2-Base backbone, wider hidden dim
LARGE_CFG = dict(
    **BASE_CFG,
    encoder="dinov2_windowed_base",    # ViT-Base/14, no registers, windowed attn
    hidden_dim=384,
    sa_nheads=12,
    ca_nheads=24,
    dec_n_points=4,
    vit_encoder_num_layers=12,
)


def _coerce_image_size(value: Any) -> int | tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _infer_num_prefix_tokens(cfg: Any) -> int:
    num_register_tokens = int(
        getattr(cfg, "num_register_tokens", getattr(cfg, "register_token_count", 0)) or 0
    )
    if hasattr(cfg, "num_cls_tokens") and getattr(cfg, "num_cls_tokens") is not None:
        num_cls_tokens = int(getattr(cfg, "num_cls_tokens"))
    else:
        num_cls_tokens = 1 if getattr(cfg, "use_cls_token", True) else 0
    return num_cls_tokens + num_register_tokens


def _infer_evenly_spaced_feature_indexes(num_hidden_layers: int, num_features: int = 4) -> list[int]:
    indexes = sorted({max(0, math.ceil((i + 1) * num_hidden_layers / num_features) - 1) for i in range(num_features)})
    if len(indexes) == num_features:
        return indexes
    return list(range(max(0, num_hidden_layers - num_features), num_hidden_layers))


def inspect_hf_backbone(backbone_path: str) -> dict[str, Any]:
    resolved_path = str(Path(backbone_path).expanduser().resolve())
    cfg = AutoConfig.from_pretrained(resolved_path, local_files_only=True, trust_remote_code=True)

    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "embed_dim", None))
    num_hidden_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "depth", None))
    patch_size = getattr(cfg, "patch_size", None)
    image_size = _coerce_image_size(getattr(cfg, "image_size", None))
    num_prefix_tokens = _infer_num_prefix_tokens(cfg)

    missing = [
        name
        for name, value in {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "patch_size": patch_size,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            f"Backbone config at '{resolved_path}' is missing required field(s): {', '.join(missing)}"
        )

    return {
        "path": resolved_path,
        "config": cfg,
        "model_type": getattr(cfg, "model_type", "unknown"),
        "hidden_size": int(hidden_size),
        "num_hidden_layers": int(num_hidden_layers),
        "patch_size": int(patch_size),
        "image_size": image_size,
        "num_prefix_tokens": int(num_prefix_tokens),
    }


class LocalHFVisionEncoder(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        out_feature_indexes: list[int],
        patch_size: int | None = None,
        num_prefix_tokens: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        meta = inspect_hf_backbone(backbone_path)

        self.backbone_path = meta["path"]
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size or meta["patch_size"]
        self.num_prefix_tokens = meta["num_prefix_tokens"] if num_prefix_tokens is None else num_prefix_tokens
        self.hidden_size = meta["hidden_size"]

        self.encoder = AutoModel.from_pretrained(
            self.backbone_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self._out_feature_channels = [self.hidden_size] * len(self.out_feature_indexes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"Backbone requires input shape divisible by patch_size={self.patch_size}, got {tuple(x.shape)}"
            )

        outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Local backbone did not return hidden states; cannot build RF-DETR feature maps")

        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        expected_tokens = grid_h * grid_w

        feature_maps: list[torch.Tensor] = []
        for index in self.out_feature_indexes:
            sequence = hidden_states[index + 1]
            if sequence.ndim != 3:
                raise RuntimeError(f"Expected 3D hidden state tensor, got shape {tuple(sequence.shape)}")

            patch_tokens = sequence[:, self.num_prefix_tokens :, :]
            if patch_tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    "Backbone token count does not match the input resolution. "
                    f"Expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}."
                )

            batch_size, _, channels = patch_tokens.shape
            feature_map = patch_tokens.reshape(batch_size, grid_h, grid_w, channels).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(feature_map)

        return feature_maps


class LocalHFBackbone(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        out_channels: int,
        out_feature_indexes: list[int],
        projector_scale: list[str],
        freeze_encoder: bool,
        layer_norm: bool,
        gradient_checkpointing: bool,
        patch_size: int | None = None,
        num_prefix_tokens: int | None = None,
    ):
        super().__init__()
        self.encoder = LocalHFVisionEncoder(
            backbone_path=backbone_path,
            out_feature_indexes=out_feature_indexes,
            patch_size=patch_size,
            num_prefix_tokens=num_prefix_tokens,
            gradient_checkpointing=gradient_checkpointing,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[level] for level in projector_scale]
        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=False,
        )

    def forward_backbone(self, tensor_list: NestedTensor) -> list[NestedTensor]:
        feats = self.projector(self.encoder(tensor_list.tensors))
        out = []
        for feat in feats:
            mask = tensor_list.mask
            assert mask is not None
            resized_mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, resized_mask))
        return out

    def forward(self, tensor_list: NestedTensor):
        return self.forward_backbone(tensor_list)


def build_local_hf_backbone(
    backbone_path: str,
    out_channels: int,
    out_feature_indexes: list[int],
    projector_scale: list[str],
    hidden_dim: int,
    position_embedding: str,
    freeze_encoder: bool,
    layer_norm: bool,
    gradient_checkpointing: bool,
    patch_size: int | None = None,
    num_prefix_tokens: int | None = None,
):
    backbone = LocalHFBackbone(
        backbone_path=backbone_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        gradient_checkpointing=gradient_checkpointing,
        patch_size=patch_size,
        num_prefix_tokens=num_prefix_tokens,
    )
    position_embedding_module = build_position_encoding(hidden_dim, position_embedding)
    return Joiner(backbone, position_embedding_module)


# ─────────────────────────────────── Builder ─────────────────────────────────── #

def build_rfdetr(
    num_classes: int,
    preset: str = "base",
    load_dinov2_weights: bool = True,
    device: str | None = None,
    freeze_encoder: bool = False,
    gradient_checkpointing: bool = False,
    backbone_path: str | None = None,
    backbone_out_feature_indexes: list[int] | None = None,
    backbone_num_prefix_tokens: int | None = None,
) -> tuple[LWDETR, SetCriterion, dict]:
    """
    Build an RF-DETR model, criterion, and postprocessors.

    Args:
        num_classes: Number of foreground classes in your dataset.
        preset: "base" (DINOv2-Small, 560px) or "large" (DINOv2-Base, 560px).
        load_dinov2_weights:
            True  → downloads facebook/dinov2-small (or -base) from HuggingFace.
                     Best quality; requires internet access once to cache weights.
            False → random initialisation for the ViT backbone.
                     Use this if you have zero internet / intranet access.
        backbone_path:
            Optional local Hugging Face model directory containing a vision backbone
            (for example your frozen DINOv3 checkpoint with config.json + safetensors).
            When set, the stock DINOv2 backbone is bypassed.
        backbone_out_feature_indexes:
            Optional backbone block indexes to expose as RF-DETR features.
            Defaults to four evenly spaced blocks across the encoder depth.
        backbone_num_prefix_tokens:
            Optional override for how many non-patch tokens to strip from each
            hidden state (CLS + register tokens).
        device: Target device string.  Defaults to CUDA → MPS → CPU.
        freeze_encoder: Freeze the DINOv2 encoder during training.
        gradient_checkpointing: Trade compute for memory in the ViT encoder.

    Returns:
        model          – LWDETR instance (not yet moved to device)
        criterion      – SetCriterion loss module
        postprocessors – dict {"bbox": PostProcess()}
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    cfg = dict(BASE_CFG if preset == "base" else LARGE_CFG)
    resolution = cfg["resolution"]

    # ── 1. Backbone (DINOv2 + multi-scale projector + position encoding) ── #
    if backbone_path:
        backbone_meta = inspect_hf_backbone(backbone_path)
        out_feature_indexes = backbone_out_feature_indexes or _infer_evenly_spaced_feature_indexes(
            backbone_meta["num_hidden_layers"]
        )
        invalid_indexes = [i for i in out_feature_indexes if i < 0 or i >= backbone_meta["num_hidden_layers"]]
        if invalid_indexes:
            raise ValueError(
                "Invalid backbone feature indexes "
                f"{invalid_indexes}; backbone only has {backbone_meta['num_hidden_layers']} hidden layers"
            )
        backbone = build_local_hf_backbone(
            backbone_path=backbone_meta["path"],
            out_channels=cfg["hidden_dim"],
            out_feature_indexes=out_feature_indexes,
            projector_scale=cfg["projector_scale"],
            hidden_dim=cfg["hidden_dim"],
            position_embedding="sine",
            freeze_encoder=freeze_encoder,
            layer_norm=cfg["layer_norm"],
            gradient_checkpointing=gradient_checkpointing,
            patch_size=backbone_meta["patch_size"],
            num_prefix_tokens=backbone_num_prefix_tokens,
        )
    else:
        backbone = build_backbone(
            encoder=cfg["encoder"],
            vit_encoder_num_layers=cfg["vit_encoder_num_layers"],
            pretrained_encoder=None,
            window_block_indexes=None,       # computed internally by DinoV2
            drop_path=0.0,
            out_channels=cfg["hidden_dim"],
            out_feature_indexes=cfg["out_feature_indexes"],
            projector_scale=cfg["projector_scale"],
            use_cls_token=False,
            hidden_dim=cfg["hidden_dim"],
            position_embedding="sine",
            freeze_encoder=freeze_encoder,
            layer_norm=cfg["layer_norm"],
            target_shape=(resolution, resolution),
            rms_norm=False,
            backbone_lora=False,
            force_no_pretrain=not load_dinov2_weights,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=cfg["patch_size"],
            num_windows=cfg["num_windows"],
            positional_encoding_size=cfg["positional_encoding_size"],
        )

    # ── 2. Transformer decoder ── #
    num_feature_levels = len(cfg["projector_scale"])
    transformer = Transformer(
        d_model=cfg["hidden_dim"],
        sa_nhead=cfg["sa_nheads"],
        ca_nhead=cfg["ca_nheads"],
        num_queries=cfg["num_queries"],
        dropout=cfg["dropout"],
        dim_feedforward=cfg["dim_feedforward"],
        num_decoder_layers=cfg["dec_layers"],
        return_intermediate_dec=True,
        group_detr=cfg["group_detr"],
        two_stage=cfg["two_stage"],
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg["dec_n_points"],
        lite_refpoint_refine=cfg["lite_refpoint_refine"],
        decoder_norm_type="LN",
        bbox_reparam=cfg["bbox_reparam"],
    )

    # ── 3. Full model (backbone + transformer + detection heads) ── #
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,          # detection only — no seg head
        num_classes=num_classes + 1,     # +1 for no-object background class
        num_queries=cfg["num_queries"],
        aux_loss=True,                   # auxiliary loss per decoder layer
        group_detr=cfg["group_detr"],
        two_stage=cfg["two_stage"],
        lite_refpoint_refine=cfg["lite_refpoint_refine"],
        bbox_reparam=cfg["bbox_reparam"],
    )

    # ── 4. Hungarian matcher ── #
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal_alpha=0.25,
    )

    # ── 5. Loss weights — replicated for each auxiliary decoder head ── #
    weight_dict = {
        "loss_ce":   1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }
    # Auxiliary outputs (intermediate decoder layers)
    aux_weight_dict = {}
    for i in range(cfg["dec_layers"] - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    # Two-stage encoder output
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    # ── 6. Criterion ── #
    criterion = SetCriterion(
        num_classes=num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
        group_detr=cfg["group_detr"],
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=True,               # IoU-aware BCE (better than plain focal)
    )

    # ── 7. Postprocessor (converts normalised coords → pixel coords) ── #
    postprocessors = {"bbox": PostProcess(num_select=cfg["num_queries"])}

    return model, criterion, postprocessors


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
