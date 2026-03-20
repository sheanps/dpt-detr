import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """
    Creates per-scale channel projection convolutions (layer1_rn ... layer4_rn).
    Used in the DPT head to normalize all scales to a common feature dimension
    before FPN-style fusion.

    Ref: DPT (Vision Transformers for Dense Prediction, Ranftl et al.)
    """
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """
    Two-layer residual conv block applied within each FPN stage.
    Refines features after scale fusion before passing to the next stage.

    Ref: DPT (Vision Transformers for Dense Prediction, Ranftl et al.)
    """

    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """
    Fuses two adjacent FPN scales via residual refinement.

    For detection (DETR) we do NOT upsample to pixel resolution —
    we stop at the finest spatial scale and pass to the decoder as memory.
    Upsampling is controlled by passing an explicit `size` argument to
    hold spatial resolution steady rather than doubling it.

    Ref: DPT (Vision Transformers for Dense Prediction, Ranftl et al.)
    Adapted: removed full-resolution upsample path used in segmentation.
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        self.size = size

        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            # Refine skip connection then fuse
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        # Refine after fusion
        output = self.resConfUnit2(output)

        # Upsample — for detection, caller passes size=current_scale_shape
        # to hold resolution instead of doubling (unlike segmentation head)
        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


class SimpleFPN(nn.Module):
    """
    Lightweight FPN that projects 4 ViT feature scales to a common dimension
    and fuses them top-down via nearest-neighbour upsample + addition.

    With DINOv2 as backbone, ViT self-attention already provides global context
    so no encoder (AIFI / CCFM) is needed before the DETR decoder.

    Ref: Feature Pyramid Networks (Lin et al.)
    Inspiration: RF-DETR uses single fused scale as decoder memory rather
    than multi-scale concat (RT-DETR style) — we follow the same decision.
    """

    def __init__(self, in_channels, out_dim=256):
        """
        Args:
            in_channels: list of 4 channel dims from ViT intermediate layers
            out_dim: common feature dimension after projection
        """
        super().__init__()

        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_dim, kernel_size=1) for c in in_channels
        ])
        # Light 3x3 refinement after each top-down fusion step
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, features):
        """
        Args:
            features: list of 4 tensors, each (B, C_i, H_i, W_i)
        Returns:
            finest fused feature map (B, out_dim, H_0, W_0)
        """
        maps = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down fusion: coarse → fine
        for i in range(len(maps) - 2, -1, -1):
            maps[i] = maps[i] + F.interpolate(
                maps[i + 1], size=maps[i].shape[-2:], mode="nearest"
            )
            maps[i] = self.smooths[i](maps[i])

        return maps[0]