import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    """
    Single DETR decoder layer with:
      - Self-attention over object queries
      - Cross-attention from queries to encoder memory
      - FFN

    Positional queries are added to Q and K in both attention steps
    but NOT to V — this is standard practice from the original DETR paper.

    Ref: DETR (End-to-End Object Detection with Transformers, Carion et al.)
    """

    def __init__(self, dim=256, num_heads=8, ffn_dim=1024, dropout=0.0):
        super().__init__()

        # Self-attention — queries attend to each other
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)

        # Cross-attention — queries attend to flattened encoder memory (path_1)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt, memory, pos):
        """
        Args:
            tgt:    (Q, B, dim) — object query content
            memory: (HW, B, dim) — flattened FPN output (path_1)
            pos:    (Q, B, dim) — learned positional embeddings for queries
        Returns:
            tgt:    (Q, B, dim)
        """
        # Self-attention — add pos to Q and K only
        q = k = tgt + pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = self.norm1(tgt + tgt2)

        # Cross-attention — queries (with pos) attend to memory
        q = tgt + pos
        tgt2 = self.cross_attn(q, memory, memory)[0]
        tgt = self.norm2(tgt + tgt2)

        # FFN
        tgt = self.norm3(tgt + self.ffn(tgt))

        return tgt


class DETRDecoder(nn.Module):
    """
    DETR transformer decoder.

    Takes flattened FPN memory and refines a set of learned object queries
    across N decoder layers. Each layer output is supervised during training
    via auxiliary losses to accelerate convergence.

    Design decisions:
      - Single scale memory (path_1 from FPN) rather than multi-scale concat.
        DINOv2 + FPN fusion means the memory is already rich — no need for
        multi-scale decoder input as used in RT-DETR.
      - Focal loss on classification means num_classes output (no +1 for background).
        Background is handled implicitly by the focal loss down-weighting.

    Ref: DETR (End-to-End Object Detection with Transformers, Carion et al.)
    Ref: RT-DETR (DETRs Beat YOLOs on Real-time Object Detection, Zhao et al.)
         — auxiliary loss per decoder layer for faster convergence
    Ref: RF-DETR (Roboflow) — single scale memory from fused backbone features
    """

    def __init__(
        self,
        num_queries=300,
        dim=256,
        num_heads=8,
        num_layers=6,
        num_classes=2,
        ffn_dim=1024,
        dropout=0.0,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.dim = dim
        self.num_layers = num_layers

        # Learned object queries — content and positional are separate
        # Ref: Conditional DETR / DAB-DETR decomposition of content vs pos
        self.query_content = nn.Embedding(num_queries, dim)
        self.query_pos = nn.Embedding(num_queries, dim)

        self.layers = nn.ModuleList([
            DecoderLayer(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

        # Classification head — num_classes only, no background class needed
        # Background handled implicitly by focal loss (see loss.py)
        self.class_head = nn.Linear(dim, num_classes)

        # Box regression head — outputs normalized [cx, cy, w, h] via sigmoid
        # 3-layer MLP following original DETR paper
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 4),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_content.weight)
        nn.init.xavier_uniform_(self.query_pos.weight)

    def forward(self, memory):
        """
        Args:
            memory: (HW, B, dim) — flattened path_1 from FPN
        Returns:
            logits: (B, Q, num_classes)
            boxes:  (B, Q, 4) normalized [cx, cy, w, h]
            aux_outputs: list of (logits, boxes) per intermediate layer
                         used for auxiliary losses during training
        """
        B = memory.shape[1]

        tgt = self.query_content.weight.unsqueeze(1).expand(-1, B, -1)  # (Q, B, dim)
        pos = self.query_pos.weight.unsqueeze(1).expand(-1, B, -1)      # (Q, B, dim)

        aux_outputs = []

        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, memory, pos)

            # Auxiliary loss on every layer except the last
            # Ref: RT-DETR — per-layer supervision accelerates convergence
            if i < self.num_layers - 1:
                tgt_norm = self.norm(tgt).permute(1, 0, 2)  # (B, Q, dim)
                aux_outputs.append({
                    "logits": self.class_head(tgt_norm),
                    "boxes": self.bbox_head(tgt_norm),
                })

        tgt = self.norm(tgt).permute(1, 0, 2)  # (B, Q, dim)

        logits = self.class_head(tgt)   # (B, Q, num_classes)
        boxes = self.bbox_head(tgt)     # (B, Q, 4)

        return logits, boxes, aux_outputs