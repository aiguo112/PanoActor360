"""Spherical transformer for cached DINOv2 segment features.

Grid tokens use the 8x16 pooled ERP. Row r is the latitude center
90 - (r + 0.5) * 22.5 degrees (row 0 is north). Column c is the longitude
center -180 + (c + 0.5) * 22.5 degrees. Person lat/lon are radians and person
fov is degrees.

The dataset applies yaw and flips to the feature grid before this module runs.
Position encodings stay tied to the array axes, so a rolled column receives the
longitude embedding of its new column.
"""

from __future__ import annotations

import math

import torch
from torch import nn

N_FRAMES = 16
GRID_H = 8
GRID_W = 16
FEAT_DIM = 768
D_MODEL = 256
MAX_PERSONS = 128
N_CLASSES = 27
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
CELL_DEG = 22.5

TYPE_CLS = 0
TYPE_FRAME = 1
TYPE_PERSON = 2
TYPE_NULL = 3


def latitude_degrees(rows: int = GRID_H, step: float = CELL_DEG) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.float32)
    return 90.0 - (row + 0.5) * step


def longitude_degrees(cols: int = GRID_W, step: float = CELL_DEG) -> torch.Tensor:
    col = torch.arange(cols, dtype=torch.float32)
    return -180.0 + (col + 0.5) * step


def make_encoder(d: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d,
        nhead=N_HEADS,
        dim_feedforward=4 * d,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttentionPool(nn.Module):
    """Attention pool over an ERP grid. Area mode reweights by cos(latitude)."""

    def __init__(self, d: int, area_pool: bool):
        super().__init__()
        self.area_pool = area_pool
        self.proj = nn.Linear(d, d)
        self.score = nn.Linear(d, 1, bias=False)

    def forward(self, tokens: torch.Tensor, cos_lat: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, H, W, D]
        logits = self.score(torch.tanh(self.proj(tokens))).squeeze(-1)
        weights = torch.softmax(logits.flatten(-2), dim=-1).view_as(logits)
        if self.area_pool:
            weights = weights * cos_lat.view(1, 1, -1, 1)
            weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (weights.unsqueeze(-1) * tokens).sum(dim=(-3, -2))


class GatedAttentionPool(nn.Module):
    """Gated attention MIL (Ilse et al., 2018)."""

    def __init__(self, d: int):
        super().__init__()
        self.V = nn.Linear(d, d)
        self.U = nn.Linear(d, d)
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, N, D], valid: [B, N] True where the token is real
        gates = torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens))
        score = self.w(gates).squeeze(-1)
        score = score.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(score, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        pooled = (alpha.unsqueeze(-1) * tokens).sum(dim=1)
        return pooled, alpha


class HARModel(nn.Module):
    def __init__(self, config: dict, num_classes: int = N_CLASSES, d: int = D_MODEL, dropout: float = DROPOUT):
        super().__init__()
        self.global_stream = config["global_stream"]
        self.temporal = config["temporal"]
        self.lat_enc = bool(config["lat_enc"])
        self.lon_mode = config["lon_mode"]
        self.area_pool = bool(config["area_pool"])
        self.person_kind = config["persons"]
        self.use_persons = self.person_kind is not None
        self.use_grid = self.global_stream == "grid"
        self.mil_enabled = bool(config["mil"])
        self.d = d
        if self.lon_mode not in ("circular", "linear", "none"):
            raise ValueError(f"unknown lon_mode {self.lon_mode}")
        if self.global_stream not in ("grid", "cls", "cubemap", "none"):
            raise ValueError(f"unknown global_stream {self.global_stream}")
        if self.mil_enabled and not self.use_persons:
            raise ValueError("mil requires a person stream")
        if self.global_stream == "none" and not self.use_persons:
            raise ValueError("the model needs a global stream or a person stream")

        self.mean_baseline = self.global_stream == "cls" and self.temporal == "mean"
        self.temporal_baseline = self.global_stream in ("cls", "cubemap") and not self.mean_baseline
        if self.mean_baseline:
            head_in = FEAT_DIM
        elif self.mil_enabled:
            head_in = d * 2
        else:
            head_in = d
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, num_classes))

        if self.temporal_baseline or self.use_grid or self.use_persons:
            self.temporal_embed = nn.Embedding(N_FRAMES, d)
        if self.temporal_baseline:
            self.frame_proj = nn.Linear(FEAT_DIM, d)
            self.temporal_encoder = make_encoder(d, dropout)
        if self.use_grid or self.use_persons:
            self.fusion = make_encoder(d, dropout)
            self.type_embed = nn.Embedding(4, d)
            self.cls_token = nn.Parameter(torch.empty(1, 1, d))
            nn.init.normal_(self.cls_token, std=0.02)
        if self.use_grid:
            self.grid_proj = nn.Linear(FEAT_DIM, d)
            self.spatial_pool = SpatialAttentionPool(d, self.area_pool)
            lat = torch.deg2rad(latitude_degrees())
            lon = torch.deg2rad(longitude_degrees())
            self.register_buffer("cos_lat", torch.cos(lat).contiguous())
            if self.lat_enc:
                lat_feat = torch.stack((torch.sin(lat), torch.cos(lat)), dim=-1)
                self.lat_mlp = MLP(2, d, dropout)
                self.register_buffer("lat_feat", lat_feat.contiguous())
            if self.lon_mode == "circular":
                lon_feat = torch.stack((torch.sin(lon), torch.cos(lon)), dim=-1)
                self.lon_mlp = MLP(2, d, dropout)
                self.register_buffer("lon_feat", lon_feat.contiguous())
            elif self.lon_mode == "linear":
                self.lon_mlp = MLP(1, d, dropout)
                self.register_buffer("lon_feat", (lon / math.pi).unsqueeze(-1).contiguous())
        if self.use_persons:
            self.person_proj = nn.Linear(FEAT_DIM, d)
            self.person_mlp = MLP(6, d, dropout)
            self.null_token = nn.Parameter(torch.empty(1, 1, d))
            nn.init.normal_(self.null_token, std=0.02)
        if self.mil_enabled:
            self.mil = GatedAttentionPool(d)

    def forward(self, batch: dict, return_alpha: bool = False):
        frame_mask = batch["frame_mask"]
        if self.mean_baseline:
            logits = self._mean_logits(batch["cls"], frame_mask)
            return (logits, None) if return_alpha else logits
        if self.temporal_baseline:
            feats = batch["faces"].mean(dim=2) if self.global_stream == "cubemap" else batch["cls"]
            logits = self._temporal_logits(feats, frame_mask)
            return (logits, None) if return_alpha else logits
        logits, alpha = self._fusion_logits(batch, frame_mask)
        if return_alpha:
            return logits, alpha
        return logits

    def _mean_logits(self, cls_tokens: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        weights = frame_mask.unsqueeze(-1).to(cls_tokens.dtype)
        pooled = (cls_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)

    def _temporal_logits(self, frame_feats: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.frame_proj(frame_feats) + self.temporal_embed.weight.unsqueeze(0)
        encoded = self.temporal_encoder(tokens, src_key_padding_mask=~frame_mask)
        weights = frame_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)

    def _person_valid(self, batch: dict, frame_mask: torch.Tensor) -> torch.Tensor:
        frames = batch["person_frame"].clamp(0, N_FRAMES - 1)
        on_kept_frame = frame_mask.gather(1, frames)
        return batch["person_mask"] & on_kept_frame

    def _grid_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.shape[1:] != (N_FRAMES, GRID_H, GRID_W, FEAT_DIM):
            raise RuntimeError(f"expected grid [B, {N_FRAMES}, {GRID_H}, {GRID_W}, {FEAT_DIM}], got {tuple(grid.shape)}")
        tokens = self.grid_proj(grid)
        if self.lat_enc:
            lat = self.lat_mlp(self.lat_feat).view(1, 1, GRID_H, 1, self.d)
            tokens = tokens + lat
        if self.lon_mode != "none":
            lon = self.lon_mlp(self.lon_feat).view(1, 1, 1, GRID_W, self.d)
            tokens = tokens + lon
        return tokens

    def _person_tokens(self, batch: dict, valid: torch.Tensor) -> torch.Tensor:
        lat = batch["person_lat"]
        lon = batch["person_lon"]
        geom = torch.stack(
            (
                torch.sin(lat),
                torch.cos(lat),
                torch.sin(lon),
                torch.cos(lon),
                batch["person_fov"] / 90.0,
                batch["person_conf"],
            ),
            dim=-1,
        )
        tokens = self.person_proj(batch["person_feat"]) + self.person_mlp(geom)
        tokens = tokens + self.temporal_embed(batch["person_frame"].clamp(0, N_FRAMES - 1))
        return tokens * valid.unsqueeze(-1).to(tokens.dtype)

    def _fusion_logits(self, batch: dict, frame_mask: torch.Tensor):
        pieces = []
        keep = []
        batch_size = frame_mask.shape[0]
        ones = torch.ones(batch_size, 1, dtype=torch.bool, device=frame_mask.device)
        cls = self.cls_token.expand(batch_size, -1, -1) + self.type_embed.weight[TYPE_CLS]
        pieces.append(cls)
        keep.append(ones)

        if self.use_grid:
            pooled = self.spatial_pool(self._grid_tokens(batch["grid"]), self.cos_lat)
            frames = pooled + self.temporal_embed.weight.unsqueeze(0)
            frames = frames + self.type_embed.weight[TYPE_FRAME]
            frames = frames * frame_mask.unsqueeze(-1).to(frames.dtype)
            pieces.append(frames)
            keep.append(frame_mask)

        alpha = None
        if self.use_persons:
            valid = self._person_valid(batch, frame_mask)
            persons = self._person_tokens(batch, valid) + self.type_embed.weight[TYPE_PERSON]
            persons = persons * valid.unsqueeze(-1).to(persons.dtype)
            null = self.null_token.expand(batch_size, -1, -1) + self.type_embed.weight[TYPE_NULL]
            pieces.append(persons)
            pieces.append(null)
            keep.append(valid)
            keep.append(ones)
        else:
            valid = None

        sequence = torch.cat(pieces, dim=1)
        padding = ~torch.cat(keep, dim=1)
        if sequence.shape[1] != padding.shape[1]:
            raise RuntimeError(f"sequence {sequence.shape[1]} != mask {padding.shape[1]}")
        encoded = self.fusion(sequence, src_key_padding_mask=padding)
        cls_out = encoded[:, 0]

        if self.mil_enabled:
            offset = 1 + (N_FRAMES if self.use_grid else 0)
            person_out = encoded[:, offset : offset + MAX_PERSONS]
            null_out = encoded[:, offset + MAX_PERSONS : offset + MAX_PERSONS + 1]
            mil_tokens = torch.cat((person_out, null_out), dim=1)
            mil_valid = torch.cat((valid, ones), dim=1)
            pooled, alpha = self.mil(mil_tokens, mil_valid)
            logits = self.head(torch.cat((cls_out, pooled), dim=-1))
        else:
            logits = self.head(cls_out)
        return logits, alpha
