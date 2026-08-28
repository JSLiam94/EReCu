from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dinov2_backbone import DinoV2Backbone
from .dinov2_modules import DepthwiseSeparableHead, LocalPseudoLabelRefinement, MultiCueNativePerception, PoolMaskHead, SpectralTensorAttentionFusion
from .utils import safe_logit, safe_minmax


@dataclass
class EReCuOutput:
    probability: torch.Tensor
    probability_grid: torch.Tensor
    dsc_probabilities: list[torch.Tensor]
    pool_probabilities: list[torch.Tensor]
    attention_seed: torch.Tensor
    local_pseudo: torch.Tensor | None
    mnp_loss: torch.Tensor | None
    mnp_score: torch.Tensor | None
    diagnostics: dict[str, torch.Tensor]
    mnp_features: torch.Tensor | None


class DinoV2EReCuModel(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        resnet_pretrained: bool = False,
        resnet_weights_path: str | None = None,
        mnp_sigma1: float = 1.0,
        mnp_sigma2: float = 2.0,
        mnp_threshold: float = 0.5,
        mnp_dilation: int = 5,
        mnp_samples: int = 5,
        mnp_patch_size: int = 15,
        mnp_grid_patch_size: int | None = None,
        tas_seed_blend: float = 0.65,
        tas_temperature: float = 0.15,
        layers: tuple[int, int, int] = (4, 8, 12),
        backbone_name: str = "weights/dinov2_vitb14_pretrain.pth",
        backbone_local_files_only: bool = False,
        backbone_frozen: bool = False,
        backbone_trainable_last_blocks: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.backbone = DinoV2Backbone(
            backbone_name,
            local_files_only=backbone_local_files_only,
            frozen=backbone_frozen,
            trainable_last_blocks=backbone_trainable_last_blocks,
        )
        self.mnp = MultiCueNativePerception(
            resnet_pretrained=resnet_pretrained,
            resnet_weights_path=resnet_weights_path,
            sigma1=mnp_sigma1,
            sigma2=mnp_sigma2,
            threshold=mnp_threshold,
            dilation=mnp_dilation,
            samples=mnp_samples,
            patch_size=mnp_patch_size,
            backbone_patch_size=self.backbone.patch_size,
            grid_patch_size=mnp_grid_patch_size,
        )
        self.tas_seed_blend = tas_seed_blend
        self.tas_temperature = tas_temperature
        self.dsc_heads = nn.ModuleList([DepthwiseSeparableHead(self.backbone.embed_dim) for _ in layers])
        self.pool_heads = nn.ModuleList([PoolMaskHead() for _ in layers])
        # Preserve the original STAF 50% head-rank compression: DINOv1-S used
        # rank 3 for six heads, while DINOv2-Base needs rank 6 for 12 heads.
        self.staf = SpectralTensorAttentionFusion(
            layers=len(layers),
            heads=self.backbone.num_heads,
            head_rank=max(1, self.backbone.num_heads // 2),
        )
        self.final_head = nn.Sequential(
            nn.Conv2d(1 + 2 * len(layers), 24, 3, padding=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 1, 1),
        )
        # The trainable head predicts a residual over the already useful DINO attention prior.
        # This avoids destroying the pretrained object-discovery signal at the start of self-training.
        nn.init.zeros_(self.final_head[-1].weight)
        nn.init.zeros_(self.final_head[-1].bias)
        self.lpr = LocalPseudoLabelRefinement()

    def forward(
        self,
        image: torch.Tensor,
        native: torch.Tensor,
        make_local_pseudo: bool = False,
        mnp_features: torch.Tensor | None = None,
        compute_mnp: bool = True,
        quality_seed: bool = False,
    ) -> EReCuOutput:
        backbone = self.backbone(image, requested_layers=self.layers)
        tokens = [backbone.tokens[layer] for layer in self.layers]
        dsc_logits = [head(token) for head, token in zip(self.dsc_heads, tokens)]
        pool_logits = [head(token) for head, token in zip(self.pool_heads, tokens)]
        attention_tensor = torch.stack([backbone.attentions[layer] for layer in self.layers], dim=1)
        last_attention = backbone.attentions[self.layers[-1]]
        base_attention_seed = safe_minmax(safe_minmax(last_attention).mean(dim=1, keepdim=True))
        if mnp_features is None and (compute_mnp or make_local_pseudo or quality_seed):
            mnp_features = self.mnp.extract(native, backbone.grid_size)
        quality_analysis = None
        if quality_seed:
            if mnp_features is None:
                raise RuntimeError("TAS quality seed requires MNP features")
            quality_analysis = self.lpr.analyze(last_attention, self.mnp, mnp_features)
            attention_seed = self.lpr.quality_fuse(
                last_attention,
                quality_analysis,
                blend=self.tas_seed_blend,
                temperature=self.tas_temperature,
            )
        else:
            attention_seed = base_attention_seed
        staf_logits = self.staf(attention_tensor)
        residual_logits = self.final_head(torch.cat([staf_logits, *dsc_logits, *pool_logits], dim=1))
        prior_logits = safe_logit(base_attention_seed)
        final_logits = prior_logits + residual_logits
        probability_grid = torch.sigmoid(final_logits)
        probability = F.interpolate(probability_grid, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if mnp_features is None:
            # Inference only: MNP is a training/pseudo-label cue and its frozen
            # ResNet pass is unnecessary when exporting a final mask.
            mnp_score = None
            mnp_loss = None
        else:
            mnp_score = self.mnp.score(probability_grid, mnp_features, soft=True)
            mnp_loss = 1.0 - mnp_score.mean()
        local_pseudo = None
        diagnostics: dict[str, torch.Tensor] = {"staf_failures": torch.tensor(float(self.staf.failures), device=image.device)}
        if mnp_score is not None:
            diagnostics["mnp_score"] = mnp_score.detach()
        if make_local_pseudo:
            if mnp_features is None:
                raise RuntimeError("Local pseudo labels require MNP features")
            local_pseudo, lpr_diag = self.lpr(last_attention, self.mnp, mnp_features, analysis=quality_analysis)
            diagnostics.update(lpr_diag)
        return EReCuOutput(
            probability=probability,
            probability_grid=probability_grid,
            dsc_probabilities=[torch.sigmoid(value) for value in dsc_logits],
            pool_probabilities=[torch.sigmoid(value) for value in pool_logits],
            attention_seed=attention_seed,
            local_pseudo=local_pseudo,
            mnp_loss=mnp_loss,
            mnp_score=mnp_score,
            diagnostics=diagnostics,
            mnp_features=mnp_features,
        )
