from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .utils import safe_minmax


def gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(axis.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.outer(kernel)


class TextureExtractor(nn.Module):
    def __init__(self, sigma1: float = 1.0, sigma2: float = 2.0) -> None:
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def _blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel = gaussian_kernel(sigma, x.device, x.dtype)[None, None]
        pad = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=pad)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        lbp = torch.zeros_like(gray)
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[:, :, 1 + dy : 1 + dy + gray.shape[-2], 1 + dx : 1 + dx + gray.shape[-1]]
            lbp = lbp + (neighbor >= gray).to(gray.dtype) * float(2**bit)
        lbp = lbp / 255.0

        dog = self._blur(gray, self.sigma2) - self._blur(gray, self.sigma1)
        dog = safe_minmax(dog)
        return torch.cat([lbp, dog], dim=1)


class FrozenResNet18(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
                backbone = resnet18(weights=weights)
            except Exception as exc:  # network/cache failure should not block smoke runs
                warnings.warn(f"ImageNet ResNet-18 unavailable; falling back to deterministic untrained ResNet-18: {exc}")
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(2026)
                    backbone = resnet18(weights=None)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026)
                backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.stem(image)
        level1 = self.layer1(value)
        level2 = self.layer2(level1)
        return level1, level2


@dataclass
class MNPResult:
    features: torch.Tensor
    soft_score: torch.Tensor
    hard_score: torch.Tensor
    loss: torch.Tensor


class MultiCueNativePerception(nn.Module):
    """LBP + DoG + frozen ResNet-18 native cue module.

    The training objective uses differentiable soft regions; hard score is used for TAS diagnostics.
    """

    def __init__(
        self,
        resnet_pretrained: bool = False,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        threshold: float = 0.5,
        dilation: int = 5,
        samples: int = 5,
        patch_size: int = 15,
    ) -> None:
        super().__init__()
        self.texture = TextureExtractor(sigma1, sigma2)
        self.semantic = FrozenResNet18(resnet_pretrained)
        self.threshold = threshold
        self.dilation = dilation if dilation % 2 else dilation + 1
        self.samples = samples
        self.patch_size = patch_size

    def extract(self, native: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        texture = self.texture(native)
        semantic_input = (native - native.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]) / native.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        semantic_levels = self.semantic(semantic_input)
        texture = F.interpolate(texture, size=grid_size, mode="bilinear", align_corners=False)
        normalized_levels = []
        for semantic in semantic_levels:
            semantic = F.interpolate(semantic, size=grid_size, mode="bilinear", align_corners=False)
            semantic = (semantic - semantic.mean(dim=(-2, -1), keepdim=True)) / (semantic.std(dim=(-2, -1), keepdim=True) + 1e-5)
            normalized_levels.append(semantic)
        
        return torch.cat([texture, *normalized_levels], dim=1)

    def _regions(self, mask: torch.Tensor, soft: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if soft:
            inside = torch.sigmoid((mask - self.threshold) * 12.0)
            dilated = F.max_pool2d(inside, self.dilation, stride=1, padding=self.dilation // 2)
            surface = (dilated - inside).clamp(0, 1)
            outside = (1.0 - dilated).clamp(0, 1)
        else:
            inside = (mask > self.threshold).float()
            dilated = F.max_pool2d(inside, self.dilation, stride=1, padding=self.dilation // 2)
            surface = (dilated - inside).clamp(0, 1)
            outside = (1.0 - dilated).clamp(0, 1)
        return inside, surface, outside

    @staticmethod
    def _prototype(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        numerator = (features * weights).sum(dim=(-2, -1))
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-5)
        return numerator / denominator

    def score(self, mask: torch.Tensor, features: torch.Tensor, soft: bool) -> torch.Tensor:
        inside, surface, outside = self._regions(mask, soft)
        fi = F.normalize(self._prototype(features, inside), dim=1)
        fs = F.normalize(self._prototype(features, surface), dim=1)
        fo = F.normalize(self._prototype(features, outside), dim=1)
        cio = (fi * fo).sum(dim=1)
        cis = (fi * fs).sum(dim=1)
        cso = (fs * fo).sum(dim=1)
        
        dio = (1.0 - cio) * 0.5
        dis = (1.0 - cis) * 0.5
        sso = (cso + 1.0) * 0.5
        return ((dio + dis + sso) / 3.0).clamp(0, 1)

    @staticmethod
    def _balanced_descriptors(
        local_features: torch.Tensor,
        purity: torch.Tensor,
        samples: int,
    ) -> torch.Tensor:

        channels, height, width = local_features.shape
        flat_features = local_features.reshape(channels, height * width).transpose(0, 1)
        flat_purity = purity.flatten()
        candidates = torch.nonzero(flat_purity >= 0.5, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = flat_purity.topk(k=min(samples, flat_purity.numel())).indices
        # Sort spatially, then cover the full region instead of clustering samples.
        candidates = candidates.sort().values
        positions = torch.linspace(0, max(candidates.numel() - 1, 0), samples, device=candidates.device)
        indices = candidates[positions.round().long().clamp_max(candidates.numel() - 1)]
        return flat_features[indices]

    def patch_score(self, mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            inside, surface, outside = self._regions(mask.detach(), soft=False)
            grid_patch = max(1, int(round(self.patch_size / 8.0)))
            if grid_patch % 2 == 0:
                grid_patch += 1
            padding = grid_patch // 2
            local_features = F.avg_pool2d(features.detach(), grid_patch, stride=1, padding=padding)
            region_purities = [
                F.avg_pool2d(region, grid_patch, stride=1, padding=padding)
                for region in (inside, surface, outside)
            ]
            results = []
            for batch_index in range(mask.shape[0]):
                descriptors = [
                    self._balanced_descriptors(local_features[batch_index], purity[batch_index, 0], self.samples)
                    for purity in region_purities
                ]
                fi, fs, fo = [F.normalize(value, dim=1) for value in descriptors]
                cio = (fi * fo).sum(dim=1).mean()
                cis = (fi * fs).sum(dim=1).mean()
                cso = (fs * fo).sum(dim=1).mean()
                dio = (1.0 - cio) * 0.5
                dis = (1.0 - cis) * 0.5
                sso = (cso + 1.0) * 0.5
                results.append(((dio + dis + sso) / 3.0).clamp(0, 1))
            return torch.stack(results)

    def forward(self, mask: torch.Tensor, native: torch.Tensor, grid_size: tuple[int, int]) -> MNPResult:
        features = self.extract(native, grid_size)
        soft_score = self.score(mask, features, soft=True)
        with torch.no_grad():
            hard_score = self.score(mask.detach(), features.detach(), soft=False)
        return MNPResult(features=features, soft_score=soft_score, hard_score=hard_score, loss=1.0 - soft_score.mean())


class DepthwiseSeparableHead(nn.Module):
    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = nn.GroupNorm(24, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.pointwise(F.gelu(self.norm(self.depthwise(x)))))


class PoolMaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual = nn.Conv2d(2, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = safe_minmax(x.mean(dim=1, keepdim=True))
        maximum = safe_minmax(x.amax(dim=1, keepdim=True))
        pooled = 0.5 * (mean + maximum)
        base_logits = torch.logit(pooled.clamp(1e-4, 1.0 - 1e-4))
        return base_logits + self.residual(torch.cat([mean, maximum], dim=1))


class SpectralTensorAttentionFusion(nn.Module):
    def __init__(self, layers: int = 3, heads: int = 6, layer_rank: int = 2, head_rank: int = 3, spectral_rank: int = 4) -> None:
        super().__init__()
        self.layers, self.heads = layers, heads
        self.layer_rank, self.head_rank, self.spectral_rank = layer_rank, head_rank, spectral_rank
        self.project = nn.Conv2d(layers * heads, 1, 1)
        self.residual_gain = nn.Parameter(torch.tensor(0.15))
        self.failures = 0

    @staticmethod
    def _basis(unfolded: torch.Tensor, rank: int) -> torch.Tensor:
        u, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
        return u[:, :, : min(rank, u.shape[-1])]

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        # attention: B, L, H, Hp, Wp
        batch, layers, heads, height, width = attention.shape
        if (layers, heads) != (self.layers, self.heads):
            raise ValueError(f"Expected {(self.layers, self.heads)}, got {(layers, heads)}")
        x = safe_minmax(attention)
        fallback = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        try:
            with torch.autocast(device_type=attention.device.type, enabled=False):
                value = x.float()
                ul = self._basis(value.reshape(batch, layers, heads * height * width), self.layer_rank)
                uh = self._basis(value.permute(0, 2, 1, 3, 4).reshape(batch, heads, layers * height * width), self.head_rank)
                core = torch.einsum("blhxy,blr,bhs->brsxy", value, ul, uh)
                flat = core.flatten(1, 2).flatten(2)
                u, s, vh = torch.linalg.svd(flat, full_matrices=False)
                rank = min(self.spectral_rank, s.shape[-1])
                reconstruction = (u[:, :, :rank] * s[:, None, :rank]) @ vh[:, :rank]
                core = reconstruction.reshape(batch, ul.shape[-1], uh.shape[-1], height, width)
                value = torch.einsum("brsxy,blr,bhs->blhxy", core, ul, uh)
                value = value.reshape(batch, layers * heads, height, width).to(dtype=attention.dtype)
        except RuntimeError:
            self.failures += 1
            value = x.reshape(batch, layers * heads, height, width)
        fused = self.project(value) + self.residual_gain * fallback
        return fused


class LocalPseudoLabelRefinement(nn.Module):
    def __init__(self, entropy_threshold: float = 0.5, score_threshold: float = 0.5, alpha_init: float = 1.2) -> None:
        super().__init__()
        self.entropy_threshold = nn.Parameter(torch.tensor(entropy_threshold))
        self.score_threshold = nn.Parameter(torch.tensor(score_threshold))
        raw = torch.log(torch.expm1(torch.tensor(max(alpha_init - 1.0, 1e-3))))
        self.alpha_raw = nn.Parameter(raw)

    def analyze(
        self,
        teacher_attention: torch.Tensor,
        mnp: MultiCueNativePerception,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = teacher_attention / teacher_attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=(-2, -1))
        entropy = entropy / math_log(float(teacher_attention.shape[-2] * teacher_attention.shape[-1]))
        scores = []
        for head in range(teacher_attention.shape[1]):
            scores.append(mnp.patch_score(safe_minmax(teacher_attention[:, head : head + 1]), features))
        score = torch.stack(scores, dim=1)
        tau_e = self.entropy_threshold.clamp(0.1, 0.9)
        tau_s = self.score_threshold.clamp(0.1, 0.9)
        gate = torch.sigmoid((tau_e - entropy) * 12.0) * torch.sigmoid((score - tau_s) * 12.0)
        # Ensure at least one useful head per image.
        ranking = (score - entropy).argmax(dim=1)
        fallback = torch.zeros_like(gate).scatter(1, ranking[:, None], 0.5)
        gate = torch.maximum(gate, fallback)
        return entropy, score, gate

    def quality_fuse(
        self,
        attention: torch.Tensor,
        analysis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        blend: float,
        temperature: float,
    ) -> torch.Tensor:
        entropy, score, gate = analysis
        maps = safe_minmax(attention)
        quality = (1.0 - entropy) * score
        weights = torch.softmax(quality / max(temperature, 1e-3), dim=1)
        weights = weights * gate.clamp_min(1e-3)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        selected = (maps * weights[:, :, None, None]).sum(dim=1, keepdim=True)
        mean = maps.mean(dim=1, keepdim=True)
        return safe_minmax(blend * selected + (1.0 - blend) * mean)

    def forward(
        self,
        teacher_attention: torch.Tensor,
        mnp: MultiCueNativePerception,
        features: torch.Tensor,
        analysis: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # teacher_attention B,H,Hp,Wp; only teacher branch calls this under no_grad.
        if analysis is None:
            analysis = self.analyze(teacher_attention, mnp, features)
        entropy, score, gate = analysis
        mean = teacher_attention.mean(dim=(-2, -1), keepdim=True)
        std = teacher_attention.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        alpha = 1.0 + F.softplus(self.alpha_raw)
        local = torch.sigmoid((teacher_attention - mean - alpha * std) * 20.0)
        local = 1.0 - torch.prod(1.0 - local * gate[:, :, None, None], dim=1, keepdim=True)
        return local.detach(), {"entropy": entropy.detach(), "score": score.detach(), "gate": gate.detach(), "alpha": alpha.detach()}


def math_log(value: float) -> float:
    # Kept separate so TorchScript-free forward stays readable.
    import math

    return math.log(max(value, 2.0))
