from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from ..config import Config
from ..physics import SensorOperator, SensorCorrector
from .blocks import BandStem, SpectralAttention, NAFBlock, SwinBlock, FiLM, DegradationEncoder, ResidualDecoder


@dataclass
class SRResult:
    baseline: torch.Tensor
    deterministic_residual: torch.Tensor
    deterministic: torch.Tensor
    pre_correction: torch.Tensor
    corrected: torch.Tensor
    pre_cycle: torch.Tensor
    post_cycle: torch.Tensor
    coverage: torch.Tensor
    valid: torch.Tensor
    correction_log: list
    features: torch.Tensor
    decoder_features: torch.Tensor
    temporal_support: torch.Tensor | None = None
    temporal_gate_loss: torch.Tensor | None = None
    uncertainty_scale: torch.Tensor | None = None
    ensemble_disagreement: torch.Tensor | None = None
    ensemble_predictions: torch.Tensor | None = None
    stochastic_residual: torch.Tensor | None = None
    prior_gate: torch.Tensor | None = None
    stochastic_metadata: dict | None = None


class EvidenceSR(nn.Module):
    """Physical reflectance in/out; feature tensors standardized internally.

    Cycle residuals: [B,4,H,W] at 10 m. Other image outputs: [B,4,4H,4W].
    Invalid observations must have zero weight. Optional outputs are None.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config.validate()
        a = config.architecture
        for name in ("mean", "std", "low", "high"):
            self.register_buffer(name, torch.tensor(getattr(config.normalization, name)).view(1, 4, 1, 1))
        self.stem = BandStem(a.stem_blocks, a.naf_expansion) if a.spectral_mode == "band_tokens" or a.rich_heads else None
        self.spectral = nn.Sequential(*[SpectralAttention() for _ in range(a.spectral_layers)]) if a.spectral_mode == "band_tokens" else nn.Identity()
        self.joint = nn.Conv2d(4, 64, 3, padding=1) if a.spectral_mode == "joint_conv" else None
        self.naf = nn.ModuleList([NAFBlock(64, a.naf_expansion) for _ in range(a.naf_blocks)])
        self.swin = nn.Sequential(*[SwinBlock(window=a.window, shift=0 if i % 2 == 0 else a.window // 2, mlp_ratio=a.mlp_ratio) for i in range(a.swin_blocks)])
        self.degradation = DegradationEncoder(a.metadata_dim) if a.degradation else None
        self.films = nn.ModuleDict({str(i): FiLM(64) for i in range(max(0, a.naf_blocks - 3), a.naf_blocks)}) if a.degradation else nn.ModuleDict()
        self.decoder = ResidualDecoder(a.rich_heads, a.degradation, a.naf_expansion)
        self.sensor = SensorOperator(config.sensor)
        self.corrector = SensorCorrector(config.correction)
        self.temporal = None
        self.uncertainty = None
        self.flow = None
        if a.temporal:
            from .temporal import TemporalEvidence
            self.temporal = TemporalEvidence(a)
        if a.uncertainty:
            from .uncertainty import UncertaintyHead
            self.uncertainty = UncertaintyHead(a.ensemble_members)
        if a.flow:
            from .flow import ResidualFlow
            self.flow = ResidualFlow(a)

    def set_normalization(self, values):
        for name in ("mean", "std", "low", "high"):
            getattr(self, name).copy_(torch.tensor(getattr(values, name), device=self.mean.device).view(1, 4, 1, 1))
        self.config.normalization = values

    def normalize(self, x):
        return (torch.minimum(torch.maximum(x, self.low), self.high) - self.mean) / self.std

    def residual_to_physical(self, residual):
        return residual * self.std  # Never add a mean to a residual.

    def encode_bands(self, x):
        bands = self.spectral(self.stem(x)) if self.stem is not None else None
        return bands, self.joint(x) if self.joint is not None else bands.flatten(1, 2)

    def forward(self, x, weights=None, metadata=None, availability=None, auxiliaries=None,
                mode=None, seed=0, gate_override=None, sampling_steps=None):
        if x.ndim != 4 or x.shape[1] != 4 or min(x.shape[-2:]) < 1:
            raise ValueError("Input must be nonempty [B,4,H,W] in B2/B3/B4/B8 reflectance order")
        if weights is None:
            weights = torch.isfinite(x).to(x.dtype)
        weights = weights.expand_as(x).float()
        if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise ValueError("Pixel reliability weights must be finite in [0,1]")
        if not torch.isfinite(torch.where(weights > 0, x, torch.zeros_like(x))).all():
            raise ValueError("Nonfinite observations need zero validity weight")
        safe = torch.where(weights > 0, x, self.mean.to(x)).float()
        learning = self.normalize(safe)
        bands, features = self.encode_bands(learning)
        condition = self.degradation(learning, metadata, availability) if self.degradation is not None else None
        for i, block in enumerate(self.naf):
            features = block(features)
            if str(i) in self.films:
                features = self.films[str(i)](features, condition)
        features = self.swin(features)
        support, temporal_loss = None, None
        if auxiliaries and self.temporal is None:
            raise ValueError("Auxiliary dates supplied while temporal branch is disabled")
        if self.temporal is not None:
            features, support, temporal_loss = self.temporal(features, safe, weights, auxiliaries or [], self)
        residual, decoded = self.decoder(features, bands, condition)
        # Disable autocast for physical operations and reductions.
        with torch.autocast(device_type=x.device.type, enabled=False):
            physical_residual = self.residual_to_physical(residual.float())
            baseline, coverage = self.sensor.baseline(safe, weights, self.mean)
            if self.config.architecture.baseline == "bicubic":
                baseline = F.interpolate(safe, scale_factor=4, mode="bicubic", align_corners=False)
            deterministic = baseline + physical_residual
            candidate = deterministic
            scale, disagreement, members = None, None, None
            if self.uncertainty is not None:
                scale, corrections = self.uncertainty(decoded.float())
                if corrections is not None:
                    member_list = [self.corrector(deterministic, safe, baseline, self.sensor, weights)[0]]
                    for correction in corrections:
                        member_list.append(self.corrector(deterministic + self.residual_to_physical(correction), safe, baseline, self.sensor, weights)[0])
                    members = torch.stack(member_list)
                    disagreement = members.var(dim=0, unbiased=False).sqrt()
            prior, gate, stochastic = None, None, None
            selected_mode = mode or self.config.architecture.mode
            if selected_mode not in ("analytical", "generative", "change_detection", "disaster"):
                raise ValueError("Unknown inference mode")
            if selected_mode == "generative":
                if self.flow is None:
                    raise ValueError("Generative mode requires enabled V4 components")
                if not bool(self.flow.flow_ready) and gate_override != 0:
                    raise RuntimeError("Flow has no completed training stage; refusing untrained generative output")
                prior, gate = self.flow.generate(features.float(), deterministic, weights, support, self.sensor,
                                                 safe, seed, sampling_steps, gate_override)
                candidate = deterministic + gate * prior
                stochastic = {"seed": seed, "steps": sampling_steps or self.config.architecture.sampling_steps,
                              "experimental": True, "gate_is_calibrated_confidence": False}
            corrected, log = self.corrector(candidate, safe, baseline, self.sensor, weights)
            valid = (F.interpolate(weights, scale_factor=4, mode="nearest") > 0) & (coverage > 0)
            return SRResult(baseline, physical_residual, deterministic, candidate, corrected,
                            self.sensor(candidate) - safe, self.sensor(corrected) - safe,
                            coverage, valid, log, features, decoded, support, temporal_loss,
                            scale, disagreement, members, prior, gate, stochastic)
