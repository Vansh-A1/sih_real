"""Strict JSON configuration: unknown fields and incompatible modes fail early."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math

BANDS = ("B2", "B3", "B4", "B8")


@dataclass
class Normalization:
    mean: list[float] = field(default_factory=lambda: [0.2, 0.25, 0.3, 0.45])
    std: list[float] = field(default_factory=lambda: [0.2] * 4)
    low: list[float] = field(default_factory=lambda: [-0.2] * 4)
    high: list[float] = field(default_factory=lambda: [1.5] * 4)
    provenance: str = "unfitted_example_only"


@dataclass
class Sensor:
    kernel_size: int = 9
    sigma: list[float] = field(default_factory=lambda: [1.3, 1.35, 1.4, 1.5])
    kernels: list | None = None
    phase: list[float] = field(default_factory=lambda: [1.5, 1.5])
    boundary: str = "zero"
    assumption: str = "synthetic Gaussian effective PSF; not Sentinel calibration"


@dataclass
class Correction:
    mode: str = "fourier_landweber"
    iterations: int = 1
    cutoff: list[float] = field(default_factory=lambda: [0.125] * 4)
    transition: float = 0.025
    norm_iterations: int = 12
    step_margin: float = 0.9
    backtracks: int = 20
    output_range: list[float] | None = None


@dataclass
class Architecture:
    spectral_layers: int = 2
    naf_blocks: int = 6
    naf_expansion: int = 2
    swin_blocks: int = 4
    window: int = 8
    mlp_ratio: int = 2
    stem_blocks: int = 1
    rich_heads: bool = False
    spectral_mode: str = "band_tokens"
    baseline: str = "adjoint"
    degradation: bool = True
    metadata_dim: int = 8
    temporal: bool = False
    max_shift: int = 8
    max_local_offset: float = 2.0
    stability_tau: float = 0.08
    uncertainty: bool = False
    ensemble_members: int = 1
    flow: bool = False
    latent_channels: int = 8
    sampling_steps: int = 8
    mode: str = "analytical"


@dataclass
class Train:
    seed: int = 42
    device: str = "cpu"
    threads: int = 4
    max_steps: int = 10000
    batch_size: int = 1
    accumulation: int = 4
    lr: float = 0.0002
    weight_decay: float = 0.0001
    warmup_steps: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: str = "none"
    validate_every: int = 100
    checkpoint_every: int = 100
    panel_every: int = 100
    stage: str = "analytical"
    fit_normalization: bool = True
    time_budget_seconds: float | None = None


@dataclass
class Loss:
    radiometric: float = 1.0
    cycle: float = 0.5
    sam: float = 0.1
    high_frequency: float = 0.1
    uncertainty: float = 0.01
    temporal_gate: float = 0.1


@dataclass
class Tiling:
    core: int = 128
    halo: int = 16
    overlap: int = 16
    cog: bool = False


@dataclass
class Config:
    version: str = "v1"
    normalization: Normalization = field(default_factory=Normalization)
    sensor: Sensor = field(default_factory=Sensor)
    correction: Correction = field(default_factory=Correction)
    architecture: Architecture = field(default_factory=Architecture)
    train: Train = field(default_factory=Train)
    loss: Loss = field(default_factory=Loss)
    tiling: Tiling = field(default_factory=Tiling)
    metric_range: float = 1.0

    def validate(self):
        a, s, c, t = self.architecture, self.sensor, self.correction, self.train
        if self.version not in ("v1", "v2", "v3", "v4"):
            raise ValueError("version must be v1, v2, v3, or v4")
        for name in ("mean", "std", "low", "high"):
            v = getattr(self.normalization, name)
            if len(v) != 4 or not all(math.isfinite(x) for x in v):
                raise ValueError(f"normalization.{name} needs four finite values")
        if min(self.normalization.std) <= 0 or any(l >= h for l, h in zip(self.normalization.low, self.normalization.high)):
            raise ValueError("normalization std must be positive and low < high")
        if s.boundary != "zero" or s.kernel_size < 1 or s.kernel_size % 2 != 1:
            raise ValueError("sensor requires odd kernel size and explicit zero boundary")
        if len(s.phase) != 2 or any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p < 4 for p in s.phase):
            raise ValueError("phase is two finite HR-center offsets in [0,4); fractional values use matched bilinear sampling")
        if len(s.sigma) != 4 or min(s.sigma) <= 0:
            raise ValueError("sigma requires four positive values")
        if c.mode not in ("none", "fourier", "landweber", "fourier_landweber"):
            raise ValueError("unknown sensor correction mode")
        if c.iterations < 1 or c.norm_iterations < 1 or c.backtracks < 1 or not 0 < c.step_margin < 2:
            raise ValueError("invalid correction iteration or step settings")
        if len(c.cutoff) != 4 or any(not 0 < x <= .5 for x in c.cutoff) or c.transition <= 0:
            raise ValueError("Fourier cutoffs are cycles/HR-pixel, in (0,.5]")
        if c.output_range is not None and (len(c.output_range) != 2 or c.output_range[0] >= c.output_range[1]):
            raise ValueError("output_range must be [minimum, maximum]")
        if a.window < 2 or a.window % 2 or a.naf_expansion < 2 or a.naf_expansion % 2:
            raise ValueError("window and NAF expansion must be positive even integers")
        if min(a.spectral_layers, a.naf_blocks, a.swin_blocks, a.stem_blocks) < 0 or a.mlp_ratio < 1:
            raise ValueError("block counts cannot be negative")
        if a.spectral_mode not in ("band_tokens", "joint_conv") or a.baseline not in ("adjoint", "bicubic"):
            raise ValueError("unknown encoder or baseline ablation")
        if a.metadata_dim < 1 or a.ensemble_members < 1 or a.sampling_steps < 1:
            raise ValueError("invalid optional-module dimension")
        if a.mode not in ("analytical", "generative", "change_detection", "disaster"):
            raise ValueError("unknown application mode")
        if a.mode == "generative" and not a.flow:
            raise ValueError("generative mode requires an enabled, trained flow")
        if a.flow and self.version != "v4":
            raise ValueError("flow requires version v4")
        if self.version == "v1" and (a.temporal or a.uncertainty or a.ensemble_members > 1):
            raise ValueError("V1 must remain single-date without uncertainty modules")
        if a.ensemble_members > 1 and not a.uncertainty:
            raise ValueError("ensemble requires uncertainty enabled")
        if a.max_shift < 0 or a.max_local_offset < 0 or a.stability_tau <= 0:
            raise ValueError("invalid alignment settings")
        if min(t.batch_size, t.accumulation, t.max_steps, t.threads, t.validate_every, t.checkpoint_every, t.panel_every) < 1:
            raise ValueError("training counts must be positive")
        if t.lr <= 0 or t.weight_decay < 0 or not 0 <= t.ema_decay < 1 or t.grad_clip <= 0 or t.warmup_steps < 0:
            raise ValueError("invalid optimizer settings")
        if t.amp not in ("none", "fp16", "bf16") or (t.device == "cpu" and t.amp == "fp16"):
            raise ValueError("CPU FP16 training is not supported; use none or bf16")
        if t.stage not in ("analytical", "autoencoder", "flow", "gate"):
            raise ValueError("unknown training stage")
        if t.stage != "analytical" and not a.flow:
            raise ValueError("residual training stages require V4 flow components")
        if self.tiling.core < 1 or self.tiling.halo < 0 or not 0 <= self.tiling.overlap < self.tiling.core:
            raise ValueError("tiling requires 0 <= overlap < core and halo >= 0")
        if self.metric_range <= 0 or any(v < 0 for v in asdict(self.loss).values()):
            raise ValueError("invalid metric range or loss coefficient")
        return self

    def to_dict(self):
        return asdict(self)


def config_from_dict(raw: dict) -> Config:
    sections = {"normalization": Normalization, "sensor": Sensor, "correction": Correction,
                "architecture": Architecture, "train": Train, "loss": Loss, "tiling": Tiling}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    values = dict(raw)
    try:
        for key, cls in sections.items():
            if key in values:
                values[key] = cls(**values[key])
        return Config(**values).validate()
    except TypeError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


def load_config(path: str | Path) -> Config:
    return config_from_dict(json.loads(Path(path).read_text()))
