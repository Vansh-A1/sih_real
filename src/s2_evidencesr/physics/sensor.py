"""D=S H with zero-padded grouped cross-correlation and an exact discrete transpose."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..config import Sensor


class SensorOperator(nn.Module):
    def __init__(self, config: Sensor):
        super().__init__()
        self.config = config
        k = config.kernel_size
        if config.kernels is None:
            p = torch.arange(k, dtype=torch.float64) - k // 2
            yy, xx = torch.meshgrid(p, p, indexing="ij")
            kernel = torch.stack([torch.exp(-(xx.square() + yy.square()) / (2 * s * s)) for s in config.sigma])
        else:
            kernel = torch.tensor(config.kernels, dtype=torch.float64)
        if kernel.shape != (4, k, k) or not torch.isfinite(kernel).all() or (kernel < 0).any() or (kernel.sum((1, 2)) <= 0).any():
            raise ValueError("PSF kernels must be four finite, nonnegative, nonempty KxK arrays")
        kernel = kernel / kernel.sum((1, 2), keepdim=True)
        self.register_buffer("kernel", kernel[:, None].float())
        self.radius = k // 2
        self.phase = tuple(config.phase)
        self._norm_cache = {}

    def forward(self, y):
        """Physical [B,4,4H,4W] -> physical [B,4,H,W]."""
        if y.ndim != 4 or y.shape[1] != 4 or any(n % 4 for n in y.shape[-2:]):
            raise ValueError("Sensor HR tensor must be [B,4,4H,4W]")
        blurred = F.conv2d(y, self.kernel.to(y), padding=self.radius, groups=4)
        h, w = y.shape[-2] // 4, y.shape[-1] // 4
        sampled = y.new_zeros(y.shape[0], 4, h, w)
        for py, px, coefficient in self.sampling_terms():
            part = blurred[..., py::4, px::4]
            sampled[..., :part.shape[-2], :part.shape[-1]] += coefficient * part
        return sampled

    def adjoint(self, x, hr_shape=None):
        """True transpose, including zero extension, sampling phase, and crop."""
        shape = hr_shape or (4 * x.shape[-2], 4 * x.shape[-1])
        if tuple(shape) != (4 * x.shape[-2], 4 * x.shape[-1]):
            raise ValueError("Explicit HR shape must equal four times observation shape")
        up = x.new_zeros(*x.shape[:2], *shape)
        for py, px, coefficient in self.sampling_terms():
            selected = up[..., py::4, px::4]
            up[..., py::4, px::4] += coefficient * x[..., :selected.shape[-2], :selected.shape[-1]]
        return F.conv_transpose2d(up, self.kernel.to(x), padding=self.radius, groups=4)

    def sampling_terms(self):
        """Bilinear interpolation weights; its transpose scatters the same weights.

        Phase 1.5 aligns a native pixel center to the center of its 4x4 HR block.
        Out-of-domain taps use the declared zero extension in both directions.
        """
        iy, ix = math.floor(self.phase[0]), math.floor(self.phase[1])
        fy, fx = self.phase[0] - iy, self.phase[1] - ix
        terms = []
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                if wy * wx > 0:
                    terms.append((iy + dy, ix + dx, wy * wx))
        return terms

    def baseline(self, x, weights=None, fill=None):
        """Mask-aware normalized adjoint; returns estimate and actual coverage.

        Unsupported pixels use a declared neutral physical value (zero by default),
        never an epsilon-disguised observation. Coverage is returned separately.
        """
        w = torch.ones_like(x) if weights is None else weights.expand_as(x).to(x)
        if not torch.isfinite(w).all() or (w < 0).any() or (w > 1).any():
            raise ValueError("Observation weights must be finite in [0,1]")
        safe = torch.where(w > 0, x, torch.zeros_like(x))
        if not torch.isfinite(safe).all():
            raise ValueError("Nonfinite physical observations must have zero weight")
        coverage = self.adjoint(w)
        numerator = self.adjoint(safe * w)
        supported = coverage > torch.finfo(x.dtype).eps
        value = numerator / coverage.clamp_min(torch.finfo(x.dtype).eps)
        neutral = torch.zeros_like(value) if fill is None else fill.expand_as(value)
        return torch.where(supported, value, neutral), coverage

    def observation_weights(self, hr_weights):
        """Conservative minimum reliability over each finite PSF footprint.

        Outside-scene zero extension is known boundary data, not a missing HR pixel.
        """
        w = hr_weights.expand(-1, 4, -1, -1)
        patches = F.unfold(F.pad(w, [self.radius] * 4, value=1), self.config.kernel_size)
        b, _, n = patches.shape
        patches = patches.reshape(b, 4, -1, n)
        support = (self.kernel[:, 0].reshape(1, 4, -1, 1) > 0)
        values = torch.where(support, patches, torch.ones_like(patches)).amin(2)
        values = values.reshape_as(w)
        result = w.new_ones(w.shape[0], 4, w.shape[-2] // 4, w.shape[-1] // 4)
        for py, px, _ in self.sampling_terms():
            part = values[..., py::4, px::4]
            region = (..., slice(0, part.shape[-2]), slice(0, part.shape[-1]))
            result[region] = torch.minimum(result[region], part)
        return result

    @torch.no_grad()
    def norm_squared(self, lr_shape, device, iterations=12):
        """Deterministic power estimate, safeguarded by correction backtracking."""
        key = (tuple(lr_shape), str(device), iterations)
        if key not in self._norm_cache:
            v = torch.ones(1, 4, 4 * lr_shape[0], 4 * lr_shape[1], device=device)
            v /= v.norm()
            for _ in range(iterations):
                v = self.adjoint(self(v))
                v /= v.norm().clamp_min(1e-12)
            self._norm_cache[key] = float(self(v).square().sum().clamp_min(1e-8))
        return self._norm_cache[key]
