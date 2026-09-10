"""Approximate corrections; only isolated safeguarded Landweber ensures descent."""
import torch
from torch import nn
from ..config import Correction


class SensorCorrector(nn.Module):
    def __init__(self, config: Correction):
        super().__init__()
        self.config = config

    def fourier(self, y, baseline):
        h, w = y.shape[-2:]
        fy = torch.fft.fftfreq(h, device=y.device)
        fx = torch.fft.fftfreq(w, device=y.device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        cutoff = y.new_tensor(self.config.cutoff).view(1, 4, 1, 1)
        # Real even mask preserves Hermitian symmetry. Circular FFT boundary.
        mask = torch.sigmoid((cutoff - radius) / self.config.transition)
        return torch.fft.ifft2(mask * torch.fft.fft2(baseline) + (1 - mask) * torch.fft.fft2(y)).real

    @staticmethod
    def objective(sensor, y, x, weights):
        residual = sensor(y) - x
        return .5 * (residual.square() * weights).sum()

    def forward(self, y, x, baseline, sensor, weights):
        cfg = self.config
        x = torch.where(weights > 0, x, torch.zeros_like(x))
        records = []

        def record(name, value):
            records.append({"component": name, "weighted_squared_error": float(self.objective(sensor, value, x, weights).detach()),
                            "valid_weight": float(weights.sum().detach())})

        record("before", y)
        if cfg.mode in ("fourier", "fourier_landweber"):
            y = self.fourier(y, baseline)
            record("fourier", y)
        if cfg.mode in ("landweber", "fourier_landweber") and bool((weights > 0).any()):
            norm = sensor.norm_squared(x.shape[-2:], x.device, cfg.norm_iterations)
            # grad .5||sqrt(W)(DY-X)||² = D^T W(DY-X); W <= I.
            for i in range(cfg.iterations):
                direction = sensor.adjoint(weights * (x - sensor(y)))
                old = self.objective(sensor, y, x, weights).detach()
                eta = cfg.step_margin / norm
                accepted = False
                for _ in range(cfg.backtracks):
                    candidate = y + eta * direction
                    new = self.objective(sensor, candidate, x, weights).detach()
                    if torch.isfinite(new) and new <= old:
                        y = candidate
                        accepted = True
                        break
                    eta *= .5
                record(f"landweber_{i + 1}", y)
                records[-1].update(step=eta if accepted else 0.0, accepted=accepted)
        if cfg.output_range is not None:
            y = y.clamp(*cfg.output_range)
            record("range_safeguard", y)
        record("delivered", y)
        return y, records
