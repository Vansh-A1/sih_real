"""Experimental residual autoencoder, conditional flow, and two-stage gate."""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import NAFBlock
from ..losses import highpass, masked_mean, footprint_weights


class ResidualAutoencoder(nn.Module):
    def __init__(self, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.GELU(), NAFBlock(32),
                                     nn.Conv2d(32, latent, 3, stride=2, padding=1))
        self.decoder = nn.Sequential(nn.Conv2d(latent, 128, 3, padding=1), nn.PixelShuffle(2), NAFBlock(32),
                                     nn.Conv2d(32, 128, 3, padding=1), nn.PixelShuffle(2), NAFBlock(32), nn.Conv2d(32, 4, 3, padding=1))

    def encode(self, residual):
        return self.encoder(residual)

    def decode(self, latent):
        return highpass(self.decoder(latent))

    def forward(self, residual):
        return self.decode(self.encode(residual))


class VelocityNetwork(nn.Module):
    def __init__(self, latent, condition=73):
        super().__init__()
        self.input = nn.Conv2d(latent + condition + 1, 64, 3, padding=1)
        self.body = nn.Sequential(NAFBlock(64), NAFBlock(64), NAFBlock(64))
        self.output = nn.Conv2d(64, latent, 3, padding=1)

    def forward(self, z, time, condition):
        t = torch.as_tensor(time, device=z.device, dtype=z.dtype)
        if t.ndim == 0:
            t = t.expand(z.shape[0])
        t = t.view(-1, 1, 1, 1).expand(-1, 1, *z.shape[-2:])
        return self.output(self.body(self.input(torch.cat([z, condition, t], 1))))


class CandidateGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.preliminary = nn.Sequential(nn.Conv2d(73, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 4, 1))
        self.refined = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 4, 1))

    def forward(self, condition, deterministic, prior, cycle):
        preliminary = F.interpolate(torch.sigmoid(self.preliminary(condition)), size=deterministic.shape[-2:], mode="bilinear", align_corners=False)
        cycle_up = F.interpolate(cycle.abs(), size=deterministic.shape[-2:], mode="bilinear", align_corners=False)
        refined = torch.sigmoid(self.refined(torch.cat([deterministic, prior, cycle_up, preliminary], 1)))
        return preliminary * refined


class ResidualFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.autoencoder = ResidualAutoencoder(config.latent_channels)
        self.velocity = VelocityNetwork(config.latent_channels)
        self.gate = CandidateGate()
        self.register_buffer("autoencoder_ready", torch.tensor(False))
        self.register_buffer("flow_ready", torch.tensor(False))
        self.register_buffer("gate_ready", torch.tensor(False))
        self.register_buffer("autoencoder_relative_mae", torch.tensor(float("inf")))

    def condition(self, features, deterministic, weights, support):
        spatial = features.shape[-2:]
        low_det = F.interpolate(deterministic, size=spatial, mode="area")
        if support is None:
            support = weights.new_zeros(weights.shape[0], 1, *spatial)
        return torch.cat([features, low_det, weights, support], 1)

    def sample(self, condition, seed=0, steps=None, method="euler"):
        steps = self.config.sampling_steps if steps is None else steps
        if type(steps) is not int or steps < 1 or method not in ("euler", "heun"):
            raise ValueError("Sampler requires positive integer steps and euler or heun method")
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        z = torch.randn(condition.shape[0], self.config.latent_channels, *condition.shape[-2:], generator=generator, device=condition.device)
        dt = 1. / steps
        for i in range(steps):
            first = self.velocity(z, i * dt, condition)
            candidate = z + dt * first
            if method == "heun":
                second = self.velocity(candidate, (i + 1) * dt, condition)
                candidate = z + .5 * dt * (first + second)
            z = candidate
        return self.autoencoder.decode(z)

    def generate(self, features, deterministic, weights, support, sensor, observation, seed=0, steps=None, gate_override=None):
        if gate_override == 0:
            return torch.zeros_like(deterministic), torch.zeros_like(deterministic)
        if gate_override is not None and not 0 <= gate_override <= 1:
            raise ValueError("gate_override must be in [0,1]")
        if gate_override is None and not bool(self.gate_ready):
            raise RuntimeError("Candidate gate has not completed training; run the separate gate stage")
        condition = self.condition(features, deterministic, weights, support)
        prior = self.sample(condition, seed, steps)
        cycle = sensor(deterministic + prior) - observation
        gate = self.gate(condition, deterministic, prior, cycle)
        if gate_override is not None:
            gate = torch.full_like(gate, gate_override)
        valid = F.interpolate(weights, scale_factor=4, mode="nearest")
        return prior, gate * valid

    def stage_loss(self, stage, result, batch, sensor, corrector):
        target = torch.where(batch["valid"] > 0, batch["y"], result.deterministic.detach())
        residual = highpass(target - result.deterministic.detach())
        mask = footprint_weights(batch["valid"] * result.valid)
        condition = self.condition(result.features.detach().float(), result.deterministic.detach(), batch["weights"], result.temporal_support)
        if stage == "autoencoder":
            reconstructed = self.autoencoder(residual)
            loss = masked_mean((reconstructed - residual).abs(), mask)
            return loss, {"raw": {"autoencoder_reconstruction": loss}, "weighted": {"autoencoder_reconstruction": loss}}
        if stage == "flow":
            if not bool((batch["valid"] == 1).all()):
                from ..losses import NoValidData
                raise NoValidData("Flow latent supervision requires a fully valid reference chip because the encoder pools globally")
            if not bool(self.autoencoder_ready):
                raise RuntimeError("Autoencoder must pass its reconstruction check before flow training")
            with torch.no_grad():
                z1 = self.autoencoder.encode(residual)
            z0 = torch.randn_like(z1)
            t = torch.rand(z0.shape[0], device=z0.device)
            zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
            velocity = self.velocity(zt, t, condition)
            # Conservative 7x7 HR input support for the two strided encoder convolutions.
            latent_mask = F.interpolate(footprint_weights(batch["valid"].amin(1, keepdim=True), 3), size=z0.shape[-2:], mode="area")
            latent_mask = (latent_mask >= 1 - 1e-6).float()
            loss = masked_mean((velocity - (z1 - z0)).square(), latent_mask)
            return loss, {"raw": {"flow_matching": loss}, "weighted": {"flow_matching": loss}}
        if stage == "gate":
            if not bool(self.flow_ready):
                raise RuntimeError("Flow must complete its training stage first")
            with torch.no_grad():
                prior = self.sample(condition, seed=int(torch.randint(0, 2**31 - 1, ()).item()))
                # Teacher compares PRE-correction local L1 in both cases, no inference HR.
                det_error = F.avg_pool2d((result.deterministic - target).abs(), 3, stride=1, padding=1)
                candidate_error = F.avg_pool2d((result.deterministic + prior - target).abs(), 3, stride=1, padding=1)
                teacher = torch.sigmoid((det_error - candidate_error) / .01)
                cycle = sensor(result.deterministic + prior) - torch.nan_to_num(batch["x"])
            gate = self.gate(condition, result.deterministic, prior, cycle)
            loss = masked_mean(F.binary_cross_entropy(gate.clamp(1e-6, 1 - 1e-6), teacher, reduction="none"), mask)
            return loss, {"raw": {"gate_teacher_bce": loss}, "weighted": {"gate_teacher_bce": loss}}
        raise ValueError(f"Unknown residual training stage: {stage}")

    @torch.no_grad()
    def assess_autoencoder(self, examples):
        """examples yields (physical residual, reliable HR weights) on validation data."""
        numerator, denominator = 0., 0.
        for target, weight in examples:
            reconstructed = self.autoencoder(target)
            numerator += float(((reconstructed - target).abs() * weight).sum())
            denominator += float((target.abs() * weight).sum())
        if denominator <= 0:
            raise ValueError("Cannot assess autoencoder on an empty/zero-residual reference set")
        ratio = numerator / denominator
        self.autoencoder_relative_mae.fill_(ratio)
        self.autoencoder_ready.fill_(ratio < 1.)
        return {"relative_mae_to_zero_residual_baseline": ratio, "passed": ratio < 1., "scientific_validation": False}
