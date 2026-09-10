"""Explicitly invoked training loop. Importing this module never starts a run."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import math
import time
import torch
from ..config import Config
from ..data.manifest import PairedDataset, collate_samples, fingerprint, validate_manifest
from ..data.normalization import fit_normalization
from ..models.network import EvidenceSR
from ..losses import reconstruction_loss, NoValidData
from ..metrics import image_metrics, aggregate_scenes
from .state import EMA, StatefulSampler, seed_everything, rng_state, restore_rng, environment, code_fingerprint, save_checkpoint, load_checkpoint


def forward_batch(model, batch, mode="analytical"):
    return model(batch["x"], batch["weights"], batch["metadata"], batch["availability"],
                 batch["auxiliaries"] if model.temporal is not None else [], mode=mode)


def detached_log(log):
    return {k: detached_log(v) if isinstance(v, dict) else float(v.detach()) if torch.is_tensor(v) else v for k, v in log.items()}


@torch.no_grad()
def evaluate(model, dataset, output=None, device="cpu", calibration=None):
    from ..metrics.quality import uncertainty_metrics
    model.eval()
    if calibration is not None:
        from ..inference.tiled import product_signature
        if calibration["product_signature"] != product_signature(model):
            raise ValueError("Calibration weights or delivered product do not match whole-chip evaluation")
        if any(q is None for q in calibration["quantiles"]):
            raise ValueError("Insufficient calibration units for finite evaluation intervals")
    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]
        batch = collate_samples([sample], device)
        result = forward_batch(model, batch)
        valid = batch["valid"] * result.valid
        metrics = image_metrics(result.corrected, batch["y"], valid, model.config.metric_range, result.uncertainty_scale)
        for key, error in (("pre_cycle_mae", result.pre_cycle), ("post_cycle_mae", result.post_cycle)):
            weight = batch["weights"]
            metrics[key] = float((error.abs() * weight).sum() / weight.sum()) if weight.sum() > 0 else None
        if calibration is not None:
            from ..metrics.calibration import intervals
            interval = intervals(result.corrected, result.uncertainty_scale, calibration)
            metrics["uncertainty"] = uncertainty_metrics(result.corrected, torch.nan_to_num(batch["y"]), valid, result.uncertainty_scale, interval)
        rows.append({"id": sample["id"], "scene": sample["scene"], "group": sample["group"], "metrics": metrics})
    report = {"split": dataset.split, "synthetic": dataset.doc.get("synthetic", False), "prediction": "analytical_corrected_whole_chip",
              "scenes": rows, "aggregate": aggregate_scenes(rows), "opensr": "unavailable", "scientific_validation": False}
    if output:
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def diagnostic_panel(result, batch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.nn import functional as F
    x = F.interpolate(torch.nan_to_num(batch["x"][:1]), scale_factor=4, mode="bilinear", align_corners=False)[0].detach().cpu()
    y, sr = torch.nan_to_num(batch["y"][0]).cpu(), result.corrected[0].detach().cpu()
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    rgb = lambda z: z[[2, 1, 0]].permute(1, 2, 0).clamp(0, 1)
    for ax, image, title in zip(axes[0], (rgb(x), rgb(sr), rgb(y), sr[[3, 2, 1]].permute(1, 2, 0).clamp(0, 1)), ("LR interpolation", "SR estimate", "HR reference", "False color NIR")):
        ax.imshow(image); ax.set_title(title)
    for b, ax in enumerate(axes[1]):
        ax.imshow((sr[b] - y[b]).abs(), cmap="magma"); ax.set_title(f"Band {b + 1} absolute error")
    ndvi = lambda z: (z[3] - z[2]) / (z[3] + z[2]).clamp_min(1e-5)
    axes[2, 0].imshow((ndvi(sr) - ndvi(y)).abs(), cmap="magma"); axes[2, 0].set_title("NDVI error")
    axes[2, 1].imshow(result.post_cycle[0].detach().cpu().abs().mean(0), cmap="magma"); axes[2, 1].set_title("Cycle residual (10 m)")
    axes[2, 2].imshow(result.valid[0].cpu().all(0), cmap="gray"); axes[2, 2].set_title("Valid support")
    if result.uncertainty_scale is not None:
        axes[2, 3].imshow(result.uncertainty_scale[0].detach().cpu().mean(0)); axes[2, 3].set_title("Predicted scale (uncalibrated)")
    else:
        axes[2, 3].text(.1, .5, "Uncertainty absent in V1"); axes[2, 3].set_title("Version diagnostic")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def configure_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(stage == "analytical")
    if model.flow is not None:
        for p in model.flow.parameters():
            p.requires_grad_(False)
    if stage != "analytical":
        if model.flow is None:
            raise ValueError("Requested stage needs V4 components")
        module = {"autoencoder": model.flow.autoencoder, "flow": model.flow.velocity, "gate": model.flow.gate}[stage]
        if stage == "flow" and not bool(model.flow.autoencoder_ready):
            raise RuntimeError("Train and assess the residual autoencoder before flow matching")
        if stage == "gate" and not bool(model.flow.flow_ready):
            raise RuntimeError("Train the residual flow before the candidate gate")
        for p in module.parameters():
            p.requires_grad_(True)


def train(config: Config, manifest, run_directory, resume=None, initialize=None, stop_after=None):
    """Create a NEW run directory or explicitly resume; never started automatically.

    stop_after is an engineering-test interrupt limit without changing the schedule.
    No multiworker/prefetched loader: exact CPU resume is testable at optimizer boundaries.
    """
    config.validate()
    summary = validate_manifest(manifest)
    run = Path(run_directory)
    if run.exists() and any(run.iterdir()) and resume is None:
        raise FileExistsError("Run directory is not empty; choose a new run or use --resume")
    run.mkdir(parents=True, exist_ok=True)
    t = config.train
    seed_everything(t.seed, t.threads)
    device = torch.device(t.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose cpu")
    train_set, validation = PairedDataset(manifest, "train"), PairedDataset(manifest, "validation")
    if train_set.doc.get("sensor") is not None and train_set.doc["sensor"] != asdict(config.sensor):
        raise ValueError("Fixture sensor does not match model sensor; use the generating configuration")
    data_hash = fingerprint(manifest)
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        # Any changed schedule/data/model setting makes this initialization, not exact resume.
        if checkpoint["config"] != config.to_dict() or checkpoint["dataset_fingerprint"] != data_hash:
            from ..config import config_from_dict
            old = checkpoint["config"]
            new = config.to_dict()
            new["normalization"] = old["normalization"]
            if new != old or checkpoint["dataset_fingerprint"] != data_hash:
                raise ValueError("Resume requires matching resolved configuration and dataset fingerprint")
            config = config_from_dict(old)
    elif t.fit_normalization:
        config.normalization = fit_normalization(train_set, seed=t.seed)
    elif config.normalization.provenance == "unfitted_example_only":
        raise ValueError("Fit training-only normalization or supply verified fixed statistics")
    model = EvidenceSR(config).to(device)
    if initialize:
        initial = load_checkpoint(initialize)
        if initial["config"]["sensor"] != asdict(config.sensor):
            raise ValueError("Initialization sensor configuration differs; reconcile it explicitly before transfer")
        current = model.state_dict()
        compatible = {k: v for k, v in initial["ema"].items() if k in current and current[k].shape == v.shape}
        if not compatible:
            raise ValueError("Initialization checkpoint has no compatible parameters")
        model.load_state_dict(compatible, strict=False)
        # Preserve initialization normalization to avoid silently changing feature units.
        from ..config import Normalization
        model.set_normalization(Normalization(**initial["config"]["normalization"]))
        config = model.config
        (run / "initialization.json").write_text(json.dumps({"checkpoint": str(initialize), "loaded": sorted(compatible), "new": sorted(set(current) - set(compatible))}, indent=2))
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if t.stage != "analytical" and not (initialize or resume):
        raise ValueError("Residual stages require --initialize or --resume from the analytical system")
    configure_stage(model, t.stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t.lr, weight_decay=t.weight_decay)
    def schedule(step):
        if step < t.warmup_steps:
            return (step + 1) / max(1, t.warmup_steps)
        progress = min(1, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
        return .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t.amp == "fp16")
    ema = EMA(model, t.ema_decay)
    sampler = StatefulSampler(len(train_set), t.seed)
    step, attempts = 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"]); sampler.load_state_dict(checkpoint["sampler"])
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema"].items()}
        step = checkpoint["step"]; attempts = checkpoint["attempts"]; restore_rng(checkpoint["rng"])
    provenance = {"environment": environment(), "code_fingerprint": code_fingerprint(), "dataset_fingerprint": data_hash,
                  "data": summary, "training_stage": t.stage, "real_reference_evaluation": False}
    (run / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2))
    def save():
        save_checkpoint(run / "last.pt", {"model": model.state_dict(), "ema": ema.shadow, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "attempts": attempts,
                "epoch": sampler.epoch, "sampler": sampler.state_dict(), "rng": rng_state(), "config": config.to_dict(),
                "dataset_fingerprint": data_hash, "provenance": provenance})
    start = time.monotonic()
    interrupted = False
    with (run / "train.jsonl").open("a") as logfile:
        try:
            while step < t.max_steps and (stop_after is None or step < stop_after):
                if t.time_budget_seconds and time.monotonic() - start >= t.time_budget_seconds:
                    break
                model.train(); optimizer.zero_grad(set_to_none=True)
                attempts += 1
                if attempts > max(100, t.max_steps * 20):
                    raise RuntimeError("Too many invalid/overflow attempts; inspect train.jsonl")
                failed, log = False, {}
                for _ in range(t.accumulation):
                    samples = [train_set[i] for i in sampler.take(t.batch_size)]
                    batch = collate_samples(samples, device)
                    try:
                        if t.stage == "analytical":
                            with torch.autocast(device_type=device.type, dtype=torch.float16 if t.amp == "fp16" else torch.bfloat16, enabled=t.amp != "none"):
                                result = forward_batch(model, batch)
                            with torch.autocast(device_type=device.type, enabled=False):
                                loss, log = reconstruction_loss(result, batch["y"], batch["valid"], batch["x"], batch["weights"], model.sensor, config.loss)
                        else:
                            with torch.no_grad():
                                result = forward_batch(model, batch)
                            loss, log = model.flow.stage_loss(t.stage, result, batch, model.sensor, model.corrector)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite total loss")
                        scaler.scale(loss / t.accumulation).backward()
                    except (NoValidData, FloatingPointError) as exc:
                        logfile.write(json.dumps({"attempt": attempts, "status": "skipped", "reason": str(exc), "samples": batch["ids"]}) + "\n")
                        failed = True; break
                if failed:
                    optimizer.zero_grad(set_to_none=True); continue
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
                old_scale = scaler.get_scale()
                if not torch.isfinite(norm) and not scaler.is_enabled():
                    logfile.write(json.dumps({"attempt": attempts, "status": "nonfinite_gradients"}) + "\n"); continue
                scaler.step(optimizer); scaler.update()
                success = not scaler.is_enabled() or scaler.get_scale() >= old_scale
                if not success:
                    logfile.write(json.dumps({"attempt": attempts, "status": "amp_overflow"}) + "\n"); continue
                step += 1; scheduler.step(); ema.update(model)
                record = {"step": step, "epoch": sampler.epoch, "seconds": time.monotonic() - start,
                          "lr": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm), **detached_log(log)}
                logfile.write(json.dumps(record, allow_nan=False) + "\n"); logfile.flush()
                if step % t.panel_every == 0:
                    diagnostic_panel(result, batch, run / f"panel_{step:06d}.png")
                if step % t.validate_every == 0:
                    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema.shadow)
                    evaluate(model, validation, run / f"validation_{step:06d}.json", device)
                    model.load_state_dict(original)
                if step % t.checkpoint_every == 0:
                    save()
        except KeyboardInterrupt:
            # Never serialize an ambiguous partially updated optimizer transaction.
            interrupted = True
            (run / "interrupted.json").write_text(json.dumps({"last_completed_in_memory_step": step, "resume": "last.pt contains the last atomic checkpoint, if present"}))
    if not interrupted:
        if model.flow is not None and t.stage in ("autoencoder", "flow", "gate") and step >= t.max_steps:
            if t.stage == "autoencoder":
                from ..losses import highpass, footprint_weights
                def examples():
                    for i in range(len(validation)):
                        batch = collate_samples([validation[i]], device)
                        with torch.no_grad():
                            result = forward_batch(model, batch)
                            target = torch.where(batch["valid"] > 0, batch["y"], result.deterministic)
                            yield highpass(target - result.deterministic), footprint_weights(batch["valid"])
                original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.shadow)
                with torch.no_grad():
                    assessment = model.flow.assess_autoencoder(examples())
                model.load_state_dict(original_state)
                model.flow.autoencoder_ready.fill_(assessment["passed"])
                model.flow.autoencoder_relative_mae.fill_(assessment["relative_mae_to_zero_residual_baseline"])
                (run / "autoencoder_assessment.json").write_text(json.dumps(assessment, indent=2))
                ema.shadow["flow.autoencoder_ready"].copy_(model.flow.autoencoder_ready)
                ema.shadow["flow.autoencoder_relative_mae"].copy_(model.flow.autoencoder_relative_mae)
            else:
                getattr(model.flow, f"{t.stage}_ready").fill_(True)
                ema.shadow[f"flow.{t.stage}_ready"].fill_(True)
        save()
    return {"step": step, "interrupted": interrupted, "elapsed_seconds": time.monotonic() - start, "run": str(run)}
