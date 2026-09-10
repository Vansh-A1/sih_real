import argparse
from dataclasses import asdict
from pathlib import Path
import json
import resource
import statistics
import time
import torch
from ..config import load_config, config_from_dict


def emit(value, output=None):
    text = json.dumps(value, indent=2, allow_nan=False)
    if output:
        p = Path(output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    print(text)


def benchmark(config, device="cpu", size=128, repeats=10, batch=1):
    from ..models.network import EvidenceSR
    from ..training.state import environment, seed_everything
    seed_everything(config.train.seed, config.train.threads)
    model = EvidenceSR(config).to(device).eval()
    x = torch.rand(batch, 4, size, size, device=device)
    sync = lambda: torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
    with torch.inference_mode():
        for _ in range(2): model(x, mode="analytical")
        if str(device).startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            sync(); started = time.perf_counter(); model(x, mode="analytical"); sync()
            durations.append(time.perf_counter() - started)
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    return {"mode": "untrained analytical forward only; no optimizer", "input_shape": list(x.shape), "dtype": "float32",
            "device": str(device), "repetitions": repeats, "latency_seconds_median": statistics.median(durations),
            "latency_seconds_min": min(durations), "physical_batch": batch, "configured_accumulation": config.train.accumulation,
            "configured_effective_batch": batch * config.train.accumulation, "training_latency": None,
            "parameters": sum(p.numel() for p in model.parameters()), "parameters_by_module": counts,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "environment": environment()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="S2-EvidenceSR-4X research CLI. Training runs only via the explicit train command.")
    subs = parser.add_subparsers(dest="command", required=True)
    env = subs.add_parser("env", help="Inspect dependencies and hardware")
    env.add_argument("--output")
    check = subs.add_parser("check-config", help="Validate configuration without training")
    check.add_argument("--config", required=True)
    val = subs.add_parser("validate-data", help="Check decoding, masks, grids and split separation")
    val.add_argument("--manifest", required=True); val.add_argument("--metadata-only", action="store_true"); val.add_argument("--output")
    fixtures = subs.add_parser("fixtures", help="Generate labeled procedural test data")
    fixtures.add_argument("--output", required=True); fixtures.add_argument("--count", type=int, default=40)
    fixtures.add_argument("--size", type=int, default=16); fixtures.add_argument("--seed", type=int, default=42); fixtures.add_argument("--config")
    train = subs.add_parser("train", help="Explicitly start training (never invoked by other commands)")
    train.add_argument("--config", required=True); train.add_argument("--manifest", required=True); train.add_argument("--run", required=True)
    train.add_argument("--resume"); train.add_argument("--initialize")
    evaluate = subs.add_parser("evaluate", help="Evaluate a checkpoint against a separate manifest role")
    evaluate.add_argument("--checkpoint", required=True); evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test"); evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu"); evaluate.add_argument("--calibration")
    infer = subs.add_parser("infer", help="Windowed analytical GeoTIFF/COG inference")
    infer.add_argument("--checkpoint", required=True); infer.add_argument("--manifest", required=True)
    infer.add_argument("--sample", required=True); infer.add_argument("--output", required=True); infer.add_argument("--device", default="cpu")
    infer.add_argument("--core", type=int); infer.add_argument("--halo", type=int); infer.add_argument("--overlap", type=int)
    infer.add_argument("--cutoff"); infer.add_argument("--cog", action="store_true"); infer.add_argument("--calibration")
    bench = subs.add_parser("benchmark", help="Measure forward latency and memory without training")
    bench.add_argument("--config", required=True); bench.add_argument("--device", default="cpu"); bench.add_argument("--size", type=int, default=128)
    bench.add_argument("--repeats", type=int, default=10); bench.add_argument("--batch", type=int, default=1); bench.add_argument("--output")
    calibrate = subs.add_parser("calibrate", help="Fit quantiles on the separate calibration split")
    calibrate.add_argument("--checkpoint", required=True); calibrate.add_argument("--manifest", required=True); calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--alpha", type=float, default=.1); calibrate.add_argument("--unit", choices=["scene_max", "pixel"], default="scene_max")
    calibrate.add_argument("--product", choices=["tiled", "whole_chip"], default="tiled"); calibrate.add_argument("--device", default="cpu")
    ablate = subs.add_parser("ablations", help="Prepare reproducible experiment configurations; executes only with --execute")
    ablate.add_argument("--config", required=True); ablate.add_argument("--output", required=True)
    ablate.add_argument("--manifest"); ablate.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "env":
            from ..training.state import environment
            emit(environment(), args.output)
        elif args.command == "check-config":
            emit({"valid": True, "config": load_config(args.config).to_dict()})
        elif args.command == "validate-data":
            from ..data.manifest import validate_manifest
            emit(validate_manifest(args.manifest, not args.metadata_only), args.output)
        elif args.command == "fixtures":
            from ..data.fixtures import generate_fixtures
            sensor = load_config(args.config).sensor if args.config else None
            emit({"manifest": str(generate_fixtures(args.output, args.count, args.size, args.seed, sensor))})
        elif args.command == "train":
            from ..training.engine import train
            emit(train(load_config(args.config), args.manifest, args.run, args.resume, args.initialize))
        elif args.command == "benchmark":
            if min(args.repeats, args.batch, args.size) < 1: raise ValueError("Benchmark counts must be positive")
            emit(benchmark(load_config(args.config), args.device, args.size, args.repeats, args.batch), args.output)
        elif args.command in ("evaluate", "calibrate", "infer"):
            from ..training.state import model_from_checkpoint
            from ..data.manifest import validate_manifest, PairedDataset, load_manifest
            validate_manifest(args.manifest, read_arrays=args.command != "infer")
            model, checkpoint = model_from_checkpoint(args.checkpoint, args.device)
            torch.set_num_threads(model.config.train.threads)
            if args.command == "evaluate":
                from ..training.engine import evaluate
                calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
                if calibration:
                    from ..inference.tiled import product_signature
                    if calibration["product_signature"] != product_signature(model):
                        raise ValueError("Calibration does not match whole-chip evaluation; recalibrate the delivered product")
                emit(evaluate(model, PairedDataset(args.manifest, args.split), args.output, args.device, calibration))
            elif args.command == "calibrate":
                from ..metrics.calibration import calibrate
                emit(calibrate(model, PairedDataset(args.manifest, "calibration"), args.output, args.alpha, args.unit, args.product, args.device))
            else:
                from ..inference.tiled import infer_tiled
                doc, root = load_manifest(args.manifest)
                sample = next((s for s in doc["samples"] if s["id"] == args.sample), None)
                if sample is None: raise ValueError(f"Sample id {args.sample} not found")
                for field in ("core", "halo", "overlap"):
                    if getattr(args, field) is not None: setattr(model.config.tiling, field, getattr(args, field))
                if args.cog: model.config.tiling.cog = True
                model.config.validate()
                result = infer_tiled(model, sample["lr"], root, args.output, device=args.device,
                                     auxiliaries=sample.get("auxiliaries", []) if model.temporal is not None else [],
                                     metadata=sample.get("sensor_metadata"), availability=sample.get("metadata_available"), cutoff=args.cutoff)
                if args.calibration:
                    from ..inference.intervals import export_intervals
                    result["intervals"] = export_intervals(model, result, json.loads(Path(args.calibration).read_text()))
                emit(result)
        elif args.command == "ablations":
            from .experiments import prepare_ablations
            emit(prepare_ablations(load_config(args.config), args.output, args.manifest, args.execute))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"s2sr: {exc}\n")


if __name__ == "__main__":
    main()
