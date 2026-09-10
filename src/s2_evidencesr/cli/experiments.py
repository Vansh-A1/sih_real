from copy import deepcopy
from pathlib import Path
import json


def prepare_ablations(config, output, manifest=None, execute=False):
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Choose a new ablation directory")
    if execute and not manifest:
        raise ValueError("--execute requires --manifest")
    directory.mkdir(parents=True, exist_ok=True)
    variants = {"reference": {}, "joint_conv": {"architecture.spectral_mode": "joint_conv"},
                "naf_only": {"architecture.swin_blocks": 0}, "bicubic_baseline": {"architecture.baseline": "bicubic"},
                "no_correction": {"correction.mode": "none"}, "fourier_only": {"correction.mode": "fourier"},
                "landweber_only": {"correction.mode": "landweber"}, "no_degradation_condition": {"architecture.degradation": False}}
    runs = []
    for name, modifications in variants.items():
        cfg = deepcopy(config)
        for field, value in modifications.items():
            parent, attr = field.split("."); setattr(getattr(cfg, parent), attr, value)
        cfg.validate()
        path = directory / f"{name}.json"; path.write_text(json.dumps(cfg.to_dict(), indent=2))
        item = {"name": name, "config": str(path), "seed": cfg.train.seed, "status": "prepared_not_run"}
        if execute:
            from ..training.engine import train
            item["result"] = train(cfg, manifest, directory / name); item["status"] = "executed"
        runs.append(item)
    report = {"runs": runs, "data_control": "identical manifest, seeds and optimizer-step budgets; report capacity differences",
              "additional_experiments": ["physical versus bicubic training data", "single versus repeated dates", "raw versus calibrated uncertainty", "analytical versus stochastic output"]}
    (directory / "experiments.json").write_text(json.dumps(report, indent=2))
    return report
