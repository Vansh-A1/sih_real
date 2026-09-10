import numpy as np
from ..config import Normalization


def fit_normalization(dataset, max_pixels_per_band=250000, seed=42):
    """Deterministic bounded reservoir over valid TRAIN observations only."""
    if dataset.split != "train":
        raise ValueError("Normalization may only be fitted to training data")
    rng = np.random.default_rng(seed)
    reservoirs = [[] for _ in range(4)]
    seen = np.zeros(4, dtype=np.int64)
    for i in range(len(dataset)):
        sample = dataset[i]
        for b in range(4):
            a = sample["x"][b].numpy()
            w = sample["weights"][b].numpy()
            vals = a[(w == 1) & np.isfinite(a)]
            # Uniform random priorities retain a bounded random sample across all scenes.
            keys = rng.random(vals.size)
            reservoirs[b].append((keys, vals))
            seen[b] += vals.size
            if sum(len(x[0]) for x in reservoirs[b]) > 2 * max_pixels_per_band:
                kk = np.concatenate([x[0] for x in reservoirs[b]])
                vv = np.concatenate([x[1] for x in reservoirs[b]])
                selected = np.argpartition(kk, max_pixels_per_band - 1)[:max_pixels_per_band]
                reservoirs[b] = [(kk[selected], vv[selected])]
    means, stds, lows, highs = [], [], [], []
    for b in range(4):
        if seen[b] == 0:
            raise ValueError(f"No fully valid training observations for band {b}")
        keys = np.concatenate([x[0] for x in reservoirs[b]])
        vals = np.concatenate([x[1] for x in reservoirs[b]])
        vals = vals[np.argsort(keys)[:max_pixels_per_band]]
        lo, hi = np.quantile(vals, [.001, .999])
        if hi - lo < 1e-5:
            lo, hi = lo - 1e-4, hi + 1e-4
        clipped = vals.clip(lo, hi)
        means.append(float(clipped.mean()))
        stds.append(float(max(clipped.std(), 1e-4)))
        lows.append(float(lo)); highs.append(float(hi))
    return Normalization(means, stds, lows, highs, f"training_reservoir_seed_{seed}; valid_counts={seen.tolist()}")
