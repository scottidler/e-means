#!/usr/bin/env python3
"""Export e-means parity fixtures from the Faiss reference (`faiss-cpu==1.16.0`, `NumpyEmeans`).

Each fixture is one Faiss run driven through its caller-driven API (`begin`, `accumulate`, `update`,
`end_epoch`, `finalize`) with an explicit init and explicit pass orders. Faiss's results go to
`tests/fixtures/`: final centroids, last-pass labels in row order, per-pass objective and split count, and
the resolved recipe, as flat little-endian files plus one `manifest.json`.

Two kinds of draw:

- `faiss`: the init and pass orders `NumpyEmeans.train(x)` itself draws (MT19937, `REF/e_means.py:594-614`),
  stored as files. The rows are stored f32 and off-grid on purpose, so per-cluster f32 sums round
  (deviation D2). Used by the n=4000 fixtures.
- `splitmix`: the rows, init and pass orders come from an integer-only recipe (SplitMix64, Fisher-Yates,
  Irwin-Hall grid noise) that `tests/parity.rs` regenerates bit for bit, so nothing over 1 MB is committed.
  The manifest carries an FNV-1a hash of each, which the Rust side checks. Used by the n=100000 fixtures.

Run through `otto compat` (after `otto compat-setup`).
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means.base import _epoch_seed

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
EMEANS_SEED = 1234

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB

# Streams of the shared recipe: SplitMix64 seeded with `stream_seed + stream id`.
INIT_STREAM = 1
PASS_STREAM = 2  # pass p uses PASS_STREAM + p

# `grid_blobs`: Irwin-Hall noise of 12 ten-bit uniforms, about N(0, 1) in steps of 2^-10.
GRID_SCALE_LOG2 = -10
IRWIN_HALL_TERMS = 12
IRWIN_HALL_CENTER = 6138  # 12 * 1023 / 2


class SplitMix:
    """SplitMix64 over Python ints, the scalar twin of `splitmix_block`."""

    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * MIX1) & MASK
        z = ((z ^ (z >> 27)) * MIX2) & MASK
        return z ^ (z >> 31)


def splitmix_block(seed, start, count):
    """Outputs `start .. start + count` (0-based) of SplitMix64 seeded with `seed`, vectorized."""
    i = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK) + i * np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(MIX1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(MIX2)
    return z ^ (z >> np.uint64(31))


def permutation(n, seed):
    """Fisher-Yates from the top: for i = n-1 down to 1, swap i with `next() % (i + 1)`."""
    rng = SplitMix(seed)
    ids = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.next() % (i + 1)
        ids[i], ids[j] = ids[j], ids[i]
    return np.array(ids, dtype=np.int64)


def grid_noise(seed, start, values):
    """`values` Irwin-Hall draws from SplitMix64 output `start * 12` on, in 2^-10 steps (int64)."""
    raw = splitmix_block(seed, start * IRWIN_HALL_TERMS, values * IRWIN_HALL_TERMS) >> np.uint64(54)
    return raw.reshape(values, IRWIN_HALL_TERMS).astype(np.int64).sum(axis=1) - IRWIN_HALL_CENTER


def grid_blobs(n, d, count, center_scale, seed):
    """Row i = center of blob `i % count` + noise, exactly on the 2^-10 grid; f32.

    Draw order: `count * d` center draws, then `n * d` noise draws, row-major. Center steps are
    `center_scale * draw`; `center_scale` is an integer.
    """
    centers = center_scale * grid_noise(seed, 0, count * d).reshape(count, d)
    steps = np.empty((n, d), dtype=np.int64)
    chunk = 8192
    for begin in range(0, n, chunk):
        end = min(begin + chunk, n)
        noise = grid_noise(seed, count * d + begin * d, (end - begin) * d).reshape(end - begin, d)
        steps[begin:end] = centers[np.arange(begin, end) % count] + noise
    if np.abs(steps).max() >= 1 << 24:
        sys.exit("grid steps leave the exact f32 range")
    return (steps.astype(np.float64) * 2.0**GRID_SCALE_LOG2).astype(np.float32)


def blobs(n, d, count, scale, seed):
    """Off-grid f32 rows: centers ~ N(0, scale^2), unit noise, row i in blob `i % count` (NumPy RNG)."""
    rs = np.random.RandomState(seed)
    centers = rs.randn(count, d) * scale
    x = centers[np.arange(n) % count] + rs.randn(n, d)
    return x.astype("float32")


def normalized(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / norms).astype("float32")


def fnv1a(data):
    """FNV-1a 64 over bytes; the Rust side computes the same over what it regenerates."""
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & MASK
    return h


def fnv1a_u32(values):
    return "%016x" % fnv1a(np.ascontiguousarray(values, dtype="<u4").tobytes())


def fnv1a_f32(values):
    return "%016x" % fnv1a(np.ascontiguousarray(values, dtype="<f4").tobytes())


def make_emeans(d, k, nepochs, spherical, unused_threshold):
    return NumpyEmeans(d, k, nepochs=nepochs, seed=EMEANS_SEED, spherical=spherical,
                       unused_threshold=unused_threshold)


def epoch_rows(n, d, k, nepochs, spherical, unused_threshold):
    """The pass sizes Faiss resolves for this run."""
    km = make_emeans(d, k, nepochs, spherical, unused_threshold)
    km.begin(n, np.zeros((k, d), dtype="float32"))
    return km.recipe["epoch_rows"]


def caller_run(x, init, orders, k, nepochs, spherical, unused_threshold):
    """Drive Faiss's caller-driven API: every pass in its given order, one accumulate + update per batch."""
    n, d = x.shape
    km = make_emeans(d, k, nepochs, spherical, unused_threshold)
    km.begin(n, init)
    batch = km.recipe["batch_size"]
    labels = np.full(n, -1, dtype=np.int64)
    last = len(km.recipe["epoch_rows"]) - 1
    for epoch, order in enumerate(orders):
        for begin in range(0, len(order), batch):
            ids = order[begin:begin + batch]
            stats = km.accumulate(x[ids])
            if epoch == last:
                labels[ids] = stats.pop("labels")
            km.update(stats)
        km.end_epoch()
    centroids = km.finalize()
    return km, centroids, labels


def centroid_drift(a, b):
    """Worst per-centroid `||a - b||_2 / ||b||_2`, in f64."""
    a, b = a.astype("float64"), b.astype("float64")
    return float((np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)).max())


def write(name, array, dtype):
    """Write `array` as flat little-endian `dtype`; a value that does not convert exactly is an error."""
    out = np.ascontiguousarray(array, dtype={"f32": "<f4", "u32": "<u4"}[dtype])
    if not np.array_equal(out, array):
        sys.exit("%s: values change when written as %s" % (name, dtype))
    out.tofile(FIXTURES / name)
    return {"file": name, "dtype": dtype}


def export(spec, x, x_entry):
    name, k = spec["name"], spec["k"]
    n, d = x.shape
    args = (k, spec["epochs"], spec["spherical"], spec["unused_threshold"])
    rows = epoch_rows(n, d, *args)
    stream_seed = spec.get("stream_seed")
    if stream_seed is None:
        init = x[np.random.RandomState(EMEANS_SEED).permutation(n)[:k]]
        orders = [np.random.RandomState(_epoch_seed(EMEANS_SEED, p)).permutation(n)[:r]
                  for p, r in enumerate(rows)]
        stored = spec.get("shared", name)
        draws = {
            "draw": "faiss",
            "init": write(stored + ".init.bin", init, "f32"),
            "orders": write(stored + ".orders.bin", np.concatenate(orders), "u32"),
        }
    else:
        init = x[permutation(n, stream_seed + INIT_STREAM)[:k]]
        orders = [permutation(n, stream_seed + PASS_STREAM + p)[:r] for p, r in enumerate(rows)]
        draws = {"draw": "splitmix", "stream_seed": stream_seed}
    km, centroids, labels = caller_run(x, init, orders, *args)
    if (labels < 0).any():
        sys.exit("%s: the last pass left rows unlabeled" % name)
    recipe = km.recipe
    entry = {
        "name": name,
        "class": spec["class"],
        "n": n,
        "d": d,
        "k": k,
        "epochs": spec["epochs"],
        "spherical": spec["spherical"],
        "unused_threshold": recipe["unused_threshold"],
        "batch_law": recipe["batch_law"],
        "batch_size": recipe["batch_size"],
        "alpha0": recipe["alpha0"],
        "epoch_rows": recipe["epoch_rows"],
        "total_steps": recipe["total_steps"],
        "objectives": [float(stat["obj"]) for stat in km.iteration_stats],
        "splits": [int(stat["nsplit"]) for stat in km.iteration_stats],
        **draws,
        "x": x_entry,
        "x_fnv1a": fnv1a_f32(x),
        "init_fnv1a": fnv1a_f32(init),
        "orders_fnv1a": fnv1a_u32(np.concatenate(orders)),
        "centroids": write(name + ".centroids.bin", centroids, "f32"),
        "labels": write(name + ".labels.bin", labels, "u32"),
    }
    if spec["class"] == "separated":
        # Faiss against itself with its BLAS kernel forced: a separated fixture must not move.
        saved = faiss.cvar.distance_compute_blas_threshold
        faiss.cvar.distance_compute_blas_threshold = 0
        try:
            blas_km, blas_centroids, blas_labels = caller_run(x, init, orders, *args)
        finally:
            faiss.cvar.distance_compute_blas_threshold = saved
        obj, blas_obj = entry["objectives"][-1], blas_km.iteration_stats[-1]["obj"]
        noise = {
            "centroid": centroid_drift(blas_centroids, centroids),
            "objective": abs(blas_obj - obj) / obj,
            "labels": float((blas_labels == labels).mean()),
        }
        entry["faiss_blas_forced"] = noise
        if noise["labels"] != 1.0:
            sys.exit("%s: not separated, Faiss moves its own labels under a forced BLAS kernel: %s"
                     % (name, noise))
    print("%-24s B=%-5d alpha0=%.6g steps=%-4d obj=%.9g splits=%s%s" % (
        name, recipe["batch_size"], recipe["alpha0"], recipe["total_steps"],
        entry["objectives"][-1], entry["splits"],
        "  noise=%s" % entry["faiss_blas_forced"] if "faiss_blas_forced" in entry else ""))
    return entry


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for stale in FIXTURES.glob("*.bin"):
        stale.unlink()
    print("faiss %s, numpy %s, blas threshold %d" % (
        faiss.__version__, np.__version__, faiss.cvar.distance_compute_blas_threshold))

    separated = dict(cls="separated", n=4000, k=16)
    specs = [
        dict(separated, name="separated_d8", d=8, epochs=5.0, spherical=False),
        dict(separated, name="separated_d32", d=32, epochs=5.0, spherical=False),
        dict(separated, name="separated_d8_t2_5", d=8, epochs=2.5, spherical=False),
        dict(separated, name="separated_spherical_d32", d=32, epochs=5.0, spherical=True),
        dict(separated, name="separated_d2", d=2, epochs=5.0, spherical=False),
        dict(cls="loose", name="oversplit", n=4000, d=32, k=64, epochs=5.0, spherical=False),
    ]
    entries = []
    # Fixtures on the same rows share one x file: the d=8 pair, and separated_d32 with oversplit.
    written = {}
    for spec in specs:
        data = "blobs_d%d%s" % (spec["d"], "_unit" if spec["spherical"] else "")
        x = blobs(spec["n"], spec["d"], 16, 10.0, seed=spec["d"])
        if spec["spherical"]:
            x = normalized(x)
        if data not in written:
            written[data] = write(data + ".x.bin", x, "f32")
        spec = dict(spec, unused_threshold=0.0)
        spec["class"] = spec.pop("cls")
        entries.append(export(spec, x, written[data]))

    # Revival off and on share the generated rows, init and orders.
    large_x = {"generator": "grid_blobs", "n": 100000, "d": 32, "blobs": 256, "center_scale": 3,
               "seed": 32}
    x = grid_blobs(100000, 32, 256, 3, seed=32)
    large = dict(cls="loose", k=256, epochs=5.0, spherical=False, stream_seed=32000)
    large["class"] = large.pop("cls")
    entries.append(export(dict(large, name="large", unused_threshold=0.0), x, large_x))
    entries.append(export(dict(large, name="large_revival", unused_threshold=None), x, large_x))

    manifest = {
        "faiss": faiss.__version__,
        "numpy": np.__version__,
        "emeans_seed": EMEANS_SEED,
        "fixtures": entries,
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote %d fixtures to %s" % (len(entries), FIXTURES))


if __name__ == "__main__":
    main()
