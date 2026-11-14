# e-means

A Rust port of e-means, the streaming mini-batch k-means trainer from Faiss `contrib/e_means`.

Each centroid is the ratio of two exponential moving averages (per-cluster sum and per-cluster count). The port
follows the CPU reference `NumpyEmeans`, pinned at Faiss commit `d1d9789a` (2026-11-12). Where the paper
(arXiv 2611.09501) and the code disagree, the code wins.

## Why this exists

I wanted an e-means implementation I own, in Rust, that I can read end to end and trust, with parity against Faiss
proven by tests that run in CI.

There is another Rust port on crates.io, `emeans`. This crate is not it and shares no code with it. It is
published nowhere: consumers take a git tag. No BLAS linkage: the Gemm kernel is `matrixmultiply`, pure Rust.

## What it does

- Data is flat row-major `&[f32]`, length `n * d`. CPU only, no matrix crate in the API.
- `train(&x)` for data in memory; `begin / accumulate / update / end_epoch / finalize` for data that does not fit.
- Deterministic: the same seed gives bitwise the same result for any thread count and any chunking.
- Spherical mode (`.spherical(true)`), dead-cluster revival, the cosine step schedule, the closing pass.

## Use

```toml
e-means = { git = "https://github.com/scottidler/e-means", tag = "v0.2.1", version = "0.2.1" }
```

In-memory data:

```rust
use e_means::EMeans;

let model = EMeans::new(d, k).epochs(5.0).seed(7).train(&x)?;
let centroids = model.centroids();          // k * d, row-major
let nearest = model.assign(&x)?;            // labels and squared distances
```

Data that does not fit in memory. You draw the init rows and each pass's row order; the trainer never sees
more than one logical batch at a time:

```rust
let builder = EMeans::new(d, k).epochs(5.0);
let mut trainer = builder.begin(n, &init)?;          // init: k * d initial centroids
let batch = trainer.batch_rows();
for &rows_in_pass in &trainer.recipe().epoch_rows.clone() {
    for ids in order_for_this_pass[..rows_in_pass].chunks(batch) {
        // gather these rows from your store, then feed them in any chunking
        trainer.accumulate(&rows)?;
        trainer.update()?;
    }
    trainer.end_epoch()?;
}
let model = trainer.finalize()?;
```

A runnable version of the first form is `cargo run --example blobs`.

## Limit: Gemm and a common offset

Above `B * d` = 128000 the assignment kernel uses a matrix multiply and the norm expansion
`||x||^2 + ||c||^2 - 2 x.c`. On data with a large common offset f32 cancellation picks the wrong centroid. Example:
`x = 10000`, centroids `[10002, 10001]`: direct distances `[4, 1]`, expansion `[0, 0]`, tie to index 0, the wrong
centroid. Faiss has the same limit above its BLAS threshold. Center such data first. Below the threshold the direct
kernel is used and has no such limit. `EMeans::always_gemm(true)` opts into Gemm at every batch size for speed on small
batches and takes the same offset limit.

## Deviations from the Faiss reference

| # | Faiss reference | This crate | why |
|---|---|---|---|
| D1 | NumPy MT19937 / PCG64 streams | own seeded stream | porting NumPy's legacy shuffle buys nothing; parity tests feed init and orders |
| D2 | batch sums f32 (f64 scratch above 2^24 rows); pass sums always f32; pass counts f64 | batch and pass sums f64, counts `u64`, mean taken in f64 then rounded to f32 | one code path, tighter means. Measured in Faiss by forcing its f64 scratch path on 5 small fixtures: worst per-centroid `||dc|| / ||c||` 1.3e-7, labels identical |
| D3 | `n < k` dies with a NumPy `IndexError` | `Error::TooFewRows` before any work | fail loudly with the cause |
| D4 | NaN input dies with a `ValueError` | `Error::NonFiniteInput` from an entry scan | same |
| D5 | labels `int64`, -1 for unvisited | `Vec<Option<u32>>` | `None` reads as "not visited" with no magic value to remember |
| D6 | object guarded against a second run at runtime | `finalize(self)` consumes the trainer | the compiler enforces it |
| D7 | `faiss.knn` picks its kernel per call by `nx * d` of that call | one kernel per run, picked by `B * d` at `begin`; batch-relative blocks | results independent of chunking and thread count |
| D8 | revival threshold sums the masses with NumPy's pairwise f32 reduction | sequential f32 sum | no port of NumPy's reduction; they differ in the last bits for k > 8, and only a cluster within a few ULP of the threshold is affected |
| D9 | no check on `alpha0` range, `batch_size == 0`, batch or pass row counts, or external statistics | typed errors for each | fail loudly at the call that is wrong |
| D10 | spherical `assign` returns similarities | `Assignment.distances` holds `2 - 2 * ip` in spherical mode | one meaning for the field in both modes |

## Parity with Faiss

The reference's own unit fixtures are ported, and end-to-end fixtures are exported from `faiss-cpu==1.16.0`
(the wheel; the spec is the `d1d9789a` checkout, and the two differ only in the scatter-add implementation).
Fixtures are committed under `tests/fixtures/`, so `otto ci` needs no Python. `compat/` regenerates them
against a new pin.

Measured parity against Faiss 1.16.0 (centroid = worst per-centroid relative L2, objective = final-pass
relative, labels = last-pass agreement). The Rust side picks its kernel by `B * d`: `large` runs Gemm, the others
run the direct kernel.

| fixture | centroid | objective | labels |
|---|---|---|---|
| separated_d8 | 3.071e-7 | 1.012e-9 | 100% |
| separated_d32 | 1.013e-7 | 3.160e-9 | 100% |
| separated_d8_t2_5 | 2.976e-7 | 9.618e-10 | 100% |
| separated_spherical_d32 | 1.587e-7 | 1.725e-9 | 100% |
| separated_d2 | 4.663e-7 | 7.673e-9 | 100% |
| oversplit | 1.051e-7 | 2.831e-9 | 100% |
| large | 0 | 5.588e-8 | 100% |

`large_revival` (revival on) is a smoke test: it asserts splits > 0 and a final-pass objective within 10% of Faiss.
The donor choice differs (D1), so Rust and Faiss diverge after the first split (centroid 1.952, objective
2.741e-2, labels 78.290%); centroids and labels are not compared.

## Quality

Held-out MSE on a 256-component, d=32 Gaussian mixture (100000 train rows, 20000 held-out), e-means at 5 epochs
against random-init Lloyd-25 (25 full iterations, k distinct random rows as init, no k-means++), one run per seed:

| seed | e-means T=5 | random-init Lloyd-25, no k-means++ |
|---|---|---|
| 10 | 51.1034 | 57.6729 |
| 11 | 52.9391 | 65.5743 |
| 12 | 51.9625 | 57.7037 |

Single runs, one per seed, so one run can lose to an unlucky init. On a mixture with random component sizes, the
worst of 20 e-means seeds (57.585) lost to Lloyd-25 (57.430) on the same rows.

## Speed

Measured on a 64-thread machine (Threadripper 9970X), 2026-11-14, on the v0.2.1 tree. Load average (1 min) at the
start of each run is given with it. Every number is from the command beside it; rerun them to compare on your hardware.

`otto bench` (criterion, median of the interval; load 2.67 at start):

| bench | kernel | this crate |
|---|---|---|
| train n=100000 d=32 k=256 T=5 (B=8192) | Gemm | 18.30 ms |
| train n=100000 d=8 k=256 T=5 (B=1024) | Direct | 37.48 ms |
| assign 100000 rows d=8 k=256 | Direct | 780.5 us |
| assign 100000 rows d=8 k=256 | Gemm | 562.2 us |
| assign 100000 rows d=32 k=256 | Direct | 2.047 ms |
| assign 100000 rows d=32 k=256 | Gemm | 778.3 us |

Large k, `/usr/bin/time -f "%M KB" target/release/examples/large-k` (n=200000, d=32, k=65536, T=1, 64 threads; `train`
time printed by the example, load 2.84 at start; 3 runs): 0.625, 0.624, 0.604 s; peak RSS 155116, 154720, 151288 KB.
Before the tiling work, the same example took 4.2-4.5 s and 2.2 GB.

Against the `emeans` 0.1.1 crate, `otto h2h` (load 2.69 at start; ratio = median of 15 interleaved paired ratios, ours
/ emeans, so below 1 is faster than `emeans`; 1-thread cells pinned with `taskset -c 2`; ms are medians):

| cell | ours ms | emeans ms | ratio |
|---|---|---|---|
| train d=32, 1 thread | 92.32 | 79.80 | 1.050 |
| train d=32, 8 threads | 17.95 | 18.86 | 0.931 |
| train d=32, 64 threads | 18.82 | 24.41 | 0.772 |
| assign d=32, 1 thread | 16.72 | 16.30 | 1.024 |
| assign d=32, 64 threads | 0.90 | 0.80 | 1.128 |
| train d=8, default, 1 thread | 79.11 | 51.44 | 1.541 |
| train d=8, default, 8 threads | 21.58 | 14.31 | 1.433 |
| train d=8, default, 64 threads | 39.95 | 40.16 | 0.990 |
| assign d=8, default, 1 thread | 17.27 | 9.55 | 1.809 |
| assign d=8, default, 64 threads | 0.92 | 0.54 | 1.642 |
| train d=8, `always_gemm(true)`, 1 thread | 58.21 | 51.15 | 1.123 |
| train d=8, `always_gemm(true)`, 8 threads | 13.89 | 12.39 | 1.111 |
| train d=8, `always_gemm(true)`, 64 threads | 33.50 | 40.00 | 0.833 |
| assign d=8, `always_gemm(true)`, 1 thread | 11.16 | 10.41 | 1.072 |
| assign d=8, `always_gemm(true)`, 64 threads | 0.51 | 0.46 | 1.113 |

The 8- and 32-thread assign cells and the 32-thread train cells are in the full `otto h2h` output; they are printed,
not gated. At d=8 the default kernel is slower than `emeans` because it keeps the exact difference form (see the
limit above); `always_gemm(true)` is the speed-parity path.

Release over release, the same `otto h2h` ratios (ours / emeans). v0.1.0 is the first run, on the `v0.1.0` release
(`5f0b62d`, 2026-11-13), which had no `always_gemm` option. v0.2.0 is the run recorded at that release (2026-11-14).
v0.2.1 is the table above. Results are bitwise the same from v0.2.0 to v0.2.1.

| cell | v0.1.0 | v0.2.0 | v0.2.1 |
|---|---|---|---|
| train d=32, 1 thread | 1.648 | 1.050 | 1.050 |
| train d=32, 8 threads | 2.031 | 0.985 | 0.931 |
| train d=32, 64 threads | 1.554 | 0.926 | 0.772 |
| assign d=32, 1 thread | 1.808 | 1.074 | 1.024 |
| assign d=32, 64 threads | 2.900 | 1.194 | 1.128 |
| train d=8, default, 1 thread | 4.408 | 1.593 | 1.541 |
| train d=8, default, 8 threads | 5.844 | 1.348 | 1.433 |
| train d=8, default, 64 threads | 2.243 | 0.947 | 0.990 |
| assign d=8, default, 1 thread | 4.442 | 1.922 | 1.809 |
| assign d=8, default, 64 threads | 4.549 | 1.644 | 1.642 |
| train d=8, `always_gemm(true)`, 1 thread | n/a | 1.152 | 1.123 |
| train d=8, `always_gemm(true)`, 8 threads | n/a | 1.045 | 1.111 |
| train d=8, `always_gemm(true)`, 64 threads | n/a | 0.830 | 0.833 |
| assign d=8, `always_gemm(true)`, 1 thread | n/a | 1.099 | 1.072 |
| assign d=8, `always_gemm(true)`, 64 threads | n/a | 1.181 | 1.113 |

The 8-thread train cells swing run to run (1.32-1.67 over 20 runs of the d=8 default cell during the v0.2.0 work), so
read a difference of that size between two columns as noise.

## Development

`otto ci` runs lint, check, and test. `otto compat` regenerates the Faiss fixtures (needs Python and
`faiss-cpu==1.16.0`, pinned in `compat/`). `otto bench` runs the criterion benches.
