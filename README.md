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
e-means = { git = "https://github.com/scottidler/e-means", tag = "v0.1.0", version = "0.1.0" }
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

Measured on a 64-core machine, 2026-11-13, `cargo bench` (criterion medians). The `NumpyEmeans` reference times were
measured on the design author's data, not necessarily this mixture.

| bench | kernel | this crate | `NumpyEmeans` |
|---|---|---|---|
| train n=100000 d=32 k=256 T=5 (B=8192) | Gemm | 44.4 ms | 0.41 s |
| train n=100000 d=8 k=256 T=5 (B=1024) | Direct | 99.9 ms | 4.35 s |
| assign 100000 rows d=8 k=256 | Direct | 2.22 ms | |
| assign 100000 rows d=8 k=256 | Gemm | 1.21 ms | |
| assign 100000 rows d=32 k=256 | Direct | 7.46 ms | |
| assign 100000 rows d=32 k=256 | Gemm | 2.01 ms | |

## Development

`otto ci` runs lint, check, and test. `otto compat` regenerates the Faiss fixtures (needs Python and
`faiss-cpu==1.16.0`, pinned in `compat/`). `otto bench` runs the criterion benches.
