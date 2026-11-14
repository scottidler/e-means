//! End-to-end parity with Faiss `NumpyEmeans` (`faiss-cpu==1.16.0`), on fixtures exported by
//! `compat/generate.py` into `tests/fixtures/`.
//!
//! Each fixture replays a Faiss run through the caller-driven `Trainer`: the same initial centroids, then
//! every pass in the same row order, one `accumulate` + `update` per logical batch. The two sides share
//! the algorithm, not the RNG (deviation D1), so the init and orders are inputs, in one of two forms:
//!
//! - `faiss`: the init and orders `NumpyEmeans.train` drew, stored as files (the n=4000 fixtures).
//! - `splitmix`: rows, init and orders regenerated here from an integer-only recipe that
//!   `compat/generate.py` mirrors (SplitMix64, Fisher-Yates, Irwin-Hall grid noise), so the n=100000
//!   fixtures commit no input file. The manifest's FNV-1a hashes prove both sides drew the same thing.
//!
//! Metrics, as in the design's noise-floor table: centroid = worst per-centroid
//! `||c_rust - c_faiss||_2 / ||c_faiss||_2`; objective = `|obj_rust - obj_faiss| / obj_faiss` on the final
//! pass; labels = share of rows with equal last-pass labels. Only the separated class (k = blob count) is
//! compared tightly; on the others Faiss does not hold its own centroids under a kernel change.

use std::fs;
use std::path::{Path, PathBuf};

use e_means::EMeans;
use serde::Deserialize;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Separated fixtures: Faiss moves by at most 1.3e-7 on this metric under f64 sums, 0 under a forced BLAS
/// kernel.
const SEPARATED_CENTROID_BOUND: f64 = 1e-5;
const SEPARATED_OBJECTIVE_BOUND: f64 = 1e-6;
const LOOSE_OBJECTIVE_BOUND: f64 = 1e-3;
/// Revival smoke test: donor choice draws from a different RNG, and Faiss's own final-pass objective moves
/// 5.4% across seeds on this data.
const REVIVAL_OBJECTIVE_BOUND: f64 = 0.10;

#[derive(Debug, Deserialize)]
struct Manifest {
    faiss: String,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    class: String,
    n: usize,
    d: usize,
    k: usize,
    epochs: f64,
    spherical: bool,
    unused_threshold: f32,
    batch_size: usize,
    alpha0: f64,
    epoch_rows: Vec<usize>,
    total_steps: usize,
    objectives: Vec<f64>,
    splits: Vec<usize>,
    #[serde(flatten)]
    draw: Draw,
    x: Rows,
    x_fnv1a: String,
    init_fnv1a: String,
    orders_fnv1a: String,
    centroids: Array,
    labels: Array,
}

/// Where the init and pass orders come from.
#[derive(Debug, Deserialize)]
#[serde(tag = "draw", rename_all = "lowercase")]
enum Draw {
    /// Drawn by Faiss's `train` and stored.
    Faiss { init: Array, orders: Array },
    /// Regenerated: init = the first k of `permutation(n, stream_seed + INIT_STREAM)`, pass p's order = the
    /// first `epoch_rows[p]` of `permutation(n, stream_seed + PASS_STREAM + p)`.
    Splitmix { stream_seed: u64 },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Rows {
    File(Array),
    Grid(GridBlobs),
}

/// `grid_blobs` in `compat/generate.py`.
#[derive(Debug, Deserialize)]
struct GridBlobs {
    generator: String,
    n: usize,
    d: usize,
    blobs: usize,
    center_scale: i64,
    seed: u64,
}

/// A flat little-endian file.
#[derive(Debug, Deserialize)]
struct Array {
    file: String,
    dtype: String,
}

const INIT_STREAM: u64 = 1;
const PASS_STREAM: u64 = 2;
const GRID_STEP: f32 = 1.0 / 1024.0;
const IRWIN_HALL_TERMS: usize = 12;
const IRWIN_HALL_CENTER: i64 = 6138;

/// SplitMix64, the generator `compat/generate.py` mirrors.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sum of 12 ten-bit uniforms, centered: about N(0, 1) in steps of 2^-10.
    fn irwin_hall(&mut self) -> i64 {
        (0..IRWIN_HALL_TERMS).map(|_| (self.next() >> 54) as i64).sum::<i64>() - IRWIN_HALL_CENTER
    }
}

/// Fisher-Yates from the top: for i = n-1 down to 1, swap i with `next() % (i + 1)`.
fn permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = SplitMix(seed);
    let mut ids: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    ids
}

/// Row i = center of blob `i % blobs` + noise, on the 2^-10 grid. `blobs * d` center draws come first.
fn grid_blobs(spec: &GridBlobs) -> Vec<f32> {
    assert_eq!(spec.generator, "grid_blobs", "unknown row generator");
    let mut rng = SplitMix(spec.seed);
    let centers: Vec<i64> = (0..spec.blobs * spec.d)
        .map(|_| spec.center_scale * rng.irwin_hall())
        .collect();
    let mut x = Vec::with_capacity(spec.n * spec.d);
    for i in 0..spec.n {
        let blob = i % spec.blobs;
        for j in 0..spec.d {
            let step = centers[blob * spec.d + j] + rng.irwin_hall();
            assert!(step.abs() < 1 << 24, "grid step {step} is not exact in f32");
            x.push(step as f32 * GRID_STEP);
        }
    }
    x
}

/// FNV-1a 64 over the little-endian bytes, as hex; `compat/generate.py` writes the same.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> String {
    let hash = bytes.into_iter().fold(0xCBF2_9CE4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    });
    format!("{hash:016x}")
}

fn fnv1a_f32(values: &[f32]) -> String {
    fnv1a(values.iter().flat_map(|v| v.to_le_bytes()))
}

fn fnv1a_u32(values: &[u32]) -> String {
    fnv1a(values.iter().flat_map(|v| v.to_le_bytes()))
}

/// What one side produced.
#[derive(Debug)]
struct Run {
    centroids: Vec<f32>,
    /// Last-pass labels in row order.
    labels: Vec<u32>,
    objectives: Vec<f64>,
    splits: Vec<usize>,
}

#[derive(Debug)]
struct Metrics {
    centroid: f64,
    objective: f64,
    labels: f64,
    splits: usize,
}

fn manifest() -> Manifest {
    let path = Path::new(FIXTURES).join("manifest.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn fixture(name: &str) -> Fixture {
    manifest()
        .fixtures
        .into_iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("fixture {name} is not in the manifest"))
}

fn bytes(array: &Array, width: usize, values: usize) -> Vec<u8> {
    let path: PathBuf = Path::new(FIXTURES).join(&array.file);
    let raw = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        raw.len(),
        width * values,
        "{} has {} bytes, want {values} values of {width}",
        path.display(),
        raw.len()
    );
    raw
}

fn f32s(array: &Array, values: usize) -> Vec<f32> {
    match array.dtype.as_str() {
        "f32" => bytes(array, 4, values)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect(),
        other => panic!("{}: dtype {other} is not a float array", array.file),
    }
}

fn u32s(array: &Array, values: usize) -> Vec<u32> {
    assert_eq!(array.dtype, "u32", "{}: want u32", array.file);
    bytes(array, 4, values)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect()
}

/// What Faiss produced.
fn faiss_run(fx: &Fixture) -> Run {
    Run {
        centroids: f32s(&fx.centroids, fx.k * fx.d),
        labels: u32s(&fx.labels, fx.n),
        objectives: fx.objectives.clone(),
        splits: fx.splits.clone(),
    }
}

/// Rows, initial centroids and the concatenated pass orders of the Faiss run, checked against the
/// manifest's hashes.
fn inputs(fx: &Fixture) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let x = match &fx.x {
        Rows::File(array) => f32s(array, fx.n * fx.d),
        Rows::Grid(spec) => {
            assert_eq!((spec.n, spec.d), (fx.n, fx.d), "{}: generator shape", fx.name);
            grid_blobs(spec)
        }
    };
    let pass_rows: usize = fx.epoch_rows.iter().sum();
    let (init, orders) = match &fx.draw {
        Draw::Faiss { init, orders } => (f32s(init, fx.k * fx.d), u32s(orders, pass_rows)),
        Draw::Splitmix { stream_seed } => {
            let init_ids = permutation(fx.n, stream_seed + INIT_STREAM);
            let init = init_ids[..fx.k]
                .iter()
                .flat_map(|&i| x[i as usize * fx.d..(i as usize + 1) * fx.d].iter().copied())
                .collect();
            let orders = fx
                .epoch_rows
                .iter()
                .enumerate()
                .flat_map(|(p, &rows)| permutation(fx.n, stream_seed + PASS_STREAM + p as u64)[..rows].to_vec())
                .collect();
            (init, orders)
        }
    };
    assert_eq!(
        fnv1a_f32(&x),
        fx.x_fnv1a,
        "{}: rows differ from the ones Faiss trained on",
        fx.name
    );
    assert_eq!(
        fnv1a_f32(&init),
        fx.init_fnv1a,
        "{}: init differs from Faiss's",
        fx.name
    );
    assert_eq!(
        fnv1a_u32(&orders),
        fx.orders_fnv1a,
        "{}: pass orders differ from Faiss's",
        fx.name
    );
    (x, init, orders)
}

/// Replay the Faiss run through the caller-driven trainer: Faiss's init, Faiss's pass orders, one
/// `accumulate` per logical batch.
fn rust_run(fx: &Fixture) -> Run {
    let (x, init, orders) = inputs(fx);
    rust_run_chunked(fx, &x, &init, &orders, &[usize::MAX])
}

/// `rust_run` on prepared inputs, each logical batch fed in chunks whose row counts cycle through `chunks`
/// (`[usize::MAX]` = the whole batch in one chunk), on the current `rayon` pool.
fn rust_run_chunked(fx: &Fixture, x: &[f32], init: &[f32], orders: &[u32], chunks: &[usize]) -> Run {
    let mut trainer = EMeans::new(fx.d, fx.k)
        .epochs(fx.epochs)
        .unused_threshold(fx.unused_threshold)
        .spherical(fx.spherical)
        .begin(fx.n, init)
        .unwrap_or_else(|e| panic!("{}: begin: {e}", fx.name));

    let recipe = trainer.recipe();
    assert_eq!(recipe.batch_size, fx.batch_size, "{}: batch size", fx.name);
    assert_eq!(recipe.alpha0, fx.alpha0, "{}: alpha0", fx.name);
    assert_eq!(recipe.epoch_rows, fx.epoch_rows, "{}: epoch rows", fx.name);
    assert_eq!(recipe.total_steps, fx.total_steps, "{}: total steps", fx.name);

    let batch = trainer.batch_rows();
    let mut rows = Vec::with_capacity(batch * fx.d);
    let mut start = 0;
    let mut last_order: &[u32] = &[];
    for &count in &fx.epoch_rows {
        let order = &orders[start..start + count];
        for ids in order.chunks(batch) {
            rows.clear();
            for &i in ids {
                let i = i as usize;
                rows.extend_from_slice(&x[i * fx.d..(i + 1) * fx.d]);
            }
            let mut rest = &rows[..];
            let mut sizes = chunks.iter().cycle();
            while !rest.is_empty() {
                let size = sizes.next().expect("chunks is not empty");
                let (chunk, tail) = rest.split_at((*size).min(rest.len() / fx.d) * fx.d);
                trainer
                    .accumulate(chunk)
                    .unwrap_or_else(|e| panic!("{}: accumulate: {e}", fx.name));
                rest = tail;
            }
            trainer.update().unwrap_or_else(|e| panic!("{}: update: {e}", fx.name));
        }
        trainer
            .end_epoch()
            .unwrap_or_else(|e| panic!("{}: end_epoch: {e}", fx.name));
        start += count;
        last_order = order;
    }
    let model = trainer
        .finalize()
        .unwrap_or_else(|e| panic!("{}: finalize: {e}", fx.name));

    // Feed-order labels back onto rows.
    let fed = model
        .labels()
        .unwrap_or_else(|| panic!("{}: no labels captured", fx.name));
    assert_eq!(fed.len(), fx.n, "{}: the last pass labels every row", fx.name);
    let mut labels = vec![u32::MAX; fx.n];
    for (&row, label) in last_order.iter().zip(fed) {
        labels[row as usize] = label.unwrap_or_else(|| panic!("{}: feed-order label is None", fx.name));
    }
    Run {
        centroids: model.centroids().to_vec(),
        labels,
        objectives: model.epoch_stats().iter().map(|s| s.objective).collect(),
        splits: model.epoch_stats().iter().map(|s| s.splits).collect(),
    }
}

fn metrics(fx: &Fixture, rust: &Run, faiss: &Run) -> Metrics {
    let centroid = rust
        .centroids
        .chunks_exact(fx.d)
        .zip(faiss.centroids.chunks_exact(fx.d))
        .map(|(a, b)| {
            let diff: f64 = a
                .iter()
                .zip(b)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            let norm: f64 = b.iter().map(|&b| f64::from(b).powi(2)).sum();
            (diff / norm).sqrt()
        })
        .fold(0.0, f64::max);
    let last = |run: &Run| *run.objectives.last().expect("at least one pass");
    let objective = (last(rust) - last(faiss)).abs() / last(faiss);
    let equal = rust.labels.iter().zip(&faiss.labels).filter(|(a, b)| a == b).count();
    Metrics {
        centroid,
        objective,
        labels: equal as f64 / fx.n as f64,
        splits: rust.splits.iter().sum(),
    }
}

/// Replay `name`, print its metrics, and return them with the fixture.
fn compare(name: &str, class: &str) -> (Fixture, Metrics) {
    let fx = fixture(name);
    assert_eq!(fx.class, class, "{name}: fixture class");
    let faiss = faiss_run(&fx);
    let rust = rust_run(&fx);
    let m = metrics(&fx, &rust, &faiss);
    println!(
        "{name}: centroid {:.3e} objective {:.3e} labels {:.4}% splits rust {:?} faiss {:?} objectives rust {:?} faiss {:?}",
        m.centroid,
        m.objective,
        100.0 * m.labels,
        rust.splits,
        faiss.splits,
        rust.objectives,
        faiss.objectives
    );
    (fx, m)
}

fn check_separated(name: &str) {
    let (fx, m) = compare(name, "separated");
    assert!(
        m.centroid <= SEPARATED_CENTROID_BOUND,
        "{name}: centroid drift {:e} > {SEPARATED_CENTROID_BOUND:e}",
        m.centroid
    );
    assert!(
        m.objective <= SEPARATED_OBJECTIVE_BOUND,
        "{name}: final-pass objective off by {:e} > {SEPARATED_OBJECTIVE_BOUND:e}",
        m.objective
    );
    assert_eq!(m.labels, 1.0, "{name}: last-pass labels differ on some row");
    assert_eq!(fx.splits.iter().sum::<usize>(), 0, "{name}: revival is off");
}

fn check_loose(name: &str, min_labels: f64) {
    let (_, m) = compare(name, "loose");
    assert!(
        m.objective <= LOOSE_OBJECTIVE_BOUND,
        "{name}: final-pass objective off by {:e} > {LOOSE_OBJECTIVE_BOUND:e}",
        m.objective
    );
    assert!(
        m.labels >= min_labels,
        "{name}: label agreement {} < {min_labels}",
        m.labels
    );
}

#[test]
fn manifest_names_every_fixture() {
    let manifest = manifest();
    assert_eq!(manifest.faiss, "1.16.0");
    let names: Vec<&str> = manifest.fixtures.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "separated_d8",
            "separated_d32",
            "separated_d8_t2_5",
            "separated_spherical_d32",
            "separated_d2",
            "oversplit",
            "large",
            "large_revival",
        ]
    );
}

#[test]
fn separated_d8_quality_law() {
    check_separated("separated_d8");
}

#[test]
fn separated_d32_throughput_law() {
    check_separated("separated_d32");
}

#[test]
fn separated_d8_fractional_epochs() {
    check_separated("separated_d8_t2_5");
}

#[test]
fn separated_spherical_d32() {
    check_separated("separated_spherical_d32");
}

#[test]
fn separated_d2() {
    check_separated("separated_d2");
}

#[test]
fn oversplit_d32_k64() {
    check_loose("oversplit", 0.95);
}

#[test]
fn large_d32_k256() {
    check_loose("large", 0.99);
}

#[test]
fn large_d32_k256_revival_smoke() {
    let (_, m) = compare("large_revival", "loose");
    assert!(m.splits > 0, "large_revival: revival never split");
    assert!(
        m.objective <= REVIVAL_OBJECTIVE_BOUND,
        "large_revival: final-pass objective off by {:e} > {REVIVAL_OBJECTIVE_BOUND:e}",
        m.objective
    );
}

/// Bitwise view of a run, for equality across pools and chunkings.
fn run_bits(run: &Run) -> (Vec<u32>, &[u32], Vec<u64>, &[usize]) {
    (
        run.centroids.iter().map(|c| c.to_bits()).collect(),
        &run.labels,
        run.objectives.iter().map(|o| o.to_bits()).collect(),
        &run.splits,
    )
}

// Design criterion (Phase 8): a 1-thread and an 8-thread pool give bitwise equal centroids, labels and
// objective on every fixture; so do row-at-a-time chunks and chunks of 3, 5, 7 rows on the 8-thread pool.
#[test]
fn determinism_across_thread_counts_and_chunkings() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let eight = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("8-thread pool");
    let whole: &[usize] = &[usize::MAX];
    for fx in manifest().fixtures {
        let (x, init, orders) = inputs(&fx);
        let reference = one.install(|| rust_run_chunked(&fx, &x, &init, &orders, whole));
        let faiss = faiss_run(&fx);
        let m = metrics(&fx, &reference, &faiss);
        println!(
            "{}: 1 thread vs faiss: centroid {:.3e} objective {:.3e} labels {:.4}%",
            fx.name,
            m.centroid,
            m.objective,
            100.0 * m.labels
        );
        for chunks in [whole, &[1], &[3, 5, 7]] {
            let run = eight.install(|| rust_run_chunked(&fx, &x, &init, &orders, chunks));
            assert_eq!(
                run_bits(&run),
                run_bits(&reference),
                "{}: 8 threads, chunks {chunks:?} differ from 1 thread, whole batches",
                fx.name
            );
        }
    }
}
