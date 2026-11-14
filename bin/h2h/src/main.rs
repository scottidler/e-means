//! Head-to-head: `e-means` (by path) against `emeans` 0.1.1 (`5b283a9`), `train` and `assign`, d=32 and d=8,
//! 1 to 64 threads.
//!
//! Data: the `tests/quality.rs` mixture (n=100000, 256 equal-size components, centers ~ N(0, 3^2), unit
//! noise), k=256, T=5 for `train`; for `assign`, a T=5 model of each crate assigns the same rows (so d=32
//! assigns with the Gemm kernel and d=8 with Direct, as the recipe's `B * d` picks: the batch size depends
//! on the epoch budget, and a T=1 model would land d=32 on Direct).
//!
//! Method, per cell: one warm-up of each side, then 15 pairs, each pair running ours then emeans. The
//! cell's ratio is the median of the 15 per-pair ratios (ours / emeans). Both sides do equal work: neither
//! assign closure clones anything. Both crates run in one explicit rayon pool of the cell's thread count.
//! The 1-thread cells run in a child process pinned with `taskset -c 2`; core placement swings 1-thread
//! timings by ~30% on the reference machine.
//!
//! Every cell is printed. A cell with a bound is gated: the process exits 1 if any gated cell misses it.
//! `--cell NAME...` runs just those cells and exits on their bounds alone.

use std::hint::black_box;
use std::process::{Command, ExitCode};
use std::time::Instant;

use clap::Parser;

const K: usize = 256;
const N: usize = 100_000;
const CENTER_SCALE: f64 = 3.0;
const DATA_SEED: u64 = 10;
const EPOCHS: f64 = 5.0;
const PAIRS: usize = 15;
/// The core a 1-thread cell is pinned to.
const PIN_CORE: &str = "2";

/// Whether `e_means::EMeans::always_gemm` exists. Phase 8 adds it; until then the `always_gemm` cells print
/// `n/a` and are not gated. Flip this and wire the builder call in `ours_builder` in that phase.
const OURS_HAS_ALWAYS_GEMM: bool = false;

const THREADS: [usize; 4] = [1, 8, 32, 64];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Train,
    Assign,
}

/// One row of the Acceptance Criteria grid: a measurement and its bound at each of `THREADS`. `None` is
/// printed, not gated.
struct Row {
    label: &'static str,
    op: Op,
    d: usize,
    always_gemm: bool,
    bounds: [Option<f64>; 4],
}

/// The Acceptance Criteria grid of `docs/plan/2026-11-13-assign-performance.md`: ratio = ours / emeans,
/// median of 15 paired runs; columns are 1, 8, 32 and 64 threads. 8-thread assign and every 32-thread cell
/// are printed, not gated: no prototype measured them.
const GRID: [Row; 6] = [
    Row {
        label: "train-d32",
        op: Op::Train,
        d: 32,
        always_gemm: false,
        bounds: [Some(1.3), Some(1.3), None, Some(1.3)],
    },
    Row {
        label: "assign-d32",
        op: Op::Assign,
        d: 32,
        always_gemm: false,
        bounds: [Some(1.3), None, None, Some(1.6)],
    },
    Row {
        label: "train-d8-default",
        op: Op::Train,
        d: 8,
        always_gemm: false,
        bounds: [Some(1.75), Some(1.8), None, Some(1.3)],
    },
    Row {
        label: "assign-d8-default",
        op: Op::Assign,
        d: 8,
        always_gemm: false,
        bounds: [Some(2.1), None, None, Some(1.8)],
    },
    Row {
        label: "train-d8-gemm",
        op: Op::Train,
        d: 8,
        always_gemm: true,
        bounds: [Some(1.3), Some(1.3), None, Some(1.3)],
    },
    Row {
        label: "assign-d8-gemm",
        op: Op::Assign,
        d: 8,
        always_gemm: true,
        bounds: [Some(1.4), None, None, Some(1.6)],
    },
];

#[derive(Debug, Clone)]
struct Cell {
    name: String,
    op: Op,
    d: usize,
    always_gemm: bool,
    threads: usize,
    bound: Option<f64>,
}

impl Cell {
    /// Whether ours can run this cell yet.
    fn available(&self) -> bool {
        !self.always_gemm || OURS_HAS_ALWAYS_GEMM
    }
}

fn cells() -> Vec<Cell> {
    GRID.iter()
        .flat_map(|row| {
            THREADS.iter().zip(row.bounds).map(|(&threads, bound)| Cell {
                name: format!("{}-{threads}t", row.label),
                op: row.op,
                d: row.d,
                always_gemm: row.always_gemm,
                threads,
                bound,
            })
        })
        .collect()
}

#[derive(Parser)]
#[command(about = "Head-to-head of e-means against emeans 0.1.1; exits 1 if a gated cell misses its bound")]
struct Args {
    /// Run only these cells (space-separated names) and gate on their bounds alone.
    #[arg(long, num_args = 1..)]
    cell: Vec<String>,

    /// Internal: measure one cell in this process and print its RESULT line.
    #[arg(long, hide = true)]
    measure: Option<String>,
}

/// SplitMix64 and Box-Muller: the harness's own seeded stream, the generator of `tests/quality.rs` and
/// `benches/common/mod.rs`.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`, 53 bits.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `N` rows of `d` columns, row `i` from component `i % K`.
fn mixture(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let centers: Vec<f64> = (0..K * d).map(|_| CENTER_SCALE * rng.normal()).collect();
    let mut x = Vec::with_capacity(N * d);
    for i in 0..N {
        let c = i % K;
        for j in 0..d {
            x.push((centers[c * d + j] + rng.normal()) as f32);
        }
    }
    x
}

fn ours_builder(cell: &Cell) -> e_means::EMeans {
    let builder = e_means::EMeans::new(cell.d, K).epochs(EPOCHS).seed(DATA_SEED);
    // Phase 8: `if cell.always_gemm { builder.always_gemm(true) } else { builder }`.
    assert!(
        !cell.always_gemm || OURS_HAS_ALWAYS_GEMM,
        "{}: ours has no always_gemm yet",
        cell.name
    );
    builder
}

fn theirs_builder(cell: &Cell) -> emeans::Emeans {
    emeans::Emeans::new(cell.d, K).nepochs(EPOCHS).seed(DATA_SEED)
}

/// Both sides resolved the same batch size and step size, or the cell compares different runs.
fn assert_same_recipe(cell: &Cell, ours: &e_means::Model, theirs: &emeans::Model) {
    let (o, t) = (
        ours.recipe(),
        theirs.recipe().expect("emeans trained model has a recipe"),
    );
    assert_eq!(o.batch_size, t.batch_size, "{}: batch size differs", cell.name);
    assert_eq!(o.alpha0, t.alpha0, "{}: alpha0 differs", cell.name);
}

/// Median of a sample; the sample is sorted in place. Even lengths take the upper middle.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[derive(Debug, PartialEq)]
struct Measurement {
    ours_ms: f64,
    theirs_ms: f64,
    ratio: f64,
}

/// One warm-up of each, then `PAIRS` pairs, ours then emeans in each.
fn paired(mut ours: impl FnMut(), mut theirs: impl FnMut()) -> Measurement {
    ours();
    theirs();
    let (mut ours_ms, mut theirs_ms, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..PAIRS {
        let start = Instant::now();
        ours();
        let o = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        theirs();
        let t = start.elapsed().as_secs_f64() * 1e3;
        ours_ms.push(o);
        theirs_ms.push(t);
        ratios.push(o / t);
    }
    Measurement {
        ours_ms: median(&mut ours_ms),
        theirs_ms: median(&mut theirs_ms),
        ratio: median(&mut ratios),
    }
}

/// Measure one cell inside a pool of its thread count.
fn measure(cell: &Cell) -> Measurement {
    let x = mixture(cell.d, DATA_SEED);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cell.threads)
        .build()
        .expect("build the rayon pool");
    pool.install(|| match cell.op {
        Op::Train => {
            let (ours, theirs) = (ours_builder(cell), theirs_builder(cell));
            assert_same_recipe(
                cell,
                &ours.train(&x).expect("ours train"),
                &theirs.train(&x).expect("emeans train"),
            );
            paired(
                || {
                    black_box(ours.train(black_box(&x)).expect("ours train"));
                },
                || {
                    black_box(theirs.train(black_box(&x)).expect("emeans train"));
                },
            )
        }
        Op::Assign => {
            let ours = ours_builder(cell).train(&x).expect("ours train");
            let theirs = theirs_builder(cell).train(&x).expect("emeans train");
            assert_same_recipe(cell, &ours, &theirs);
            paired(
                || {
                    black_box(ours.assign(black_box(&x)).expect("ours assign"));
                },
                || {
                    black_box(theirs.assign(black_box(&x)).expect("emeans assign"));
                },
            )
        }
    })
}

/// Measure one cell in a child process (pinned for 1 thread) and parse its RESULT line.
fn measure_in_child(cell: &Cell) -> Result<Measurement, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let mut command = if cell.threads == 1 {
        let mut c = Command::new("taskset");
        c.args(["-c", PIN_CORE]).arg(&exe);
        c
    } else {
        Command::new(&exe)
    };
    let output = command
        .args(["--measure", &cell.name])
        .output()
        .map_err(|e| format!("spawn the {} child: {e}", cell.name))?;
    if !output.status.success() {
        return Err(format!(
            "{} child failed ({}): {}",
            cell.name,
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("RESULT "))
        .ok_or_else(|| format!("{} child printed no RESULT line: {stdout}", cell.name))?;
    parse_result(line).ok_or_else(|| format!("{}: unreadable RESULT line {line:?}", cell.name))
}

fn parse_result(line: &str) -> Option<Measurement> {
    let mut fields = line.split_whitespace().map(|f| f.parse::<f64>());
    let ours_ms = fields.next()?.ok()?;
    let theirs_ms = fields.next()?.ok()?;
    let ratio = fields.next()?.ok()?;
    fields.next().is_none().then_some(Measurement {
        ours_ms,
        theirs_ms,
        ratio,
    })
}

/// How a cell's ratio stands against its bound.
#[derive(Debug, PartialEq)]
enum Verdict {
    Pass,
    Miss,
    Printed,
}

fn verdict(ratio: f64, bound: Option<f64>) -> Verdict {
    match bound {
        None => Verdict::Printed,
        Some(bound) if ratio <= bound => Verdict::Pass,
        Some(_) => Verdict::Miss,
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let all = cells();
    if let Some(name) = args.measure {
        let Some(cell) = all.iter().find(|c| c.name == name) else {
            eprintln!("h2h: no cell named {name}");
            return ExitCode::from(2);
        };
        let m = measure(cell);
        println!("RESULT {} {} {}", m.ours_ms, m.theirs_ms, m.ratio);
        return ExitCode::SUCCESS;
    }

    let mut chosen = Vec::new();
    if args.cell.is_empty() {
        chosen.extend(all.iter());
    }
    for name in &args.cell {
        match all.iter().find(|c| &c.name == name) {
            Some(cell) => chosen.push(cell),
            None => {
                let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
                eprintln!("h2h: no cell named {name}; cells: {}", names.join(" "));
                return ExitCode::from(2);
            }
        }
    }

    println!(
        "{:<24} {:>10} {:>10} {:>7} {:>7}  verdict",
        "cell", "ours ms", "emeans ms", "ratio", "bound"
    );
    let (mut misses, mut unavailable_named) = (0, 0);
    for cell in chosen {
        let bound = cell.bound.map_or("-".to_string(), |b| format!("<={b}"));
        if !cell.available() {
            println!(
                "{:<24} {:>10} {:>10} {:>7} {:>7}  n/a (ours has no always_gemm yet)",
                cell.name, "n/a", "n/a", "n/a", bound
            );
            if !args.cell.is_empty() {
                unavailable_named += 1;
            }
            continue;
        }
        let m = match measure_in_child(cell) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("h2h: {e}");
                return ExitCode::from(2);
            }
        };
        let v = verdict(m.ratio, cell.bound);
        if v == Verdict::Miss {
            misses += 1;
        }
        println!(
            "{:<24} {:>10.2} {:>10.2} {:>7.3} {:>7}  {}",
            cell.name,
            m.ours_ms,
            m.theirs_ms,
            m.ratio,
            bound,
            match v {
                Verdict::Pass => "pass",
                Verdict::Miss => "MISS",
                Verdict::Printed => "printed",
            }
        );
    }
    if unavailable_named > 0 {
        eprintln!("h2h: {unavailable_named} named cell(s) cannot run yet, so none can be gated");
        return ExitCode::FAILURE;
    }
    if misses > 0 {
        eprintln!("h2h: {misses} gated cell(s) missed their bound");
        return ExitCode::FAILURE;
    }
    println!("h2h: every gated cell is within its bound");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_is_every_row_at_every_thread_count() {
        let all = cells();
        assert_eq!(all.len(), GRID.len() * THREADS.len());
        assert!(
            all.iter()
                .any(|c| c.name == "train-d8-default-8t" && c.bound == Some(1.8))
        );
        assert!(all.iter().any(|c| c.name == "assign-d32-8t" && c.bound.is_none()));
        let mut names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len(), "cell names are unique");
    }

    #[test]
    fn thirty_two_thread_cells_are_never_gated() {
        assert!(cells().iter().filter(|c| c.threads == 32).all(|c| c.bound.is_none()));
    }

    #[test]
    fn always_gemm_cells_are_unavailable_until_ours_has_the_option() {
        for cell in cells() {
            assert_eq!(
                cell.available(),
                !cell.always_gemm || OURS_HAS_ALWAYS_GEMM,
                "{}",
                cell.name
            );
        }
    }

    #[test]
    fn a_ratio_is_judged_against_its_bound() {
        assert_eq!(verdict(1.3, Some(1.3)), Verdict::Pass);
        assert_eq!(verdict(1.31, Some(1.3)), Verdict::Miss);
        assert_eq!(verdict(9.0, None), Verdict::Printed);
    }

    #[test]
    fn median_takes_the_middle_of_an_odd_sample() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [5.0]), 5.0);
    }

    #[test]
    fn result_lines_parse_exactly_three_numbers() {
        assert_eq!(
            parse_result("12.5 6.25 2"),
            Some(Measurement {
                ours_ms: 12.5,
                theirs_ms: 6.25,
                ratio: 2.0
            })
        );
        assert_eq!(parse_result("12.5 6.25"), None);
        assert_eq!(parse_result("1 2 3 4"), None);
        assert_eq!(parse_result("1 two 3"), None);
    }
}
