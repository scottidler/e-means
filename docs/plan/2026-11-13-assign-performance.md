# Design Document: assignment and training performance

**Author:** Scott Idler
**Date:** 2026-11-13
**Status:** Implemented
**Review Passes Completed:** 5/5 (passes 3 and 4 ran as one rewrite on the Phase 0 results; pass 5 after panel round 1; re-run after S5). Panel rounds 1 and 2 folded.

## Summary

`e-means` trains 1.7x-4.9x slower than the `emeans` 0.1.1 crate at k=256, and 10x slower with 14x the memory at k=65536. Five measured causes, all in the assignment hot path and the serial code around it. This plan fixes them without moving a result bit: same labels, distances, centroids and objective, for any thread count and chunking; Faiss parity unchanged; still exactly one `unsafe`. Phase 0 already ran five spikes. Every claim it tested held bitwise. Large k drops to 0.63 s train time and 158 MB (emeans 0.43 s, 158 MB). d=8 at 8 threads goes from ~2x behind emeans to ~1.5x (default kernel, measured in `bin/h2h`, Phase 7; the S5 prototype's 1.0-1.2x was measured in a harness that charged emeans a thread handoff `bin/h2h` does not) and 0.8-1.1x (`always_gemm`). Below 8 threads the default d=8 kernel stays ~1.6x behind at 1 thread; Scott chose (2026-11-13) a caller opt-in, `EMeans::always_gemm(true)`, for that gap.

## Problem Statement

### Background

- `e-means` shipped per `docs/plan/2026-11-13-e-means.md`; released as `v0.1.0` at `5f0b62d`. That doc's "Reduction order", "Kernel choice" and "Trainer error contract" are the contracts this plan keeps.
- Head-to-head against `emeans` 0.1.1 (`~/repos/savannahar68/emeans` at `5b283a9`), 2026-11-13, recorded in `docs/handoff/emeans-performance-parity.md`.
- Machine: Threadripper 9970X, 64 threads, rustc 1.99.1, release. Load average 2-10 during runs: read the numbers as ratios.
- Data: the `tests/quality.rs` mixture, n=100000, k=256, T=5. Both crates resolve the same recipe:
  - d=32 -> B=8192, alpha0=1 (throughput law), `B*d` = 262144 -> ours runs Gemm.
  - d=8 -> B=1024, alpha0=0.155434 (quality law), `B*d` = 8192 -> ours runs Direct.

Baseline, `train` wall time, median of 15 after one warm-up (`ours` = `bce3580`; the hot path is unchanged through `5f0b62d`):

| threads | d=32 ours | d=32 emeans | d=8 ours | d=8 emeans |
|---|---|---|---|---|
| 64 | 41.2 ms | 24.4 ms | 97.6 ms | 43.9 ms |
| 32 | 33.4 ms | 16.0 ms | 100.8 ms | 32.7 ms |
| 8 | 38.6 ms | 18.9 ms | 80.9 ms | 16.7 ms |
| 1 | 151.9 ms | 84.5 ms | 255.5 ms | 52.4 ms |

Baseline, `Model::assign`, 100000 rows:

| | d=32 ours | d=32 emeans | d=8 ours | d=8 emeans |
|---|---|---|---|---|
| 1 thread | 27.4 ms | 14.8 ms | 44.1 ms | 8.4 ms |
| 64 threads | 1.8 ms | 0.7 ms | 2.2 ms | 0.5 ms |

Baseline, large k (n=200000, d=32, k=65536, T=1):

| | ours | emeans |
|---|---|---|
| 64 threads, wall | 4.26 s | 0.43 s |
| 64 threads, peak RSS | 2.20 GB | 158 MB |
| 1 thread, wall | 25.3 s | 7.4 s |

- Thin LTO + one codegen unit (the `emeans` release profile) does not change the ratios: 42.2 vs 25.4 ms (d=32), 103.4 vs 45.4 ms (d=8), 64 threads.
- 1-thread timings swing up to ~30% with core placement on this machine (Phase 0, S2). Every 1-thread number in this doc from here on is `taskset -c 2`.

### Problem

Root causes, measured on instrumented copies (`perf` is blocked: `perf_event_paranoid=4`):

1. **Scalar argmin** in `gemm_nearest` (`src/assign.rs:94-125`). A 16-lane scan: d=32, 1 thread, train 166 -> 97 ms, assign 29.9 -> 17.1 ms, bitwise identical.
2. **Scalar Direct kernel** (`src/assign.rs:179-221`): one centroid at a time, sequential f32 fold over d. d=8 is 4.9x behind emeans at 1 thread; d=16, k=32768 is 9.5x behind (6.36 vs 0.67 s).
3. **Serial sections around the parallel region.**
   - d=32, 64 threads (of 42.8 ms per train): gather 8.5 ms (`src/config.rs:251`), finite scan 4.4 ms (`src/trainer.rs:206`), row-order reduction 5.8 ms (`src/trainer.rs:498-522`), permutations 2.1 ms; parallel assign 21.1 ms.
   - d=8, 8 threads pinned, per train (S5): permutation 1.59 ms (emeans 0.64), gather 1.04 (emeans 0: fused into its tasks), finite scan 0.92 (emeans has none), calling-thread to pool handoff 1.5 + 1.9 ms.
4. **Coarse tasks.** B=1024 / `BLOCK_ROWS`=256 = 4 tasks per batch, so at 8 threads half the workers idle and each batch waits on one 256-row task (S5: 45.8 us per batch for lane Direct, emeans 15.4 us on 128-row tasks). This is most of the 8-thread gap: 14.5 of 17.5 ms.
5. **Unbounded tile.** `gemm_nearest` allocates `rows x k` f32 per block (`src/assign.rs:85`): 64 MB per block at k=65536, ~32 live at once. Centroid norms recomputed per block (`src/assign.rs:93`).

### Goals

- Close the speed gap to `emeans` 0.1.1 for `train` and `Model::assign`, both laws, 1 to 64 threads. (Scott, handoff "Next action", 2026-11-13.)
- Bound assignment scratch memory independent of k; large-k peak RSS in the same order as `emeans`. (Same.)
- Keep every result bit. (Same, "Constraints".)

Constraints, each with the test that catches a break:

- Bitwise determinism across thread counts and chunkings: `cargo test --test parity determinism`.
- Faiss parity bounds: `cargo test --test parity` (8 fixtures).
- Direct kernel below `BLAS_THRESHOLD` by default: why ours matches Faiss on `oversplit` and survives offset data (`common_offset_direct_is_right_and_gemm_cancels`).
- Exactly one `unsafe` in `src/`: `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/` prints `src/assign.rs:1`. (`rg -l` lists files, so a second block in `assign.rs` would pass it.) Rules out `#[target_feature]` multiversioning (calling one is `unsafe`); everything here autovectorizes on the baseline target.
- NaN/inf rejected before any state change: `non_finite_row_is_rejected_before_anything_changes`, `train_errors_are_typed`.
- No BLAS link features, no `build.rs`.

### Non-Goals

Excluded:

- Any change to a result bit, including the Gemm L2 formula. emeans scans `||c||^2 - 2 x.c` and adds `||x||^2` after the argmin (`knn.rs:172,267`); that moves near-tie labels and breaks the offset test.
- SIMD intrinsics or `#[target_feature]` dispatch. Measured: AVX-512/AVX2 multiversioning on the lane scan gained nothing (108 vs 97 ms), cost 2 `unsafe` sites.
- FMA (`mul_add`) anywhere. Rust does not contract; that keeps every opt level bitwise equal.
- Changing the RNG. ChaCha8 permutations cost ~1 ms more per train than emeans' xoshiro (S5); change 8 overlaps them instead.
- emeans' accuracy tradeoff as a default: always-Gemm. (An earlier draft also listed f32 sums; emeans' batch sums are f64, `emeans.rs:414-415,546-548`.)

Parked, with the condition that reopens each:

- `Model::from_centroids` (assign with saved centroids). Surfaced by the comparison, not asked for. Reopen when Scott or a consumer needs it; it brings its own calls on `Model::recipe`, kernel choice, spherical normalization.
- `EMeans::recipe(n)` builder preview. `Recipe::resolve(&RecipeSpec)` is already public (`src/recipe.rs:67-75,179`). Reopen with `from_centroids`.
- Optional `parallel` cargo feature (emeans: `default = ["parallel"]`). Reopen when a consumer needs a no-rayon build.
- The lane Direct kernel's per-row cost (1.44-1.56x emeans' summed kernel time at any thread count, S2 and S5; 2.5x at d=16, k=32768). It is the difference form with no FMA and no `target_feature`, which the constraints fix. Reopen if a consumer runs Direct at large k; row register blocking was tried in S2 and gained nothing.

## Proposed Solution

### Overview

Eight changes, each argued bit-neutral under API Design and tested against an oracle: today's scalar code kept under `#[cfg(test)]`, compared with `to_bits`.

| # | change | root cause # | why the bits hold | Phase 0 |
|---|---|---|---|---|
| 1 | lane argmin in `gemm_nearest`, L2 and IP | 1 | same per-element score; lowest index wins ties by construction | bitwise, S3 |
| 2 | tile over centroids under `TILE_FLOATS`, per-task scratch | 5 | sgemm element = f(row, centroid, d, CPU) only; strict `<` across tiles in ascending order | S1: 0 of 360 cases differ; S3 bitwise |
| 3 | centroid norms once per call, not per block | 1, 5 | same `inner_product(c, c)` | S3 bitwise |
| 4 | one fused parallel region per batch in `train()`: gather, finite scan and assign inside each task; scan fused into `Model::assign`'s tasks | 3 | same rows, same order; same first offending index (minimum across tasks) | S5 (train path): bitwise, error text identical at 1 and 8 threads. `Model::assign`'s fused scan not prototyped: Phase 4 tests are its evidence |
| 5 | reduction partitioned by cluster | 3 | each cluster's f64 adds stay in row order | not run |
| 6 | lane-vectorized Direct, layout tiled over centroids under `TILE_FLOATS`, built per task | 2 | lane = centroid; each lane runs today's exact coordinate fold; tiles merge by change 2's strict `<` | S2: bitwise, 122 unit + parity 10/10 (untiled layout) |
| 7 | 64-row tasks inside a 256-row block when the call has fewer blocks than threads | 4 | per-row results do not depend on task composition; cuts and reduction stay at 256 | S4, S5: bitwise |
| 8 | `train()`'s loop runs on a pool thread; the next pass's permutation is drawn alongside the current pass | 3 | same ops on a different thread; each pass has its own RNG stream | S5: bitwise |

### Architecture

Unchanged module layout. Touched: `src/assign.rs`, `src/trainer.rs`, `src/config.rs`, `src/model.rs`. New: `examples/large-k.rs`, `bin/h2h/` (standalone tool crate, not part of the package).

`assign.rs` after:

```
nearest(kernel, metric, rows, centroids, norms: &[f32], d, labels, scores, scratch: &mut Vec<f32>)
  Direct -> for each centroid tile [j0, j1), width = LANES * max(1, TILE_FLOATS / (LANES * d)), clamped to k:
              transpose centroids[j0..j1] into lane layout in scratch (per task)
              direct_lanes(rows, scratch, j0) merged into each row's running best
            tail centroids (k mod LANES) by today's scalar fold, after the last full group
  Gemm   -> for each centroid tile [j0, j1), width (TILE_FLOATS / rows).clamp(1, k):
              inner_products(rows, centroids[j0*d..j1*d], d, scratch)   // the one unsafe, unchanged
              lane_argmin(scratch, norms[j0..j1], j0) merged into each row's running best
```

`train()` after (changes 4, 7, 8):

```
on a pool thread (rayon::scope), for each pass p:
  rayon::join(
    || for each batch of order_p:
         trainer.accumulate_indexed(x, ids)     // crate-private, whole batches only
           L2 only: copy snapshot from centroids // as today (trainer.rs:217); spherical keeps the readout from refresh_readout
           one par region over tasks (256-row blocks, or 64-row tasks per change 7):
             gather rows ids[task] into the task's slice of the batch buffer
             scan the slice for NaN/inf -> record first offending offset, skip assign
             nearest(...) into the task's label/score slices
           after the join: any offending offset -> Err(NonFiniteInput) with the minimum, nothing changed
           row-order reduction (change 5), unchanged order
         trainer.update()
    || if p + 1 < passes: order_{p+1} = Stream::new(seed, Pass(p+1)).permutation(n)   // none after the last pass
  )
  trainer.end_epoch()
after the loop: scatter_labels(order_last, n)   // the last pass's order, as today
```

- `norms`: centroid squared norms, computed once per top-level call (one `assign_blocks` / `accumulate_indexed` call, one `Model::assign` call), Gemm L2 only; empty otherwise. Never stored on `Model` or `Trainer`, so no derived field can drift from `centroids` / `snapshot`.
- Lane layout: built inside each rayon task, one centroid tile at a time, so it never exceeds `max(TILE_FLOATS, LANES*d)` f32 whatever k is. S2 measured one layout built on the calling thread as slower at 64 threads (2.59 vs 1.91 s at d=16, k=32768): the kernel runs slower reading memory another core wrote. Transpose cost per task O(k*d) against O(rows*k*d) of distance work.
- Snapshot: tasks read the per-batch `snapshot` copy, never `centroids`. S5 measured searching `centroids` directly: `apply` rose from 0.87 to 4.1 ms per train at 8 threads, its writes hitting cache lines the workers held shared.
- Scratch: one `Vec<f32>` per rayon leaf job via `for_each_init(Vec::new, ..)`. `for_each_init` calls init once per leaf job, not once per worker (rayon `iter/mod.rs:412`): live scratch is bounded by the thread count, allocations by the job count (one per task, as today's `vec!` tile). No `thread_local!`, no shared state.
- One parallel region per batch, as today (490 per d=8 train, the same count as emeans). S5 measured separate gather and scan regions: 1470 regions per train, lane d=8 39.3 -> 57.7 ms at 8 threads and 48.2 -> 206.9 ms at 64 threads. Never add a region per batch without a bench that shows it pays.
- The caller-driven `Trainer::accumulate(&[f32])` keeps today's path: serial finite scan over its chunk, carry, `assign_blocks`. It gets changes 1-3, 5, 6, 7 through `assign_blocks`.

### Data Model

- `TILE_FLOATS: usize = 1 << 18` in `assign.rs` (1 MB of f32 per task; Phase 3 re-measure chose 2^18, see Phase 3). S3 cap sweep at k=65536: 2^16 -> 0.63 s, 103 MB (64t), 8.32 s (1t); 2^18 -> 0.63 s, 158 MB, 10.07 s; 2^20 -> 0.78 s, 355 MB. Single runs; Phase 3 re-measures 2^16 against 2^18 and keeps the faster.
- `LANES: usize = 16` in `assign.rs`, one const for the Gemm scan and the Direct layout. S2: 8 lanes was never faster.
- Lane layout (per tile): centroids in groups of `LANES`; coordinate c of centroid `j0 + g*LANES + l` sits at `g*LANES*d + c*LANES + l`. The partial last group of k is not stored; its centroids take the scalar path. Tile widths are multiples of `LANES`, so only the final tile has a tail.
- `BLOCK_ROWS = 256` unchanged: unit of carry, cuts, the row-order reduction, and the spherical `2*rows`-per-block objective term (`trainer.rs:504`).
- `TASK_ROWS: usize = 64` and `SPLIT_MAX_THREADS: usize = 32` in `trainer.rs`: a call's 256-row blocks are cut into 64-row tasks when the call has fewer blocks than `rayon::current_num_threads()` and that count is <= 32; otherwise one task per block. Both values were tuned on this machine (S5 `matrix-threads.txt`): at 64 threads more tasks cost more in wake-up and join than they save. Also used by `Model::assign`.

### API Design

No public API change in Phases 1-7. Every new signature is `pub(crate)`. `Trainer::accumulate(&[f32])`, `EMeans::train`, `Model::assign` keep their signatures, errors and messages.

Phase 8 adds one builder method, in the style of `spherical(bool)` and `capture_labels(bool)` (`src/config.rs:96-103`):

```rust
/// Use the Gemm kernel for every batch size, not only when `B * d >= 128000`. Faster for small
/// batches; gives the wrong centroid on data with a large common offset (see README). Default false.
pub fn always_gemm(mut self, always: bool) -> Self
```

- Stored on `EMeans`, passed to `Trainer::new`, which picks `Kernel::Gemm` when set and `Kernel::for_batch(B, d)` otherwise (`src/trainer.rs:118`). `Model` already keeps the trainer's kernel, so `Model::assign` follows.
- Keyed on the builder, never on chunk size or thread count: a run still uses one kernel, so determinism holds.
- Default false: every existing caller and every parity fixture keeps Direct below the threshold.

Bit-neutrality arguments:

**Lane argmin (change 1).**
- Pass 1 computes each score exactly as today: `row_norm + centroid_norm - 2.0 * ip`, then `< 0.0 -> 0.0`. IP: `ip`.
- Pass 2: lane l keeps `(best, id)` over ids `l, l+16, ...` with strict `<`, so each lane keeps its lowest id at its minimum. Lanes merge by `v < best || (v == best && v < INF && id < best_id)`. Tail ids exceed every lane id: strict `<`.
- Result: first index at the row minimum, the same as today's scalar scan.
- NaN loses every `<` on both paths. All-`+inf` row: label 0, score `inf` on both (the `v < INF` guard). `-0.0` cannot occur: norms are >= 0 and the clamp writes `+0.0`.
- IP mirrors it with `>` and `NEG_INFINITY`.

**Centroid tiling (change 2).**
- matrixmultiply 0.3.11 accumulates each C element with one FMA chain over the depth, from zero. Depth splits (`S_KC=256`) depend only on d. Edge tiles run `masked_kernel` with the same chain. Kernel choice depends only on CPU features (`gemm.rs:264-340,516-524`, `sgemm_kernel.rs:50-77,310-380,499-517`). So element (i, j) depends on row i, centroid j, d and the CPU, never on m, n or tile position.
- S1 proved it: d in {1, 8, 32, 100, 300} x k in {17, 256, 2100} x tile widths {1, 15, 16, 17, 1000, 2048} x row splits {1, 7, 64, 128}: 0 differing elements in 360 cases.
- Within a tile: first index at the tile minimum. Across tiles in ascending j, a later tile wins only on strict `<`. Same answer as the full scan; a tie straddling a boundary goes to the lower index.

**Norms once per call (change 3).** Same function, same inputs.

**Fused train region (change 4).**
- `accumulate_indexed(x, ids)` takes one whole batch (train always feeds whole batches, `src/config.rs:250-253`), so carry is empty and blocks are cut at 256 rows from the batch start exactly as `accumulate` cuts them. Each task gathers `x[ids[i]]` for its rows into its disjoint slice of the batch buffer: the same bytes, in the same order, as today's serial `gather_into`.
- Each task scans its slice before assigning it. After the join, the error reports the minimum offending offset across tasks: the same index `rows.iter().position(..)` finds today, so the same message (`chunk row R column C is V`). S5 `nan-error-check.txt`: `chunk row 48 column 5 is NaN` at 1 and 8 threads, before and after.
- Nothing observable changes before the error, per metric:
  - L2: the snapshot is copied from `centroids` at the start of the batch (`received() == 0 && metric == L2`, `trainer.rs:217`, kept as is). It is a derived copy that the next batch overwrites, and `centroids` itself is untouched.
  - Spherical: no copy; the snapshot is the normalized readout `refresh_readout` (`trainer.rs:603`) wrote after the last update, and the batch only reads it. Copying `centroids` here would move spherical bits.
  - Both: the batch buffer and label/score slices are scratch, and the reduction, `received()` and every accumulator run only after a clean join. `train()` returns the error and drops the trainer, as today.
- `Model::assign` (`model.rs:111`): each task scans its rows, then assigns; after the join, the minimum offending offset -> the same `row R column C is V` error. Labels computed for other rows are dropped; `Model::assign` takes `&self`, so nothing changes.
- `train` keeps scanning per batch: scanning all of `x` up front would reject a NaN in a row a T<1 run never visits, which trains fine today.

**Cluster-partitioned reduction (change 5).**
- Today: one thread walks rows in order; `counts[j] += 1`, `sums[j] += row` (f64), `objective += score`.
- After: P = `rayon::current_num_threads()` tasks, each owning a contiguous range of `k / P` clusters (the last takes the remainder); each walks all of the call's labels in row order (blocks in their existing order, carry block included) and adds only rows it owns. Each `sums[j]` sees the same adds in the same order.
- Workers add in place into disjoint slices of the existing `self.batch.sums` and `self.batch.counts`. Never call-local partials merged later: a partial started at zero and merged re-associates the adds ([2^60, -2^60, 1] gives 1 in row order and 0 split and merged).
- The serial objective loop keeps `+= 2*rows` ahead of each block's scores and the f32 `-2.0 * score` before the f64 conversion. Everything joins before `batch.assigned` and `apply`.
- `objective`, the spherical `2*rows` term and `pass_labels` stay on one thread, in row order (O(rows)).
- Not the first doc's rejected "parallel per-block partial sums merged in block order": there a cluster's sum is split by block and re-associated; here it is never split.
- Each task reads every label: O(P*rows) label reads against O(rows*d) adds moved off one thread. At d=8 that is 64 reads to spread 8 adds per row, so the switch is keyed on work: serial below `PARALLEL_REDUCE_VALUES` (rows*d), value set in Phase 5 from the d=32 and d=8 benches.
- Skewed labels (one hot cluster) leave one owner doing almost all adds: no worse than today's serial loop plus the label reads.
- Above the threshold this is a second parallel region per batch. Phase 5's bench gate is what proves it pays (the S5 lesson: regions are not free).

**Lane Direct (change 6).**
- Today: `squared_l2` is `sum = sum + (x - c) * (x - c)`, f32, from 0.0, coordinate 0 first (`assign.rs:216-221`); `inner_product` is `sum + x * c` (`:212`).
- Lane version: `[f32; LANES]` accumulators; for c in 0..d, `acc[l] = acc[l] + (x[c] - lane[c][l]) * (x[c] - lane[c][l])`. Lane l runs exactly the ops of `squared_l2(row, centroid_{g*16+l})`. Argmin by change 1's rule.
- Tiling over centroids: each lane fold depends only on (row, centroid), so the change-2 argument (first index at the tile minimum, ascending tiles, strict `<`) carries over with no sgemm premise at all.
- S2: bitwise (untiled layout) on d in {1, 2, 3, 8, 16, 33} x k in {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256}, random and quantized (tie-heavy) data, both metrics; identical digests on the d=8 mixture and on d=16, k=32768.

**Task split (change 7).** Blocks are still cut at 256 rows from the batch start; the reduction still walks blocks in order, so the spherical `2*rows` grouping is unchanged. Only the rayon task over a block's rows is split into 64-row pieces writing disjoint label/score slices. Per-row results depend only on the row and the snapshot (changes 1-3, 6), so the split is invisible. The split decision reads the thread count, which is allowed for the same reason.

**Pool-thread loop and permutation overlap (change 8).**
- `train()`'s loop body runs inside `rayon::scope`, so every batch's region is forked from a pool worker, not handed over from the calling thread. Same ops, same order. S5: fork + join overhead 0.38 + 1.83 ms -> 0.03 + 0.03 ms per train at 1 thread.
- During pass p, `rayon::join` draws pass p+1's permutation from `Stream::new(seed, StreamId::Pass(p + 1))`, an independent stream: the same permutation as drawing it after pass p. No draw after the last pass; `scatter_labels` gets the last pass's order. S5: permutation 1.6 -> 0.25-0.37 ms per train.
- An error during pass p returns after the `join`, so it waits for the speculative draw: at most one `permutation(n)`. Accepted; no cancellation.
- A caller already inside a rayon pool: `scope` runs on the caller's worker, so training uses the caller's pool, as today's regions do. The Phase 1 `train()` test runs inside explicit pools to cover it.

### Implementation Plan

10 phases, 0 through 9. Phase 0 is done. One commit per phase, `otto ci` green at each. Phase 8 adds the `always_gemm` opt-in; Phase 9 measures.

#### Phase 0: Prove the bit-neutrality claims and size the wins
**Model:** opus
- Done 2026-11-13, zero repo code. Scratch crates in the session scratchpad (copies renamed `e-means-<x>` so one harness links all of them); raw logs there, numbers here.
- S1, sgemm element independence: 0 differing elements in 360 cases (change 2 above).
- S2, lane Direct (d=8 mixture, median of 15):

  | impl | 1t train (taskset) | 1t assign | 8t train | 64t train | 64t assign |
  |---|---|---|---|---|---|
  | current Direct | 230.1 ms | 43.98 ms | 80-81 ms | 96-100 ms | 2.3-2.4 ms |
  | lane L16, layout per task | 85.8 | 15.68 | 34-37 | 45-50 | 1.0-1.1 |
  | forced Gemm + lane scan + tile reuse | 60.7 | 10.47 | 27-31 | 40-43 | 0.8-0.9 |
  | emeans | 51.8 | 8.23 | 18-19 | 43-46 | 0.7-0.8 |

  - d=16, k=32768, n=200000, T=1, 64t: current 6.4 s, lane 1.9 s, emeans 0.78 s.
- S3, centroid tiling at k=65536, d=32, n=200000, T=1 (3 runs each, 2^18 cap):

  | impl | 64t wall | 64t RSS | 1t wall |
  |---|---|---|---|
  | current | 4.11-4.45 s | 2.20 GB | 25.1-25.4 s |
  | tiled | 0.63-0.64 s | 157.7 MB | 10.0-10.7 s |
  | emeans | 0.42-0.45 s | 157.5 MB | 6.8-7.3 s |

  - Bitwise identical digests to current at k=65536 (64 and 13 threads), k=8192 with cap 3000, the d=32 mixture at caps 2^18 and 1000.
- S4, sub-block tasks without the fused region: bitwise identical; alone, d=8 64t 96-100 -> 80-83 ms; on top of lane Direct, no gain at 64t and none beyond noise at 8t. S5 superseded the 8-thread half of this (below); the cause of S4's null result is not explained.
- S5, the 8-thread gap (d=8 train, ms per train, mean of 15, 8 threads pinned to one CCD):
  - Cause: of lane Direct's 17.5 ms gap, 14.5 ms is in the parallel region (4 tasks on 8 threads; the longest 256-row task decides each batch) and 3.0 ms is serial (permutation +0.95, gather +1.04, finite scan +0.92, layout prep +0.18). Forced Gemm: 6.9 + 3.1 of 10.0 ms.
  - emeans runs one region per batch (490 per train, same as ours) of 128-row tasks with gather fused in, a serial f64 row-order reduction, no per-thread partial sums and no finite scan. Nothing in its batch structure is blocked by bit-neutrality.
  - Prototype with changes 4, 7, 8 (median of 15, ours and emeans interleaved, two rounds; ratios ours / emeans):

    | cell | lane before | lane after | forced Gemm before | forced Gemm after |
    |---|---|---|---|---|
    | 1t, `taskset -c 2` | 1.56x | 1.44-1.48x | 1.11x | 0.96x |
    | 8t, unpinned | 1.97x | 1.07-1.19x | 1.60-1.66x | 0.98-1.11x |
    | 8t, `taskset -c 0-7` | 1.42x | 0.98-0.99x | 1.14x | 0.79-0.81x |
    | 64t | 1.08x | 0.93x | 0.91-0.95x | 0.755x |

  - Runs: the table quotes `pairs-2` and `pairs-3`, both the final configuration (`EM_S5=14 EM_SUBAUTO=1 EM_SNAPFIRST=1`). `pairs-1` ran without `EM_SNAPFIRST` (the rejected search-`centroids` variant): lane 8t unpinned 1.194x, pinned 2.42 -> 1.15x. The gated unpinned 8t lane cell holds in all three: 1.194, 1.067, 1.191.
  - The pinned 8t baseline is bimodal across runs: 2.40-2.42 (`pairs-1`, `attrib-1`) or 1.42-1.43 (`pairs-2`, `pairs-3`). Cause not found; pinned cells are not gated.
  - Attribution at 8 threads pinned, taken in the 2.40 mode: smaller tasks alone 1.58 (unpinned 2.10 -> 1.68), pool thread alone 2.23, fusion alone 2.28, fusion + smaller tasks 1.45, all three 1.20. Smaller tasks are the largest single lever with or without fusion, and the switches compound.
  - S5 built one shared, untiled lane layout before the region (the layout Alternative 5 rejects), not the per-task tiled layout of change 6. Going from 4 to 16 tasks multiplies per-task transpose work by 4, so the S5 8t ratios do not measure the final composition. Phase 7's `otto h2h` cell is the gate that does.
  - d=32 at 8 threads (never measured before): lane argmin + tiled Gemm 1.33-1.62x; cause the serial gather 4.2-5.6 ms, finite scan 3.2 ms, permutation 1.35 ms (emeans 0, 0, 0.6-0.9). With changes 4, 7, 8 on the forced-Gemm prototype: 1.065-1.18x. Tiled Gemm + fusion together was not run.
  - Separate parallel gather and scan regions (the first draft of this plan's Phase 4): 1470 regions per train; lane d=8 39.3 -> 57.7 ms (8t), 48.2 -> 206.9 ms (64t). Replaced by change 4.
  - Every S5 switch: bitwise digests equal to current (lane) and to the forced-Gemm prototype; `cargo test --release` parity 10/10, determinism included; one `unsafe`.
- **Success criteria:** met. S1 0 differing bits; S2, S3 and S5 prototypes pass `cargo test --release` (parity 10/10, determinism included) and match current digests.

#### Phase 1: Guard tests, the large-k example, `bin/h2h`
**Model:** sonnet
- Oracles in `src/assign/tests.rs`: today's `direct_l2`, `direct_inner_product`, and the Gemm scans, copied verbatim under `#[cfg(test)]` as `oracle_*`.
- Tie fixtures: equal minima inside one lane, across lanes, in the tail, all-zero clamped row, all-`inf` row; both metrics.
- Golden digests: `to_bits` of centroids, labels and objective for every parity fixture that runs `Kernel::Direct`, captured from `main` and committed as test constants. Direct is plain IEEE f32/f64 with no FMA, so the bits hold on any x86-64 runner. Gemm fixtures are CPU-specific (sgemm dispatches AVX-512 | FMA | fallback); the oracle tests guard them on the running machine.
- `train_matches_caller_loop_in_pools`, next to `train_matches_a_caller_loop_from_the_seeded_init` (`src/config/tests.rs:182`): `train()` against a caller loop through the public `accumulate`, bitwise centroids, epoch stats, objective and captured labels, inside explicit 1-thread and 8-thread pools (`pool.install`). Cases: L2 Direct (d=8, B=1024, so 4 blocks and 16 tasks at 8 threads), L2 Gemm (d=32, B=4096, k=64), spherical, T=5 and T=2.5 (fractional), n not a multiple of B (short final batch). Today's train-vs-caller tests run L2 only, at B=4 and B=8, one block, on the global pool. After Phase 4 this test pins `accumulate_indexed` to `accumulate`; after Phase 7 it pins the pool-thread loop and the permutation overlap.
- `examples/large-k.rs`: n=200000, d=32, k=65536, T=1, seeded data from `benches/common/mod.rs`'s generator; prints the wall time of `train()` alone (data generation excluded) and a digest of centroids and labels. Its run on unchanged kernels is recorded here as the baseline the Phase 3 gate is checked against (S3 used different data, and its 0.63 s was train time inside a 0.76-0.77 s process).
- `bin/h2h/`: the head-to-head as a committed tool, a standalone crate outside the package (its own `Cargo.toml`, not a workspace member, so e-means' dev-deps stay clean). Depends on `e-means = { path = "../.." }` and `emeans = { git = "https://github.com/savannahar68/emeans", rev = "5b283a9" }`. Data: the `tests/quality.rs` mixture.
  - Interleaved sampling: 15 pairs per cell, ours then emeans alternating, one warm-up each. Ratio = median of the 15 paired ratios (ours / emeans). Both assign closures do equal work (no clone on one side).
  - Cells and bounds come from one table in its `main.rs`, the Acceptance Criteria grid; exits 1 if any gated cell misses, prints every cell either way.
  - otto task `h2h` (not in `ci`): `cargo run --release --manifest-path bin/h2h/Cargo.toml`, with `taskset -c 2` for the 1-thread cells.
- **Success criteria:**
  - `otto ci; echo $?` prints 0, new tests passing against unchanged kernels.
  - `otto h2h` runs on unchanged kernels and prints every cell except the `always_gemm` cells, which print `n/a` until Phase 8 adds the option; its ratios are recorded here as the pre-change baseline (it is expected to exit 1).
  - `otto h2h --cell <name>` runs one cell and exits on that cell's bound alone (used by Phases 6 and 7).
  - Mutating an oracle's `<` to `<=` makes a lane or tail tie test fail (run once, restored).
  - `cargo run --release --example large-k` exits 0.
- **Recorded in Phase 1, on unchanged kernels (`5f0b62d` plus the Phase 1 tests and tools), 2026-11-13, load average 1.4-2.1:**
  - `examples/large-k.rs` (n=200000, d=32, k=65536, T=1; data from `benches/common/mod.rs`, 256 components; digest `440bf89bb2f83d80`, identical at 64 and 1 thread and across 3 runs):

    | threads | `train` printed | process wall | peak RSS |
    |---|---|---|---|
    | 64 (3 runs) | 4.242 s, 4.370 s, 4.495 s | 4.39 s, 4.51 s, 4.64 s | 2201148, 2201444, 2200412 KB |
    | 1 (`RAYON_NUM_THREADS=1 taskset -c 2`) | 25.501 s | 25.64 s | 158964 KB |

  - `otto h2h` pre-change (ratio = median of 15 paired ratios, ours / emeans; ms are medians; exit 1, all 10 gated cells miss, the 8 `always_gemm` cells print `n/a`):

    | cell | ours ms | emeans ms | ratio | bound |
    |---|---|---|---|---|
    | train-d32-1t | 142.12 | 86.44 | 1.648 | <=1.3 |
    | train-d32-8t | 37.16 | 18.70 | 2.031 | <=1.3 |
    | train-d32-32t | 32.15 | 15.20 | 2.162 | printed |
    | train-d32-64t | 42.02 | 26.56 | 1.554 | <=1.3 |
    | assign-d32-1t | 30.58 | 16.94 | 1.808 | <=1.3 |
    | assign-d32-8t | 4.89 | 2.35 | 2.065 | printed |
    | assign-d32-32t | 2.57 | 0.98 | 2.635 | printed |
    | assign-d32-64t | 2.01 | 0.70 | 2.900 | <=1.6 |
    | train-d8-default-1t | 229.27 | 52.18 | 4.408 | <=1.6 |
    | train-d8-default-8t | 84.83 | 14.41 | 5.844 | <=1.35 |
    | train-d8-default-32t | 101.97 | 29.33 | 3.499 | printed |
    | train-d8-default-64t | 88.40 | 38.92 | 2.243 | <=1.3 |
    | assign-d8-default-1t | 44.19 | 9.94 | 4.442 | <=2.1 |
    | assign-d8-default-8t | 6.68 | 1.37 | 4.809 | printed |
    | assign-d8-default-32t | 2.27 | 0.52 | 4.276 | printed |
    | assign-d8-default-64t | 2.18 | 0.48 | 4.549 | <=1.6 |

    The `always_gemm` cells (`train-d8-gemm-*`, `assign-d8-gemm-*`, 8 cells) print `n/a` until Phase 8.

#### Phase 2: Lane argmin and norms once per call
**Model:** sonnet
- Changes 1 and 3. The tile is still `rows x k`.
- **Success criteria:**
  - `cargo test oracle` passes: bitwise labels and scores against the Gemm oracles on random data and every tie fixture.
  - `cargo test --test parity` passes, determinism included.
  - `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/` prints `src/assign.rs:1`.

#### Phase 3: Centroid tiling under `TILE_FLOATS`
**Model:** opus
- Change 2. The cap is a parameter of the crate-private tile loop, so tests can force small tiles; `TILE_FLOATS` is the production argument.
- Re-measure 2^16 against 2^18 with `examples/large-k.rs`, keep the faster at 64 threads, record both here.
- `Cargo.toml`: `matrixmultiply = "=0.3.11"` (today a caret requirement). S1 becomes a unit test, `gemm_element_is_independent_of_tile_and_split`: d in {1, 8, 32, 300}, k in {17, 256}, tile widths {1, 15, 16, 17}, row splits {1, 7, 64}, `to_bits` equal.
- **Success criteria:**
  - Tiled equals the oracle bitwise with a cap forcing 5+ tiles, including a tie whose equal minima straddle a tile boundary (lower index wins).
  - `/usr/bin/time -f "%M KB" target/release/examples/large-k` at 64 threads: peak <= 200000 KB, printed `train` time <= 0.8 s, digest equal to the Phase 1 run's.
  - Parity and determinism pass.
- **Recorded in Phase 3, 2026-11-13, load average 2.5-10 (`examples/large-k.rs`, digest `440bf89bb2f83d80` on every run below):**
  - Cap re-measure, 64 threads, two binaries interleaved, 7 pairs (`train` printed, peak RSS):

    | cap | train (7 runs) | median | peak RSS |
    |---|---|---|---|
    | 2^16 | 0.630, 0.639, 0.620, 0.612, 0.621, 0.634, 0.634 s | 0.630 s | 99936-103364 KB |
    | 2^18 | 0.624, 0.618, 0.624, 0.613, 0.618, 0.626, 0.626 s | 0.624 s | 153740-156192 KB |

    A separate 3-run set before it: 2^16 0.649, 0.667, 0.660 s; 2^18 0.619, 0.657, 0.638 s. 2^18 is faster at 64 threads in both sets (5 of 7 pairs), so `TILE_FLOATS = 1 << 18`.
  - 1 thread (`RAYON_NUM_THREADS=1 taskset -c 2`), 2 runs each, not the deciding cell: 2^16 8.153, 7.924 s (98320, 97816 KB); 2^18 9.033, 8.980 s (99140, 99344 KB).
  - Gate on the committed build (2^18), `/usr/bin/time -f "%M KB" target/release/examples/large-k`, 64 threads, 3 runs: `train` 0.669, 0.630, 0.621 s; peak 154548, 154752, 157096 KB; digest `440bf89bb2f83d80`. Passes (<= 200000 KB, <= 0.8 s, Phase 1 digest).

#### Phase 4: Fused train region
**Model:** opus
- Change 4: crate-private `Trainer::accumulate_indexed(x, ids)`, used only by `train()`; `train()` stops calling `gather_into`. Fused per-task scan in `Model::assign`. `Trainer::accumulate(&[f32])` unchanged.
- Record `otto bench` and `otto h2h` here before and after the change; the after run is Phase 5's baseline.
- **Success criteria:**
  - `non_finite_row_is_rejected_before_anything_changes` and `train_errors_are_typed` pass unchanged. New tests, 1-thread and 8-thread pools, today's exact message text:
    - `train`: NaNs at gathered batch positions 5 and 900 of the first batch (the test maps them to input rows through the pass-0 permutation), which land in different tasks; the error names batch row 5. Replacing the minimum (`fetch_min`) with a last-writer store fails the test (run once, restored).
    - `Model::assign`: n=300 rows (a short final task), NaNs at rows 5 and 290; the error names row 5.
  - `train_matches_caller_loop_in_pools` and `train_matches_a_caller_loop_from_the_seeded_init` pass: `train()` through `accumulate_indexed` equals the caller loop through `accumulate`, every case, both pools.
  - `rg -n 'gather_into' src/config.rs` shows no call inside `train_rows`.
  - `otto bench` medians for both `train` benches are no worse than this phase's before run.
- **Recorded in Phase 4, 2026-11-13.** Before = `684cfe0` (Phase 3), run from a clean worktree, load average 2-3. After = the Phase 4 commit, load average 10.9 at the start of `otto bench` and 26.6 at the end of `otto h2h`. The after run is Phase 5's baseline.
  - `otto bench` (criterion estimate, median of the interval):

    | bench | before | after |
    |---|---|---|
    | `train/n100000_d32_k256_t5` | 41.879 ms [40.624, 43.027] | 29.230 ms [28.683, 29.818] |
    | `train/n100000_d8_k256_t5` | 97.075 ms [93.771, 102.92] | 95.025 ms [92.121, 97.489] |
    | `assign/direct_d8_k256` | 2.2778 ms | 2.0642 ms |
    | `assign/gemm_d8_k256` | 829.69 us | 616.39 us |
    | `assign/direct_d32_k256` | 7.5962 ms | 6.7976 ms |
    | `assign/gemm_d32_k256` | 1.6058 ms | 825.36 us |

  - `otto h2h` (ratio = median of 15 paired ratios, ours / emeans; ms are medians). Before: exit 1, 8 gated cells miss. After: exit 1, 5 gated cells miss (all d=8 default; Direct is unchanged until Phase 6). `always_gemm` cells `n/a` in both:

    | cell | before ours / emeans ms | before ratio | after ours / emeans ms | after ratio | bound |
    |---|---|---|---|---|---|
    | train-d32-1t | 92.90 / 82.19 | 1.120 | 100.12 / 92.03 | 1.089 | <=1.3 |
    | train-d32-8t | 31.50 / 18.11 | 1.707 | 21.26 / 19.07 | 1.155 | <=1.3 |
    | train-d32-32t | 30.79 / 15.29 | 2.011 | 20.77 / 15.82 | 1.302 | printed |
    | train-d32-64t | 38.23 / 26.32 | 1.479 | 25.26 / 23.19 | 1.078 | <=1.3 |
    | assign-d32-1t | 18.39 / 16.23 | 1.135 | 17.82 / 16.25 | 1.097 | <=1.3 |
    | assign-d32-8t | 2.77 / 1.97 | 1.412 | 2.70 / 2.37 | 1.151 | printed |
    | assign-d32-32t | 1.49 / 0.71 | 2.109 | 0.93 / 0.80 | 1.163 | printed |
    | assign-d32-64t | 1.46 / 0.67 | 2.228 | 0.82 / 0.66 | 1.237 | <=1.6 |
    | train-d8-default-1t | 229.44 / 52.00 | 4.411 | 226.36 / 53.72 | 4.214 | <=1.6 |
    | train-d8-default-8t | 97.34 / 16.00 | 6.144 | 86.65 / 14.70 | 5.896 | <=1.35 |
    | train-d8-default-32t | 86.19 / 28.27 | 3.080 | 88.94 / 28.48 | 3.091 | printed |
    | train-d8-default-64t | 85.49 / 37.81 | 2.271 | 88.74 / 40.95 | 2.160 | <=1.3 |
    | assign-d8-default-1t | 44.40 / 10.00 | 4.440 | 52.03 / 10.27 | 5.068 | <=2.1 |
    | assign-d8-default-8t | 5.94 / 1.27 | 4.685 | 6.66 / 1.35 | 4.965 | printed |
    | assign-d8-default-32t | 2.58 / 0.55 | 4.681 | 2.19 / 0.51 | 4.333 | printed |
    | assign-d8-default-64t | 2.14 / 0.49 | 4.338 | 2.08 / 0.48 | 4.326 | <=1.6 |

  - `assign-d8-default-1t` after (52.03 ms) is a load mode, not the change: `otto h2h --cell assign-d8-default-1t` on `684cfe0` also printed 52.00 and 51.90 ms right after, and the current build 44.10. Three interleaved pairs then (base, current): 44.23 / 44.27, 44.30 / 44.22, 44.15 / 44.23 ms; ratios 4.485 / 4.451, 4.479 / 4.478, 4.493 / 4.485.

#### Phase 5: Cluster-partitioned reduction
**Model:** opus
- Change 5, serial below `PARALLEL_REDUCE_VALUES`.
- Today's reducer copied verbatim under `#[cfg(test)]` as `oracle_reduce`.
- **Success criteria:**
  - A unit test runs the reduction at partition counts 1, 2, 7, 64 against `oracle_reduce` and asserts bitwise `counts`, `sums`, `objective`: nonzero starting accumulators, cancellation-heavy values (2^60, -2^60, 1), a carry boundary, a short final block, both metrics, and batches on both sides of `PARALLEL_REDUCE_VALUES`.
  - Determinism test and the Phase 1 golden digests pass.
  - `otto bench` medians for `train/n100000_d32_k256_t5` and `train/n100000_d8_k256_t5` are no worse than the Phase 4 run recorded here; if one is, the threshold moves until both hold.
- **Recorded in Phase 5, 2026-11-13.** The Phase 4 after run above ran at load 10.9-26.6, so the Phase 4 baseline was re-measured: `3564435` from a clean worktree, interleaved with the Phase 5 build, two rounds (`cargo bench --bench train`, criterion estimate, median of the interval). Load average (1 min) at the start of each run; most of it is the benches' own rayon workers, the machine was otherwise idle.

  | run | load | `train/n100000_d32_k256_t5` | `train/n100000_d8_k256_t5` |
  |---|---|---|---|
  | Phase 4 (`3564435`), round 1 | 13.85 | 27.971 ms [27.432, 28.468] | 84.168 ms [82.097, 87.327] |
  | Phase 5, round 1 | 18.98 | 24.799 ms [24.662, 24.950] | 82.543 ms [81.803, 83.155] |
  | Phase 4 (`3564435`), round 2 | 22.57 | 27.643 ms [27.346, 28.012] | 88.278 ms [84.615, 92.770] |
  | Phase 5, round 2 | 25.74 | 24.806 ms [24.713, 24.956] | 87.108 ms [86.000, 88.332] |

  - Gate passes: d=32 is ~11% faster in both rounds (it crosses the threshold); d=8 stays serial and is within noise of the baseline, lower in both rounds. Both are also below the Phase 4 numbers recorded above (29.230, 95.025 ms).
  - `PARALLEL_REDUCE_VALUES = 1 << 16` (65536). The d=32 train batch is 8192 x 32 = 262144 values, the d=8 batch 1024 x 8 = 8192. Threshold probe, `1 << 13` (d=8 goes parallel) against `1 << 16`, interleaved, two rounds, load 15.7-31.1: d=8 199.39, 196.89 ms against 95.133, 95.494 ms (2.1x slower: 490 extra 64-thread regions of 8 adds per row); d=32 26.744, 26.813 against 25.242, 25.429 ms (same code path at both values). Below 8193 fails the d=8 gate; above 262144 drops the d=32 gain (the serial path is the baseline). 2^16 sits inside that window; the break-even between the two was not measured.
  - Oracle tests: `partitioned_reduce_matches_the_oracle_bitwise` (owners 1, 2, 7, 64; k=100; both metrics; d=3 over 805 rows and d=8 over 513 rows) and `reduce_on_both_sides_of_the_parallel_threshold_matches_the_oracle` (1- and 8-thread pools, 805 and 8485 rows at d=8). Mutations, run once and restored, both fail both tests: an owner adding into a zero-started partial merged at the end (`sums[0] (cluster 0) got 1.5 want 1`, the [2^60, -2^60, 1] cluster with a 0.5 start, other starting sums zeroed to isolate it), and an owner walking the blocks in reverse.
  - `cargo test --release --test parity`: 11 passed, `determinism_across_thread_counts_and_chunkings` and `direct_fixtures_match_their_golden_digests` ok. `cargo test --lib`: 143 passed. `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`. `otto ci` exit 0.

#### Phase 6: Lane-vectorized Direct
**Model:** opus
- Change 6, L2 and IP, layout tiled under `TILE_FLOATS` and built per task into the task's scratch.
- Measure d=8 train against the Phase 5 numbers with `bin/h2h` and record them here; S2's 2.7x was untiled, so a tiled regression shows up here, not at release.
- **Success criteria:**
  - `cargo test oracle` passes against the Direct oracles for d in {1, 2, 3, 8, 16, 33}, k in {1, 15, 16, 17, 33, 256}, random and quantized data, both metrics.
  - Same oracle comparison with the tile cap forced small (5+ tiles, tile width a multiple of `LANES`), including a tie straddling a tile boundary.
  - A unit test asserts the task scratch capacity after a Direct call at d=64, k=65536 is <= `max(TILE_FLOATS, LANES*d)` f32.
  - Phase 1 golden digests pass (every Direct parity fixture bitwise); `common_offset_direct_is_right_and_gemm_cancels` passes.
- **Recorded in Phase 6, 2026-11-13.** Before = the Phase 5 commit `3468734`, exported to a clean tree; after = the Phase 6 build. `bin/h2h --cell train-d8-default-1t train-d8-default-8t train-d8-default-64t assign-d8-default-1t assign-d8-default-64t`, the two builds interleaved, two rounds (ratio = median of 15 paired ratios, ours / emeans; ms are medians). Load average (1 min) at the start of each run: 2.27, 2.31, 2.28, 2.55.

  | cell | Phase 5 r1 ours / emeans ms | ratio | Phase 6 r1 ours / emeans ms | ratio | Phase 5 r2 ratio | Phase 6 r2 ours / emeans ms | ratio | bound |
  |---|---|---|---|---|---|---|---|---|
  | train-d8-default-1t | 227.83 / 53.91 | 4.235 | 84.11 / 52.47 | 1.603 | 4.227 | 83.38 / 52.05 | 1.600 | <=1.6 |
  | train-d8-default-8t | 98.77 / 16.73 | 5.935 | 32.54 / 16.52 | 1.969 | 4.533 | 35.69 / 15.90 | 2.185 | <=1.35 (Phase 7 gate) |
  | train-d8-default-64t | 79.29 / 37.89 | 2.087 | 38.58 / 38.90 | 0.988 | 2.133 | 42.35 / 42.03 | 1.011 | <=1.3 |
  | assign-d8-default-1t | 44.18 / 9.52 | 4.643 | 15.67 / 9.02 | 1.736 | 4.679 | 15.63 / 9.00 | 1.737 | <=2.1 |
  | assign-d8-default-64t | 1.96 / 0.48 | 4.192 | 0.88 / 0.55 | 1.662 | 4.024 | 0.72 / 0.46 | 1.578 | <=1.6 |

  - The tiled per-task layout lands where S2's untiled lane prototype did (85.8 ms 1t train, 34-37 ms 8t, 45-50 ms 64t, 15.68 ms 1t assign): no tiling regression at d=8 (k=256 is one tile of 256 centroids, 2048 f32 of layout per task).
  - `cargo bench --bench train -- n100000_d8` (load 2.62 at start): `train/n100000_d8_k256_t5` 42.880 ms [42.206, 44.317], against Phase 5's 82.543 and 87.108 ms.
  - `train-d8-default-1t` sits on its 1.6 bound (1.603, 1.600); the 8t cell is Phase 7's gate (changes 7 and 8).
  - Mutations, run once and restored: the cross-tile merge taking an equal score (`better(best, *score) || best == *score`) fails all three `tiled_lane_direct_*` tie and tile tests (`[15, 16] L2 cap=0: left [16], right [15]`); a lane keeping the later id on an equal score fails `tie_between_two_centroids_in_one_lane_goes_to_the_lower`, `a_row_equal_to_every_centroid_takes_index_zero_with_a_positive_zero_score` and both lane-Direct oracle tests.
  - `cargo test oracle`: 11 passed. `cargo test --release --test parity`: 11 passed, determinism and `direct_fixtures_match_their_golden_digests` ok. `cargo test --lib`: 150 passed. `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`. `otto ci` exit 0.

#### Phase 7: Task split, pool-thread loop, permutation overlap
**Model:** opus
- Changes 7 and 8. `TASK_ROWS` and `SPLIT_MAX_THREADS` as named consts; the split rule in one function used by `accumulate_indexed`, `assign_blocks` and `Model::assign`.
- **Success criteria:**
  - Determinism test, Phase 1 golden digests and `assign_over_many_blocks_is_thread_count_invariant` pass.
  - A unit test asserts the split rule: 4 blocks at 8 threads -> 16 tasks; 4 blocks at 64 threads -> 4 tasks; 32 blocks at 8 threads -> 32 tasks; 8 blocks at 8 threads -> 8 tasks (blocks == threads, no split); 4 blocks at 32 threads -> 16 tasks and at 33 -> 4 (the `SPLIT_MAX_THREADS` edge); 300 rows at 8 threads -> 5 tasks (one 256-row block as 4, the 44-row block as 1).
  - `otto h2h --cell train-d8-default-8t; echo $?` prints 0 (bound 1.8x after the 2026-11-14 reset; originally 1.35x). The whole grid exits 0 only at Phase 9, after `always_gemm` exists.
- **Recorded in Phase 7, 2026-11-13. Implementation in the working tree on `d8b88b4`, NOT committed: the 8t gate misses.** (Superseded 2026-11-14: committed as `86428d5` after the bound reset; see the gate line below and Resolved Decisions.)
  - Determinism, golden digests, `assign_over_many_blocks_is_thread_count_invariant`: pass (`cargo test --release --test parity` 11 passed). Split-rule unit test passes all seven cases. `otto ci` exit 0 (152 lib tests). `rg -c` unsafe: `src/assign.rs:1`.
  - Gate: `otto h2h --cell train-d8-default-8t` exits 1. Ratios over 20 runs of the cell (load 1.5-4.1): 1.662, 1.497, 1.324 (the one pass), 1.449, 1.365, 1.532, 1.413, 1.467, 1.373, 1.629, 1.649, 1.485, 1.367, 1.352, 1.567, 1.666, 1.455, 1.594, 1.482, 1.459. Ours 21-25 ms, emeans 12-16.5 ms.
  - Full `otto h2h`, load 2.50 at start, 2.17 at end, exit 1 (ratio = median of 15 paired ratios; ms are medians):

    | cell | ours ms | emeans ms | ratio | bound | verdict |
    |---|---|---|---|---|---|
    | train-d32-1t | 99.61 | 92.74 | 1.093 | <=1.3 | pass |
    | train-d32-8t | 21.05 | 19.62 | 1.076 | <=1.3 | pass |
    | train-d32-32t | 16.39 | 16.04 | 1.031 | printed | |
    | train-d32-64t | 22.76 | 26.04 | 0.903 | <=1.3 | pass |
    | assign-d32-1t | 18.15 | 16.41 | 1.107 | <=1.3 | pass |
    | assign-d32-8t | 2.41 | 2.19 | 1.098 | printed | |
    | assign-d32-32t | 0.82 | 0.72 | 1.126 | printed | |
    | assign-d32-64t | 0.86 | 0.70 | 1.199 | <=1.6 | pass |
    | train-d8-default-1t | 83.33 | 51.67 | 1.617 | <=1.6 | MISS |
    | train-d8-default-8t | 18.62 | 12.02 | 1.517 | <=1.35 | MISS |
    | train-d8-default-32t | 29.29 | 28.78 | 1.015 | printed | |
    | train-d8-default-64t | 41.28 | 40.13 | 1.027 | <=1.3 | pass |
    | assign-d8-default-1t | 18.54 | 9.54 | 1.944 | <=2.1 | pass |
    | assign-d8-default-8t | 2.43 | 1.36 | 1.766 | printed | |
    | assign-d8-default-32t | 0.84 | 0.50 | 1.669 | printed | |
    | assign-d8-default-64t | 0.84 | 0.52 | 1.665 | <=1.6 | MISS |

    The 8 `always_gemm` cells print `n/a`. `train-d8-default-1t` runs the Phase 6 code path (4 blocks >= 1 thread: no split; `join` runs inline), as does `assign-d8-default-64t` (no split above 32 threads). Phase 6 measured them at 1.600-1.603 and 1.578-1.662. `train-d8-default-32t` went 3.09 (Phase 4) -> 1.015: the split at 32 threads.
  - Root cause of the 8t miss, measured on an instrumented scratch copy (per-train means over 17 trains, 8 threads unpinned): parallel region 18.0-18.6 ms of 20.8-22.8; summed task kernel time 101-106 ms (12.7-13.2 ms per thread if perfectly spread), gather 2-2.8, scan 1.6-2.1, per-task transpose 5.0-5.3 (all summed over tasks); serial `reduce` 1.2-1.5 ms, `update` 0.9-1.0 ms, pass-0 permutation 0.35 ms; `begin`, `finalize` and `scatter_labels` < 0.15 ms together.
    - The S5 caveat is not it. A thread-local layout transposed once per thread per batch (no per-task alloc, no per-task transpose) measured 1.41-1.50 against 1.37-1.48 for the build, interleaved, 3 rounds.
    - Task size is not it. Forced 32-, 64-, 128-row tasks, interleaved, 4 rounds: 1.45-1.49, 1.36-1.56, 1.41-1.63. Forced 256-row tasks (no split): 2.00-2.29, so the split is the win change 7 argued.
    - The S5 prototype misses this gate in `bin/h2h` too. `ours-s5` (`EM_S5=14 EM_SUBAUTO=1 EM_SNAPFIRST=1`) linked into a copy of `bin/h2h`, interleaved with this build, 8 rounds: prototype 1.351-1.541 (median ~1.42), this build 1.352-1.666 (median ~1.52); ours 20.1-23.0 ms against 21.3-24.7 ms. The same prototype in its own S5 harness the same hour: 1.121, 1.193, 1.196 (S5 recorded 1.07-1.19).
    - The harnesses disagree on emeans. The S5 harness calls both crates from the main thread on the global pool (`RAYON_NUM_THREADS=8`); `bin/h2h` calls both inside `pool.install`. In S5, emeans pays the calling-thread-to-pool handoff on each of its 490 regions; in `bin/h2h` it does not. emeans measures 18.3-20.3 ms in the S5 harness and in a `bin/h2h` copy that skips `install`, against 12-16.5 ms in `bin/h2h`. Change 8's pool-thread loop removes that handoff for ours; `bin/h2h` already removes it for both, so the S5 8t ratios carried an emeans penalty `bin/h2h` does not apply. The `bin/h2h` copy without `install` puts this build at 1.30-1.43 (3 rounds).
    - What is left is mostly the lane Direct kernel's per-row cost (parked non-goal; `assign-d8-default-1t` 1.94 here): ~13 ms of kernel per thread against emeans' whole 12-16 ms train, plus the serial reduce and update (~2.2 ms) and region sync (~5 ms).

  - Gate after the 2026-11-14 bound reset (1.8x): `otto h2h --cell train-d8-default-8t` exit 0, ratio 1.727 (ours 27.97 ms, emeans 16.50 ms), load average 6.18 at start. Phase 7 committed.

#### Phase 8: `always_gemm` opt-in
**Model:** sonnet
- `EMeans::always_gemm(bool)` per API Design; README: one line in the kernel section naming the opt-in and its offset limit.
- **Success criteria:**
  - A test trains d=8, B=1024 with `always_gemm(true)` and asserts `Model`'s kernel is Gemm, and with the default asserts Direct; `model_keeps_the_kernel_picked_from_batch_times_d` passes unchanged.
  - The offset fixture through the builder: default labels it 1 (right), `always_gemm(true)` labels it 0 (the documented limit).
  - Determinism test passes with `always_gemm(true)` on the d=8 parity fixtures (1 and 8 threads, all chunkings).

- **Observed (Phase 8, recorded not gated; gated at Phase 9):**
  - `otto h2h --cell "<the 8 always_gemm cells>"` exit 0, load average 3.23 at start, 3.27 at end (ratio = median of 15 paired ratios, ours / emeans):

    | cell | ours ms | emeans ms | ratio | bound |
    |---|---|---|---|---|
    | train-d8-gemm-1t | 67.83 | 55.99 | 1.238 | <=1.3 |
    | train-d8-gemm-8t | 18.98 | 15.85 | 1.106 | <=1.3 |
    | train-d8-gemm-32t | 26.27 | 29.24 | 0.902 | printed |
    | train-d8-gemm-64t | 34.78 | 40.86 | 0.853 | <=1.3 |
    | assign-d8-gemm-1t | 12.54 | 10.48 | 1.196 | <=1.4 |
    | assign-d8-gemm-8t | 1.77 | 1.47 | 1.169 | printed |
    | assign-d8-gemm-32t | 0.57 | 0.50 | 1.172 | printed |
    | assign-d8-gemm-64t | 0.57 | 0.49 | 1.175 | <=1.6 |

#### Phase 9: Re-measure, README
**Model:** sonnet
- Run `otto h2h`, `otto bench` and `examples/large-k.rs`; replace the README performance numbers with this run.
- **Success criteria:**
  - Every Acceptance Criteria line passes, its observed value recorded under it.
  - The README's numbers are the ones recorded in this phase, each with the command that produced it.

- **Recorded in Phase 9, 2026-11-14, on `110d271` (plus the README and this doc).** Load average (1 min) at the start of each run: large-k 3.45, 4.38, 4.38; `otto h2h` 4.03; `otto bench` 5.18 (the residue of the h2h run's own workers; the machine was otherwise idle). `otto h2h` and the full grid: see the Acceptance Criteria. Criteria walked: all five pass, observed values under each. PASS.
  - `otto bench` (criterion estimate, median of the interval):

    | bench | Phase 4 after | Phase 9 |
    |---|---|---|
    | `train/n100000_d32_k256_t5` | 29.230 ms | 21.680 ms [21.548, 21.812] |
    | `train/n100000_d8_k256_t5` | 95.025 ms | 37.005 ms [35.981, 38.160] |
    | `assign/direct_d8_k256` | 2.0642 ms | 824.59 us |
    | `assign/gemm_d8_k256` | 616.39 us | 621.86 us |
    | `assign/direct_d32_k256` | 6.7976 ms | 2.2089 ms |
    | `assign/gemm_d32_k256` | 825.36 us | 856.26 us |

  - The README Speed section carries these numbers, the large-k runs and the h2h grid (the gated cells), each with its command.
  - The README's old `NumpyEmeans` column (measured on the design author's data, not reproducible here) was dropped.

#### Post-audit addendum (2026-11-14): fused Gemm L2 scan

- Source: a comparison against `emeans` 0.1.1 after the implementation audit found one bit-neutral idea worth taking: score and argmin in one pass (`emeans` `knn.rs:241-315`). Today's Gemm L2 path wrote clamped scores back into the tile, then `lane_scan` re-read them. Scott: adopt before the release (option A).
- Change (`src/assign.rs`, `lane_scan_l2`): each 16-lane group's `row_norm + norm - 2*p`, clamped at `+0.0`, is staged in a local `[f32; LANES]`, then the same strict-`<` lane compare and the same merge rule run. Same per-element value, same comparisons, so change 1's argument holds unchanged. A one-loop version (compute, clamp, compare, select) de-vectorized and ran 3.6-5.6x slower; the staged form is the fast one.
- Evidence: `otto ci` exit 0; `cargo test --release --test parity` 12 passed (oracles, golden digests, determinism); `rg -c` unsafe `src/assign.rs:1`. Mutations, run once and restored: tail `<` -> `<=` fails 5 tests (`tie_in_the_tail_goes_to_the_lower` among them); lane merge `lane_id < best_id` -> `>` fails 3 (`tie_between_two_lanes_goes_to_the_lower` among them).
- Large-k (load 1.97): `train` 0.625, 0.603, 0.608 s; peak 151960, 144740, 151616 KB; digest `440bf89bb2f83d80`.
- Full `otto h2h`, load 1.98 at start, exit 0, every gated cell within its bound. Gemm cells against Phase 9: train-d32 1t 1.109 -> 1.050, 8t 1.065 -> 0.985, 64t 0.891 -> 0.926; assign-d32 1t 1.049 -> 1.074, 64t 1.252 -> 1.194; train-d8-gemm 1t 1.239 -> 1.152, 8t 1.147 -> 1.045, 64t 0.870 -> 0.830; assign-d8-gemm 1t 1.196 -> 1.099, 64t 1.201 -> 1.181. Direct cells unchanged within noise (train-d8-default 1t 1.593, 8t 1.348, 64t 0.947; assign-d8-default 1t 1.922, 64t 1.644).
- `otto bench` (load 3.74): assign gemm_d8 570.5 us (-7.5%), gemm_d32 785.7 us (-7.4%); train d32 21.47 ms, d8 36.83 ms. README Speed section replaced with this run.
- Not taken from the same comparison: `emeans`' missing NaN check, `cn - 2 x.c` scan, xoshiro RNG, `BLOCK_ROWS = 128`, BLAS features (each breaks a constraint); folding `Model::assign`'s finite scan into the row-norm loop (2-4% at 1 thread) left for later; `from_centroids` and `accumulate_labeled` stay parked.

## Acceptance Criteria

- [x] `otto ci; echo $?` prints 0 on `main`.
  - Observed on main: `otto ci` exit 0, `✅ All CI checks passed!` (clean worktree at `5f0b62d`).
  - Observed at Phase 9 (`110d271` plus the README and doc edits): `otto ci` exit 0, `[ci] ✅ All CI checks passed!`. PASS.
- [x] `cargo test --release --test parity` passes, `determinism_across_thread_counts_and_chunkings` included, and every Direct fixture matches its Phase 1 golden digest.
  - Observed on main: `test result: ok. 10 passed; 0 failed` (`5f0b62d`) (golden digests: Phase 1 added `direct_fixtures_match_their_golden_digests`, six Direct fixtures, passing).
  - Observed at Phase 9 (`110d271`, load 15.6 after the run's own builds): `cargo test --release --test parity` `test result: ok. 12 passed; 0 failed`; `determinism_across_thread_counts_and_chunkings`, `direct_fixtures_match_their_golden_digests` and `always_gemm_determinism_on_d8_fixtures_across_threads_and_chunkings` ok. PASS.
- [x] `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/` prints exactly `src/assign.rs:1`.
  - Observed on main: `src/assign.rs:1` (`5f0b62d`).
  - Observed at Phase 9 (`110d271`): `src/assign.rs:1`. PASS.
- [x] `/usr/bin/time -f "%M KB" target/release/examples/large-k` at 64 threads: peak <= 200000 KB and printed `train` time <= 0.8 s.
  - Observed on main (Phase 1, `examples/large-k.rs`): 64 threads, peak 2.20 GB (2201148 KB), `train` 4.242-4.495 s, digest `440bf89bb2f83d80`; 1 thread (`taskset -c 2`) 25.501 s, 158964 KB. Equivalent scratch harness on `bce3580`: 2.20 GB, 4.11-4.45 s process. Tiled prototype at cap 2^18: 157.7 MB, 0.63 s train (0.76-0.77 s process); at 2^16, 103 MB (single run).
  - Observed at Phase 9 (`110d271`, load 3.45 / 4.38 / 4.38 at the start of each run), `/usr/bin/time -f "%M KB" target/release/examples/large-k`, 64 threads, 3 runs: `train` 0.658, 0.642, 0.643 s; peak 152368, 150996, 149132 KB; digest `440bf89bb2f83d80` on all three. PASS (<= 200000 KB, <= 0.8 s, digest unchanged).
- [x] `otto h2h; echo $?` prints 0. Gated cells, ratio = median of 15 interleaved paired ratios ours / emeans; 8 threads unpinned:

  | cell | 1 thread (`taskset -c 2`) | 8 threads | 64 threads |
  |---|---|---|---|
  | train d=32 | <= 1.3x | <= 1.3x | <= 1.3x |
  | assign d=32 | <= 1.3x | printed | <= 1.6x |
  | train d=8, default | <= 1.75x | <= 1.8x | <= 1.3x |
  | assign d=8, default | <= 2.1x | printed | <= 1.8x |
  | train d=8, `always_gemm(true)` | <= 1.3x | <= 1.3x | <= 1.3x |
  | assign d=8, `always_gemm(true)` | <= 1.4x | printed | <= 1.6x |

  - 8-thread assign and all 32-thread cells are printed, not gated: no prototype measured them.
  - The three d=8 default bounds above were reset on 2026-11-14 (Scott, option A; see Resolved Decisions) from 1.6x / 1.35x / 1.6x, which came from prototype runs in the S5 and scratch harnesses.
  - Observed on main (Phase 1, `otto h2h`, 16 non-`always_gemm` cells in the Phase 1 table above): train d=32 1.648x (1t), 2.031x (8t), 1.554x (64t); assign d=32 1.808x (1t), 2.900x (64t); train d=8 default 4.408x (1t), 5.844x (8t), 2.243x (64t); assign d=8 default 4.442x (1t), 4.549x (64t); `always_gemm` cells `n/a`. Scratch harness, sequential not paired, `bce3580`: train d=32 1.8x (1t), 2.0x (8t), 1.7x (64t); assign d=32 1.9x (1t), 2.6x (64t); train d=8 4.9x (1t), 4.8x (8t), 2.2x (64t); assign d=8 5.2x (1t), 4.4x (64t).
  - Observed at Phase 9 (`110d271`), `otto h2h; echo $?`: exit 0, `h2h: every gated cell is within its bound`, load average 4.03 at start, 5.64 at end (the run's own rayon workers). All 24 cells, ratio = median of 15 paired ratios, ours / emeans; ms are medians. PASS, every gated cell on the first run (no cell was re-run):

    | cell | ours ms | emeans ms | ratio | bound | verdict |
    |---|---|---|---|---|---|
    | train-d32-1t | 103.15 | 93.32 | 1.109 | <=1.3 | pass |
    | train-d32-8t | 20.58 | 19.07 | 1.065 | <=1.3 | pass |
    | train-d32-32t | 16.48 | 16.26 | 1.022 | printed | |
    | train-d32-64t | 22.03 | 24.90 | 0.891 | <=1.3 | pass |
    | assign-d32-1t | 18.63 | 17.79 | 1.049 | <=1.3 | pass |
    | assign-d32-8t | 2.42 | 2.16 | 1.125 | printed | |
    | assign-d32-32t | 1.27 | 0.98 | 1.308 | printed | |
    | assign-d32-64t | 1.01 | 0.82 | 1.252 | <=1.6 | pass |
    | train-d8-default-1t | 82.98 | 51.82 | 1.605 | <=1.75 | pass |
    | train-d8-default-8t | 17.97 | 11.82 | 1.429 | <=1.8 | pass |
    | train-d8-default-32t | 27.42 | 28.45 | 0.989 | printed | |
    | train-d8-default-64t | 40.07 | 40.62 | 0.992 | <=1.3 | pass |
    | assign-d8-default-1t | 18.39 | 9.55 | 1.922 | <=2.1 | pass |
    | assign-d8-default-8t | 2.39 | 1.39 | 1.727 | printed | |
    | assign-d8-default-32t | 0.93 | 0.53 | 1.744 | printed | |
    | assign-d8-default-64t | 0.79 | 0.50 | 1.630 | <=1.8 | pass |
    | train-d8-gemm-1t | 61.10 | 49.12 | 1.239 | <=1.3 | pass |
    | train-d8-gemm-8t | 13.18 | 11.16 | 1.147 | <=1.3 | pass |
    | train-d8-gemm-32t | 24.95 | 29.11 | 0.852 | printed | |
    | train-d8-gemm-64t | 35.11 | 40.44 | 0.870 | <=1.3 | pass |
    | assign-d8-gemm-1t | 12.58 | 10.53 | 1.196 | <=1.4 | pass |
    | assign-d8-gemm-8t | 1.67 | 1.44 | 1.157 | printed | |
    | assign-d8-gemm-32t | 0.68 | 0.53 | 1.289 | printed | |
    | assign-d8-gemm-64t | 0.58 | 0.49 | 1.201 | <=1.6 | pass |

    The thinnest margin is `train-d8-default-1t` (1.605 against 1.75).
  - Prototypes (Phase 0, scratch harness, interleaved for S5): d=32 lane scan train 1.15x, assign 1.16x (1t); d=32 train 8t 1.065-1.18x (S5, forced-Gemm prototype); d=8 default train 1.44-1.48x (1t), 1.07-1.19x (8t), 0.93x (64t); d=8 forced Gemm train 0.96x (1t), 0.98-1.11x (8t), 0.755x (64t); d=8 assign 1.91x lane, 1.27x Gemm (1t), ~1.4x and ~1.1x (64t). The 64-thread assign cells are sub-millisecond, hence the looser bound.


## Resolved Decisions

Author decisions, 2026-11-13, before review:

- Bitwise, not tolerance. Every change is argued bit-neutral and tested against an oracle of today's code.
- `BLOCK_ROWS` stays 256. Changing it regroups the spherical objective's f64 adds (`trainer.rs:504`); change 7 gets finer tasks without it.
- Norms are computed per call; the lane layout per task, tiled under `TILE_FLOATS` (S2 measured one layout on the calling thread slower; panel round 1 required the cap). Neither is stored on `Model` or `Trainer`.
- `TILE_FLOATS` started at 2^16 (S3 sweep); Phase 3 re-measured against 2^18 and kept 2^18 by the "faster at 64 threads" rule (median 0.624 vs 0.630 s, 5 of 7 pairs). 2^16 was ~11% faster at 1 thread and ~35% lighter (101 vs 155 MB); both pass the gate.
- `train` keeps the per-batch finite scan; no up-front scan of all `x`.
- API items from the comparison parked: nobody asked for them.
- `v0.1.0` (`5f0b62d`) is the released baseline; this work ships as the next version.
- `bin/h2h` is a committed tool (owner rule: a harness worth keeping lives in `bin/`, never `docs/`), not a recorded observation: Scott's goal is the ratio, so the ratio is the gate.

Author decisions after S5, 2026-11-13:

- Fused indexed region (change 4) replaces the first draft's separate parallel gather and scan: separate regions measured 4x slower at 64 threads. This reverses the first draft's rejection of an indexed accumulate, which was rejected for adding an entry point; the entry point is crate-private, and the alternative's measured cost decides it.
- Sub-block tasks reinstated as change 7: S5 supersedes S4. S5's `attrib-1` shows 64-row tasks alone, no fusion, take the pinned 8t ratio 2.40 -> 1.58; S4 saw only noise at 8t. Why S4 missed it is not explained.
- Permutation overlap and the pool-thread loop added as change 8: 1.6 -> 0.3 ms and 2.2 -> 0.06 ms per train.

Scott, 2026-11-13:

- d=8 gap after the lane kernel (1.7x at 1 thread, ~2x at 8, ~1.1x at 64): option B, a builder opt-in `always_gemm(bool)` forcing Gemm below the threshold. Default stays Direct. Rejected: accepting the gap with no opt-in.
- 8-thread gap: option B, root-cause it with a Phase 0 spike (S5) before gating, then gate 8 threads at a real target. Rejected: gating 8 threads at the measured bounds as-is. Done: S5 above, 8-thread train cells gated.

Scott, 2026-11-14, after the Phase 7 gate missed:

- d=8 default bounds: option A, reset them from `bin/h2h` measurements. Rejected: B (reopen the lane Direct kernel non-goal; Phase 0 found no bit-neutral lever left without FMA or intrinsics) and C (ship with `otto h2h` exiting 1).
  - Why the old bounds were a doc defect, not a code miss: the S5 prototype the 1.35x bound came from measures 1.351-1.541 (median ~1.42) in `bin/h2h`, against 1.12-1.20 in its own harness the same hour. The S5 harness called emeans from the main thread, so emeans paid a pool handoff on each of its 490 regions; `bin/h2h` runs both crates inside `pool.install` (Phase 7 record).
  - The remaining gap is the kernel choice: default Direct runs the difference form, three ops per coordinate and no FMA, for the offset and `oversplit` results; emeans always runs sgemm plus an AVX `norm - 2*dot` scan (`knn.rs:204-230,266-268`). `always_gemm` is the speed-parity path.
  - New bounds, ~8% over the worst observed `bin/h2h` value (the headroom the original bounds took over their prototypes): train d=8 8t 1.35 -> 1.8 (20 runs, 1.324-1.666); train d=8 1t 1.6 -> 1.75 (1.600-1.618, Phases 6-7); assign d=8 64t 1.6 -> 1.8 (1.578-1.665, Phases 6-7).

Review panel round 1, 2026-11-13 (Gemini Architect, Codex Staff Engineer; synthesis `~/.cache/review-panel/runs/D50dtLxQ/synthesis.md`). Change and phase numbers below are current. Every finding and its disposition:

| # | finding | raised by | disposition |
|---|---|---|---|
| 1 | Per-task Direct layout is O(k*d), contradicts the k-independent scratch goal; reachable at any d, k via `batch_size` | both | Folded. Layout tiled under `TILE_FLOATS`; Phase 6 scratch-bound test |
| 2 | emeans ratio gate not reproducible: sequential runs, unequal assign closures, ratio undefined, prototype crates linked | both | Folded. `bin/h2h` committed, interleaved pairs, median of paired ratios, exit 1 on miss |
| 3 | `always_gemm` measured before it existed; "recorded" accepts a failure | Staff | Folded. Opt-in is Phase 8, measurement Phase 9, criterion "passes" |
| 4 | Large-k gate timed the process (0.76-0.77 s) against train-time margin (0.63 s); data generator differs | Staff | Folded. Example prints `train` time; Phase 1 records the baseline on its own data |
| 5 | `rg -l unsafe` counts files, not uses | Staff | Folded. `rg -c '\bunsafe (\{|fn|impl|trait|extern)'`, observed `src/assign.rs:1` |
| 6 | Change 5 lacks partition policy, perf gate; threshold on rows only | Staff | Folded. P = thread count, threshold on rows*d, Phase 5 bench non-regression |
| 7 | Summary overstated Phase 0 coverage | Staff | Folded |
| 8 | `=0.3.11` pin and S1 test had no owning phase | Staff | Folded into Phase 3 |
| 9 | `Model::assign` first-offender test; README "equal to a fresh bench" unmeetable | Staff | Folded. Phase 4 test; Phase 9 README criterion |
| 10 | Ratio gate omits d=8 assign, 64t assign, middle thread counts | Staff | Folded: d=8 and 64t assign cells gated; 8-thread train gated after S5 (Scott, option B); 8-thread assign and 32 threads printed, unmeasured |
| 11 | Release-tag consumer smoke test | Staff | Agreed with the panel: no consumers, nothing added |
| 12 | Change 5 needs in-place adds, block/carry order, `2*rows` order; test vs a frozen oracle | both | Folded into change 5 and Phase 5 |
| 13 | Why expose the `always_gemm` footgun? | Architect | Declined: Scott ruled option B, 2026-11-13 (above); default stays Direct, README names the limit |
| 14 | Phase 0 not falsifiable, logs discarded | Architect | Declined, with the panel: logs exist and Staff re-ran S1 (0 of 360). The durable part is now `bin/h2h` and the S1 unit test |
| 15 | Rollback if a compiler or microcode change alters dispatch | Architect | Declined, with the panel: exact pin plus S1 unit test; Gemm bits already CPU-specific (Risks) |
| 16 | Tiling the Direct layout also fixes the cross-core slowdown | Architect | Not claimed: unmeasured. Phase 6 measures d=8 on the tiled layout |
| 17 | Which gate catches a release that bloats Direct memory or moves Gemm training bits? | Staff | Folded: Phase 6 scratch-bound test; Gemm bits by the oracle tests, `train_matches_caller_loop_in_pools` and the large-k digest (same machine) |

Review panel round 2, 2026-11-13, scoped to changes 4, 7, 8 (same synthesis file, round 2 section). Every finding and its disposition:

| # | finding | raised by | disposition |
|---|---|---|---|
| R2-C2 | `Model::assign`'s fused scan was never prototyped; the S5 claim covers `train` only | both | Folded. Overview table says so; Phase 4 tests are its evidence |
| R2-C3 | Thread-count split rule acceptable; no simpler rule fits (always-64-row refuted at 64t) | both | Agreed, no change |
| R2-M1 | Doc dropped the L2-only guard on the snapshot copy; an unconditional copy moves spherical bits | Staff | Folded. Pseudocode and change-4 error argument, per metric |
| R2-M2 | No test runs `train()` in spherical mode, in pools, across multi-task batches | Staff | Folded. `train_matches_caller_loop_in_pools` in Phase 1 |
| R2-M3 | Pseudocode drew `order_{p+1}` after the last pass | panel, Staff | Folded. Guard and `scatter_labels` in the pseudocode |
| R2-W1 | S5 table quoted a non-final run; pinned baseline bimodal | panel | Partly folded: bimodal baseline and all three runs recorded. Pushed back with log headers: `pairs-2`/`pairs-3` carry `EM_SNAPFIRST=1`, `pairs-1` does not; the table quoted the final config. Panel agreed and corrected its synthesis |
| R2-W2 | S4 reversal reason wrong: 64-row tasks help without fusion too | Staff, panel | Folded. "S5 supersedes S4", cause of S4's null unexplained |
| R2-W3 | S5 used a shared untiled lane layout, not the design's per-task tiled one | Staff | Folded. Noted under S5; Phase 7 `otto h2h --cell` is the gate |
| R2-W4 | Phase 1 h2h cannot print `always_gemm` cells; Phase 7 vs Phase 9 assertions mixed | Staff | Folded. `n/a` until Phase 8; `--cell` for single-cell gates |
| R2-W5 | Phase 5 baseline "Phase 4 run" never recorded; Phase 4 has no perf check | panel | Folded. Phase 4 records before and after, gates on `otto bench` |
| R2-W6 | Split-rule test misses boundaries | Staff | Folded. 32/33 threads, blocks == threads, partial last block |
| R2-W7 | NaN test lacks teeth; positions ambiguous under permutation | Staff, panel | Folded. Batch positions, different tasks, `fetch_min` mutation, short final task for `Model::assign` |
| R2-D1 | An early error waits for the speculative permutation | Staff | Folded as documented behavior under change 8 |
| R2-D2 | `train()` nested inside a caller's pool untested | panel | Folded. `train_matches_caller_loop_in_pools` runs inside explicit pools |
| R2-A | Architect: every item fine, evidence "absolutely sufficient" | Architect | Superseded by the verified findings above |

No finding is open or disputed after round 2.

## Alternatives Considered

### Alternative 1: Copy emeans' kernel wholesale
- **Description:** its `knn.rs`: `cn - 2 x.c` scan, AVX2/AVX-512 argmin, always Gemm.
- **Pros:** proven speed.
- **Cons:** moves near-tie labels; loses the offset fix and `oversplit` parity; 2+ new `unsafe` sites.
- **Why not chosen:** breaks three constraints.

### Alternative 2: `BLOCK_ROWS = 128`
- **Description:** emeans' block size.
- **Cons:** regroups the spherical objective's f64 adds; result bits change against `v0.1.0`.
- **Why not chosen:** change 7 gets finer tasks with zero bit change.

### Alternative 3: Thread-local `rows x k` tile
- **Pros:** measured 4.26 -> 1.69 s at large k.
- **Cons:** RSS rose to 4.16 GB; `thread_local!` state.
- **Why not chosen:** the cap is the fix; tiling hit 0.63 s at 158 MB.

### Alternative 4: Separate parallel gather and finite-scan regions
- **Description:** the first draft's version of change 4: `par_chunks_mut` gather and `position_first` scan, each its own region; no new entry point.
- **Pros:** no new trainer entry point.
- **Cons:** three regions per batch; S5 measured lane d=8 39.3 -> 57.7 ms (8t) and 48.2 -> 206.9 ms (64t).
- **Why not chosen:** change 4 removes the same serial cost inside the one existing region.

### Alternative 5: Lane layout built once per call
- **Description:** one layout on the calling thread, shared by all tasks.
- **Why not chosen:** measured slower at 64 threads (2.59 vs 1.91 s, d=16, k=32768) and at 1 thread unpinned (134 vs 100 ms); no difference pinned to one CPU. Also O(k*d), which panel round 1 ruled out.

## Addendum: roads not taken

- Reversed from the first doc's Phase 9 tradeoffs: "tile per block call vs thread-local pool buys nothing" and "centroid norms recomputed per block vs cached". Both were judged at k=256; the large-k baseline (2.20 GB, 4.26 s) reverses them.
- AVX-512/AVX2 multiversioned argmin: no gain (108 vs 97 ms), 2 `unsafe` sites. Do not re-add.
- Row register blocking in lane Direct (2 or 4 rows per centroid-block load): bitwise, no gain (S2). Assembly not inspected.
- 8 lanes instead of 16: never faster (S2).
- A task split at 64 threads: more tasks cost more in wake-up and join than they save (S5, unfused: 4 tasks 48.2 ms, 8 tasks 52.8, 16 tasks 74.8). `SPLIT_MAX_THREADS` encodes it.

## Technical Considerations

### Dependencies

- None new in the library. `rayon` already has `for_each_init`, `scope` and `join`.
- `bin/h2h` only: `emeans` by git rev `5b283a9` (on upstream `main`) with default features, which are `parallel` only: its `mkl` and `accelerate` link features stay off, so its `build.rs` links nothing. Plus `e-means` by path. Both compile under `bin/h2h`'s one release profile, so neither side gets a profile advantage. Never a dependency of the library or its tests.
- Pin `matrixmultiply = "=0.3.11"`: change 2's bit-neutrality rests on its accumulation order. S1 becomes a unit test over tile widths, so a deliberate bump re-proves it.

### Performance

- Scratch per task: `max(TILE_FLOATS, LANES*d)` f32 for both kernels, independent of k. Without the Direct tile it would be k*d per task (262 MB per task at d=1000, k=65536, reachable through `batch_size`); today's Direct allocates nothing, so Phase 6 gates the bound.
- Parallel regions per batch: one, as today; two when change 5's threshold is crossed, gated by Phase 5's bench.
- Targets in Acceptance Criteria. Measurement: paired head-to-head runs, median of 15, 1-thread runs under `taskset -c 2`.

### Security

- No new `unsafe`; the one `sgemm` call and its asserts are unchanged.

### Testing Strategy

- Oracle tests: today's scalar code under `#[cfg(test)]`, every new path compared with `to_bits`.
- Golden digests for Direct fixtures, portable across x86-64 runners.
- Train-vs-caller-loop tests pin the fused `train()` path to the public `accumulate` path.
- Tests bite: each phase mutates one comparison once and confirms a named test fails.
- Existing parity, determinism and kernel-switch tests run unchanged.

### Rollout Plan

- Blast radius: none outside the repo. No consumers.
- Ship: Phases 1 -> 9 in order, one release at the end by `bump release`. Phase 8 adds public API: the next version is a minor bump.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A `matrixmultiply` change alters per-element accumulation with size | Low | High | exact pin; S1 kept as a unit test |
| Autovectorization silently fails in a later rustc | Med | Med | `otto h2h` gate in Phase 9; fixed-width arrays the compiler vectorizes on SSE2 (S2, S3) |
| `TASK_ROWS` / `SPLIT_MAX_THREADS` tuned on one 64-thread machine misfit other core counts | Med | Low | named consts; per-row results never depend on them, so a retune changes speed only; `otto h2h` 8-thread cells catch a regression here |
| The fused region's error path drifts from `accumulate`'s | Low | Med | same message text asserted in Phase 4; train-vs-caller tests |
| Cluster-partitioned reduction's extra region costs more than it saves | Med | Low | serial below `PARALLEL_REDUCE_VALUES` (rows*d), set and gated by Phase 5 benches |
| Gemm results differ across CPUs (AVX-512 vs FMA) | High | Low | already true; the determinism claim is same-machine; golden digests cover Direct only |
| 1-thread numbers swing ~30% with core placement | High | Low | `taskset -c 2` for every 1-thread measurement |

## Open Questions

None.

## References

- First design: `docs/plan/2026-11-13-e-means.md`; implementation notes `docs/plan/2026-11-13-e-means-implementation-notes.md`
- Handoff: `docs/handoff/emeans-performance-parity.md`
- emeans: https://github.com/savannahar68/emeans at `5b283a9` (`src/knn.rs`, `src/emeans.rs`)
- matrixmultiply 0.3.11 source: `gemm.rs`, `sgemm_kernel.rs`, `archparam_defaults.rs`
