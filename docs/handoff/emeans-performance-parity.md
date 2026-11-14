# Handoff: close the performance gap to the `emeans` crate

Branch: main

## Next action

Run `/plan:create` for a design doc, `docs/plan/<date>-assign-performance.md`. Its goal: bring `e-means` training and assignment speed, and its memory at large k, close to `emeans` 0.1.1, while keeping every correctness property listed under "Constraints". Do not change code before that doc passes `/plan:review`. These are kernel design changes, and Scott chose to route them through the pipeline (option A, 2026-11-13).

## State at handoff (2026-11-13)

- A `bump release --no-install` for v0.1.0 was running in the background when this was written. `Cargo.toml` is `0.1.0` and untagged, so `bump` releases that version as is (no bump): it pushes main, waits for CI (`ci` and `mutants` workflows), tags `v0.1.0`.
  - Check: `git -C ~/repos/scottidler/e-means ls-remote --tags origin v0.1.0` (one line = released), and `git -C ~/repos/scottidler/e-means log --oneline -1 origin/main`.
  - If the tag is missing, the release did not finish. Re-run `bump release --no-install ~/repos/scottidler/e-means` and read its refusal text; never tag by hand.
  - Still open from the original build: Phase 10 criterion "a scratch crate with `e-means = { git = ..., tag = "v0.1.0" }` builds". Run it once the tag exists.
- Commits that matter (all on main):
  - `74c01bb`..`bce3580`: Phases 1-10 of `docs/plan/2026-11-13-e-means.md`.
  - `863816f`: the Gemm kernel, `BLAS_THRESHOLD`, criterion benches. The code this optimization targets.
  - `035104d`: implementation-audit fixes (f64 norm in `normalize_rows`, `Trainer::current_centroids`, re-measured README).
  - `7088fe0` and the commit that adds this handoff: README install snippet set to v0.1.1, then back to v0.1.0 to match what `bump` tags; this handoff.

## Read first, in order

1. `docs/plan/2026-11-13-e-means.md`: sections "Reduction order", "Kernel choice", "Trainer error contract", Phase 8 and 9, Acceptance Criteria, Addendum (roads not taken: hand-written SIMD scan, parallel partial sums).
2. `docs/plan/2026-11-13-e-means-implementation-notes.md`: Phases 7-9 and the audit-fix section.
3. `src/assign.rs`: `nearest` (:47), `gemm_nearest` (:80, tile alloc :85, scalar argmin scan :94-110, centroid norms recomputed per block :94), `inner_products` (the one `unsafe`), `direct_l2` (:179), `direct_inner_product` (:196).
4. `src/trainer.rs`: `BLOCK_ROWS = 256` (:23), `accumulate` (:185, finite scan :206, snapshot copy :218, carry :230/:255), `assign_blocks` (:471), the row-order reduction after it.
5. `src/config.rs`: `train` pass loop (:249-251) and `gather_into` (:288): a serial copy of each batch before assignment.
6. The `emeans` clone, `~/repos/savannahar68/emeans` at `5b283a92696e0c0a8519f57bd3f8917fd4f8f31c`: `src/knn.rs` (`BLOCK_ROWS = 128` :43, `TILE_FLOATS = 1 << 18` :45, centroid tiling :70/:147, `for_each_init` per-thread scratch :92/:109-114, lane-vectorized argmin :180-315, indexed gather inside blocks :19-40/:131-139), `src/emeans.rs` (centroid norms cached per batch :511, indexed rows :299-309).
7. `benches/train.rs`, `benches/assign.rs`, `benches/common/mod.rs`: the in-repo benches (`otto bench`).

## Timing data (measured 2026-11-13)

Machine: Threadripper 9970X, 64 threads, rustc 1.99.1, release build. Load average was 6-10 during these runs, so read them as ratios. `ours` = `bce3580`; `src/assign.rs` and the trainer hot path are unchanged since then. Data: the `tests/quality.rs` mixture, n=100000, k=256, T=5. Both crates resolve the same recipe: d=32 -> B=8192, alpha0=1 (throughput law, `B*d` = 262144, ours runs Gemm); d=8 -> B=1024, alpha0=0.155434 (quality law, `B*d` = 8192, ours runs Direct).

`train` wall time, median of 15 after one warm-up:

| threads | d=32 ours | d=32 emeans | d=8 ours | d=8 emeans |
|---|---|---|---|---|
| 64 | 41.2 ms | 24.4 ms | 97.6 ms | 43.9 ms |
| 32 | 33.4 ms | 16.0 ms | 100.8 ms | 32.7 ms |
| 8 | 38.6 ms | 18.9 ms | 80.9 ms | 16.7 ms |
| 1 | 151.9 ms | 84.5 ms | 255.5 ms | 52.4 ms |

`Model::assign`, 100000 rows:

| | d=32 ours | d=32 emeans | d=8 ours | d=8 emeans |
|---|---|---|---|---|
| 1 thread | 27.4 ms | 14.8 ms | 44.1 ms | 8.4 ms |
| 64 threads | 1.8 ms | 0.7 ms | 2.2 ms | 0.5 ms |

Large k (n=200000, d=32, k=65536, T=1, 64 threads): ours 4.26 s and 2.20 GB peak RSS; emeans 0.43 s and 158 MB. At 1 thread: 25.3 s against 7.4 s.

Thin LTO with one codegen unit (the `emeans` release profile) does not change the ratios: 42.2 against 25.4 ms (d=32) and 103.4 against 45.4 ms (d=8) at 64 threads.

In-repo criterion medians from Phase 9 (`cargo bench`), for regression tracking: train d=32 44.4 ms, d=8 99.9 ms; assign 100000 rows k=256: d=8 Direct 2.22 ms / Gemm 1.21 ms, d=32 Direct 7.46 ms / Gemm 2.01 ms.

## Root causes found (instrumented copy of ours; `perf` is blocked, `perf_event_paranoid=4`)

1. **Scalar argmin scan** in `gemm_nearest` (`src/assign.rs:94-110`). A two-pass 16-lane scan gave bitwise-identical centroids and labels, with no new `unsafe`. d=32, 1 thread: train 166 -> 97 ms, assign 29.9 -> 17.1 ms. AVX-512/AVX2 multiversioning on top gained nothing (108 ms) and adds 2 `unsafe` sites: skip it.
2. **Direct kernel at small `B*d`.** d=8, 1 thread: forcing Gemm takes train from 294 to 122 ms. With the lane scan and tile reuse: 66.6 ms (emeans 52.4). At 64 threads: 102.5 -> 40.2 ms. At k=32768, d=16, Direct is about 9.5x slower than emeans (6.36 s against 0.67 s).
3. **Serial sections at 64 threads, d=32** (of 42.8 ms per train): gather 8.5 ms (`src/config.rs:251`), finite scan 4.4 ms (`src/trainer.rs:206`), row-order reduction 5.8 ms, permutations 2.1 ms, parallel assign 21.1 ms.
4. **Small batches cap parallelism at d=8.** B=1024 gives 4 blocks of 256 rows per batch, over 490 batches. emeans uses 128-row blocks.
5. **Unbounded tile memory.** `gemm_nearest` allocates `rows x k` floats per block (`src/assign.rs:85`): 64 MB each at k=65536, about 32 at once. A thread-local tile alone cut wall time to 1.69 s but raised RSS to 4.16 GB, so the tile also needs a cap. emeans caps scratch at 2^18 floats and tiles over centroids.

## Constraints the plan must keep (each has a test that will catch a break)

- **Bitwise determinism across thread counts and chunkings:** `cargo test --test parity determinism`. Labels and distances of a block must depend only on its rows and the snapshot, and the row-order reduction must stay on one thread. See "Reduction order" and the Addendum's rejected parallel partial sums.
- **Faiss parity bounds:** `cargo test --test parity` (8 fixtures; separated fixtures within 1e-5 centroid, 1e-6 objective, 100% labels).
- **Direct kernel below `BLAS_THRESHOLD`.** It is the reason ours matches Faiss on `oversplit` (emeans, always on the expansion, gets 3.855e-2 / 1.925e-4 / 99.0%) and survives offset data: +10000 offset at d=8, held-out MSE 8.54 for ours against 43.68 for emeans. The d=8 speed gap is a real tradeoff against this, and Scott decides it. The options measured so far: vectorize Direct (not measured), or a caller-selectable kernel.
- **Exactly one `unsafe` in `src/`:** `rg -l '\bunsafe\b' src/` prints `src/assign.rs`.
- **NaN/inf rejection** stays (the finite scan can move into the parallel blocks, but it must still run before anything changes; see "Trainer error contract").
- No BLAS link features, no `build.rs`.

## Also from the comparison (API, smaller)

- `Model::from_centroids` (emeans `src/emeans.rs:808`): today a caller cannot assign with saved centroids.
- A recipe preview without an init (emeans `Emeans::recipe(n)`, `src/emeans.rs:172`).
- An optional `parallel` feature, which their CI tests with `--no-default-features`.

## Session-scoped, may be gone

- The head-to-head harness and the instrumented variant lived in the session scratchpad, which a reboot clears: `/media/saidler/4tb/claude-tmp/claude-1000/-tmp/d64bd27e-03c5-4347-be32-78f91a84f54a/scratchpad/h2h` (`src/main.rs`, `src/fixtures.rs`; subcommands `fixtures`, `determinism`, `paired`, `quality`, `timing`, `edge`, `variant`, `bigk`) and `.../scratchpad/ours-variant` (switches via `EM_VARIANT`). Check with `ls`. If gone, rebuild a harness that depends on both crates by path, and reuse `benches/common/mod.rs` for data.
- Commands used: `RAYON_NUM_THREADS={64,32,8,1} h2h timing {32,8} 15`; `/usr/bin/time -f "%M KB" h2h bigk {ours,emeans} 200000 32 65536`.

## Suggested skills

- `/plan:create`: the design doc (Rule of Five). Put the timing tables above into it as the baseline.
- `/plan:review`: panel review before building.
- `/plan:build`: execute it; per-phase `otto ci`.
- `/rust-cli-coder`: house Rust conventions.
- `/otto`: `otto ci`, `otto bench`.
- `/plan:release`: ship it after the build's finalization checkpoint.
