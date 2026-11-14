# Implementation notes: assignment and training performance

## Phase 1: Guard tests, the large-k example, `bin/h2h`

### Design decisions
- Oracles `oracle_direct_l2`, `oracle_direct_inner_product`, `oracle_gemm_nearest` copied verbatim into `src/assign/tests.rs`, plus `oracle_nearest` / `production_nearest` wrappers returning labels and `to_bits` of scores. Every comparison is `to_bits`, never a tolerance.
- Tie fixtures share one builder, `tie_fixture(k, tied)` (d=2, row `(1,0)`, tied centroids at `(3,0)`, the rest far at `(-5-j, 0)`): small integers, so Direct and Gemm score exactly, and the tied ids are the unique L2 minimum and the unique inner-product maximum. One fixture serves all four (kernel, metric) pairs, and each asserts the oracle's labels as well as `nearest == oracle`, so a mutated oracle fails on its own labels.
- Golden digests are three FNV-1a hashes per fixture (centroid f32 bits, last-pass labels, per-pass objective f64 bits) in `tests/parity.rs`, reusing its `fnv1a*` helpers and `rust_run`. The test also asserts the manifest's Direct fixtures (`batch_size * d < 128000`) equal the golden table's names, so a new Direct fixture cannot go unguarded. Captured from the unchanged code; identical on repeat runs and at `RAYON_NUM_THREADS=1`.
- `train_matches_caller_loop_in_pools` reuses the module's `caller_run`, `rows_of`, `row_order` and `unit_rows`, and asserts each case's `B * d` picks the kernel its name says, and that `n % B != 0`.
- `benches/common/mod.rs` gained `mixture_of(n, components, d, seed)`; `mixture(d, seed)` now calls it with `(N, K)`, so existing bench data is unchanged. `examples/large-k.rs` includes the module through `#[path]` with `#[allow(dead_code)]` on the `mod`.
- `bin/h2h` re-runs each cell in a child process (`taskset -c 2` for 1 thread) and parses a `RESULT` line, so `otto h2h` pins the 1-thread cells itself, and one process never mixes pinned and unpinned cells. It asserts both crates resolved the same batch size and alpha0 before timing.
- The grid is one const table (`GRID`, label + four bounds) in `bin/h2h/src/main.rs`; 32-thread cells are in it with no bound (printed), as the Acceptance Criteria say.
- `assign` cells assign with a T=5 model from each crate. Recipe batch size depends on the epoch budget: a T=1 model resolves d=32 to Direct at ours (30 ms became 171 ms in a first run), not the Gemm the grid and the doc's baselines mean.
- Asking `--cell` for an `always_gemm` cell exits 1 ("cannot run yet"): a gate that cannot be checked is not a pass. An unknown cell name exits 2 with the list.
- `bin/h2h/Cargo.toml` has an empty `[workspace]` table so it is its own root; `bin/h2h/Cargo.lock` is committed (binary crate); `target/` is already covered by `.gitignore`.

### Deviations
- `otto h2h --cell` takes one value (otto's param is a single string): several cells go in one quoted string, `--cell "a b"`, which the task word-splits into `h2h --cell a b`. The h2h binary itself takes space-separated names.
- Direct golden fixtures: the doc's Phase 1 text implies the `B*d`-large fixtures are the only Gemm ones; by `batch_size * d < 128000` six of eight fixtures are Direct (all but `large` and `large_revival`), so six are pinned, including `oversplit`, `separated_d32` and `separated_spherical_d32`, not only the d=8 ones.
- Pair order in `bin/h2h`: ours then emeans in every pair, as the doc says, rather than alternating who goes first.

### Tradeoffs
- Child process per cell vs pinning the thread with `sched_setaffinity`: the child costs a data regeneration per cell (about 0.1 s) but needs no `libc` or `unsafe`, and `taskset` is the command the doc names.
- Golden digests as hashes vs committing centroid and label arrays: hashes are 6 short lines; the arrays are megabytes and a mismatch is equally fatal either way (the parity fixtures already hold the inputs).

### Open questions
- None.

#### Recorded observations (unchanged kernels)
- `otto ci` exit 0, `All CI checks passed!`; 129 lib tests, 11 parity, 4 quality, 1 doctest.
- Oracle mutation, run once and restored: `<` to `<=` in `oracle_direct_l2` (line 300) or in `oracle_gemm_nearest`'s L2 scan (line 356) fails `tie_between_two_centroids_in_one_lane_goes_to_the_lower`, `tie_between_two_lanes_goes_to_the_lower` and `tie_in_the_tail_goes_to_the_lower`.
- `examples/large-k.rs`, 64 threads, 3 runs: `train` 4.242, 4.370, 4.495 s; peak RSS 2201148, 2201444, 2200412 KB; digest `440bf89bb2f83d80`. 1 thread (`taskset -c 2`): `train` 25.501 s, 158964 KB, same digest.
- `otto h2h` exit 1 (all 10 gated cells miss): train d=32 1.648x / 2.031x / 1.554x (1t / 8t / 64t); assign d=32 1.808x (1t), 2.900x (64t); train d=8 default 4.408x / 5.844x / 2.243x; assign d=8 default 4.442x (1t), 4.549x (64t); 8 `always_gemm` cells `n/a`. Full table in the design doc, Phase 1.

## Phase 2: Lane argmin and norms once per call

### Design decisions
- `lane_scan(scores, worst, better)` in `src/assign.rs`: one generic scan for both metrics (`<` with `+inf` for L2, `>` with `-inf` for inner product), so the tie rule lives in one place. Ids are `u32` (labels are `u32`); `LANES = 16` is a module const. The doc's `v < INF` guard is `v != worst` here: the same guard for L2, mirrored for IP.
- Pass 1 is in place: `gemm_nearest` overwrites each tile row's inner products with the clamped L2 score (`row_norm + centroid_norm - 2.0 * ip`, then `< 0.0 -> 0.0`), so there is no second buffer. Same expression and operand order as the oracle.
- `centroid_norms(kernel, metric, centroids, d)` in `src/assign.rs` returns `||c||^2` for Gemm + L2 and an empty `Vec` otherwise. `Model::assign` and `Trainer::assign_blocks` call it once before their parallel region (the trainer from `snapshot`) and share `&norms` across tasks. `nearest` asserts `norms.len() == k` for Gemm + L2, so a missing slice fails loudly instead of scoring against zero norms.
- `scores.as_chunks::<LANES>()` instead of `chunks_exact(LANES)`: clippy (`chunks_exact_to_as_chunks`, `-D warnings`) requires it for a constant chunk size.
- New tests: `lane_scan_equals_a_sequential_scan_bitwise_on_tie_heavy_data` (lengths 0..=70 and 255-257, four-valued data, NaN, all-`+inf`, all-`-inf`, both orders), `lane_scan_of_nothing_is_id_zero_at_the_identity`, `centroid_norms_exist_only_for_gemm_l2`, `gemm_l2_rejects_missing_norms`. The Phase 1 oracle and tie tests are unchanged except for the new `norms` argument.

### Deviations
- `nearest` carries `#[allow(clippy::too_many_arguments)]`: 8 parameters against clippy's limit of 7, and the doc's final signature has 9 with scratch. The doc does not mention the lint; the allow follows the doc's own seam and its comment says Phase 3 adds the scratch.
- `cargo test oracle` matches only 2 tests by name (substring filter). The tie-fixture and golden-digest tests that exercise the oracles do not carry "oracle" in their names; all of them ran under `otto ci` and `cargo test --lib`.

### Tradeoffs
- In-place pass 1 vs a separate score buffer: no extra allocation, and the tile's inner products are dead after pass 1.
- One generic `lane_scan` with a closure vs two hand-written scans: the closure inlines, and one function cannot drift between metrics.

### Open questions
- None.

#### Recorded observations
- Mutation, run once and restored: lane-merge tiebreak `lane_id[l] < best_id` changed to `>` fails `tie_between_two_lanes_goes_to_the_lower`, `a_row_equal_to_every_centroid_takes_index_zero_with_a_positive_zero_score` and `lane_scan_equals_a_sequential_scan_bitwise_on_tie_heavy_data`.
- `cargo test --test parity`: 11 passed, `determinism_across_thread_counts_and_chunkings` ok (the Direct golden digests included). `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`. No `target_feature`, no `mul_add`.

## Phase 3: Centroid tiling under TILE_FLOATS

### Design decisions
- `TILE_FLOATS = 1 << 18` in `src/assign.rs` (1 MB of f32 per task): the faster cap at 64 threads in the re-measure (median 0.624 vs 0.630 s train, 7 interleaved pairs; numbers in the design doc, Phase 3).
- `nearest` gains `scratch: &mut Vec<f32>` (the doc's signature) and forwards to the private `nearest_capped(.., tile_floats)`; tests call `nearest_capped` to force small tiles, production passes `TILE_FLOATS`.
- `gemm_nearest` tiles over centroids with `width = (tile_floats / rows).clamp(1, k)`, the last tile shorter. Each tile runs `inner_products` on `centroids[j0*d..j1*d]` into `scratch[..rows*width]`, pass 1 in place with `norms[j0..j1]`, then `lane_scan`; the tile's best replaces the row's running best only on the strict order (`<` L2, `>` IP). Labels start at 0 and scores at the identity, so an all-`inf` or NaN row keeps label 0 and the identity, as the oracle does.
- Row norms are computed once per call into a `rows`-long `Vec`, not per tile, so a large k does not repeat them per tile.
- `Trainer::assign_blocks` and `Model::assign` switch `for_each` to `for_each_init(Vec::new, ..)`: one scratch per rayon leaf job, no `thread_local!`. Direct ignores the scratch until Phase 6.
- `Cargo.toml`: `matrixmultiply = "=0.3.11"`.
- New tests in `src/assign/tests.rs`: `tiled_gemm_equals_the_oracle_bitwise_with_five_or_more_tiles` (d in {1, 3, 8, 17, 32}, k in {17, 33, 100, 256}, n in {1, 7, 60}, widths giving 5+ tiles, caps at and just under a width boundary, cap 0, one dirty scratch reused across every call), `tiled_tie_straddling_a_tile_boundary_goes_to_the_lower_as_in_the_oracle` (k=40 at width 8: ties at 7/8, 15/16, 31/32, 23/24/25), `tiled_all_infinite_and_nan_rows_match_the_oracle`, `gemm_scratch_stays_under_the_tile_cap_whatever_k_is`, `gemm_element_is_independent_of_tile_and_split` (the doc's grid, d in {1, 8, 32, 300}, k in {17, 256}, widths {1, 15, 16, 17}, splits {1, 7, 64} over 71 rows so every split has a short chunk). The first three carry `oracle` in their names, so `cargo test oracle` runs them (5 tests now).

### Deviations
- The tiled-oracle test's widths are the doc's {1, 15, 16, 17} plus {2, 3, k/5}, filtered to those leaving 5+ tiles: at k=17 only widths 1-3 give 5 tiles, so {15, 16, 17} alone cannot meet the "5+ tiles" criterion there. The S1 unit test keeps the doc's exact grid (tile count is not its point).
- Row norms live in a per-call `Vec` beside `scratch`, not inside it: same effect, one small allocation (`rows` f32) per Gemm L2 block call, matching the per-job allocation count the doc already accepts.

### Tradeoffs
- 2^18 vs 2^16: 2^18 is about 1% faster at 64 threads (the doc's deciding cell) but uses about 54 MB more peak RSS at k=65536 (155 vs 101 MB) and is about 11% slower at 1 thread (9.0 vs 8.0 s). The doc's rule picks 64 threads; recorded as an open question.
- Running best written straight into `labels`/`scores` vs a separate best buffer: no extra memory; the outputs are fully overwritten at the start of each call.

### Open questions
- Confirm 2^18 over 2^16: the 64-thread gain (6 ms median, 5 of 7 pairs) is near noise, while 2^16 is about 11% faster at 1 thread and 35% lighter at 64 threads. I followed the doc's rule ("keep the faster at 64 threads").

#### Recorded observations
- Mutation, run once and restored: the cross-tile merge `best < *distance` changed to `<=` fails `tiled_tie_straddling_a_tile_boundary_goes_to_the_lower_as_in_the_oracle` and `tiled_all_infinite_and_nan_rows_match_the_oracle`.
- `otto ci` exit 0; 138 lib tests, 11 parity, 4 quality, 1 doctest. `cargo test --release --test parity`: 11 passed, determinism and golden digests ok. `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`; no `target_feature`, `mul_add` or `thread_local`.
- Large-k gate (2^18), 64 threads: `train` 0.669, 0.630, 0.621 s; peak 154548, 154752, 157096 KB; digest `440bf89bb2f83d80`.

## Phase 4: Fused train region

### Design decisions
- `Trainer::accumulate_indexed(x, ids)` (`src/trainer.rs`, `pub(crate)`): one `par_chunks_mut(BLOCK_ROWS * d)` region over the batch buffer zipped with `ids`, labels and scores, `enumerate`d for the block index, `for_each_init(Vec::new, ..)` for the kernel scratch. Each task gathers its block's rows, scans them with `position(!is_finite)`, and on a hit does `fetch_min` of the batch-wide value offset into an `AtomicUsize` and skips `nearest`. After the join a hit becomes `NonFiniteInput("chunk row R column C is V")`, the exact text `accumulate` gives for the gathered batch. Blocks stay one task per 256-row block.
- Whole batches only: `accumulate_indexed` `assert!`s `received() == 0` and `ids.len() == due`. They are crate-private invariants `train` always meets, so breaking one is a bug and panics, not a typed error the caller could act on.
- L2 copies the snapshot from `centroids` unconditionally at the top (`received()` is always 0 here); spherical copies nothing and assigns against the `refresh_readout` readout, per the doc's per-metric rule.
- The row-order reduction is extracted from `assign_blocks` into `Trainer::reduce(blocks, labels, scores)`, which also extends `pass_labels` and advances `batch.assigned`. Both paths call it, so `accumulate` and `accumulate_indexed` share one reduction and cannot drift. It runs only after a clean join, so on error the batch statistics, `received()` and the captured labels are unchanged.
- The batch buffer is a new `Trainer::gathered: Vec<f32>` field, reused across batches like `block_labels` / `block_scores` (taken with `mem::take`, put back after).
- `EMeans::train_rows` calls `trainer.accumulate_indexed(x, ids)`; `gather_into` is gone. Its only other caller, `gather`, now holds the loop itself.
- `Model::assign` (`src/model.rs`): the serial up-front `position` scan is replaced by the same per-task scan + `fetch_min` + skip inside the existing region. On a hit the labels already computed are dropped and the error is `row R column C is V`, as before.
- New tests, named so `cargo test non_finite` runs them: `config::tests::train_non_finite_rows_in_two_tasks_report_the_first_batch_row` (d=8, B=1024, n=3000, NaN at gathered positions 5 and 900 of the first batch, mapped through `pass_order(seed, 0, n)`; also inf at 900 alone names row 900; 1- and 8-thread pools), `model::tests::assign_non_finite_rows_in_two_tasks_report_the_first_row` (n=300, NaN at rows 5 and 290; -inf at 290 alone names row 290; both pools), `trainer::tests::accumulate_indexed_non_finite_row_changes_nothing_and_matches_accumulate` (both metrics: after the error `received()` is 0 and counts are zero, then the same batches fed through `accumulate_indexed` equal `accumulate` of the gathered rows bitwise: centroids, objectives, labels).

### Deviations
- The `train` NaN test uses `train_from` with a clean init instead of `train`: `train` draws its init from the rows, and a NaN row in the init would fail at `begin` with an init message, not the batch message under test. Same pass-0 permutation, same fused path.
- The `fetch_min` mutation was also run on `Model::assign` (not required): `assign_non_finite_rows_in_two_tasks_report_the_first_row` fails at threads=1 with `row 290 column 0 is NaN`.

### Tradeoffs
- `Relaxed` ordering on the `AtomicUsize`: the rayon join is the synchronization point before `into_inner`, and `fetch_min` is commutative, so no stronger ordering buys anything.
- `assert!` vs a typed error for a non-whole batch: the entry point is crate-private and `train` is its only caller; a typed error would be an unreachable branch to test.
- A failing task still gathers its whole block before scanning (the error reads the value back from the buffer): one block's copy on the error path only, and gather then scan keeps one pass over `x` on the hot path.

### Open questions
- None.

#### Recorded observations
- Mutation, run once and restored: `first_bad.fetch_min(..)` changed to `first_bad.store(..)` in `accumulate_indexed` fails `train_non_finite_rows_in_two_tasks_report_the_first_batch_row` at threads=1 (`left: NonFiniteInput("chunk row 900 column 6 is NaN")`, `right: NonFiniteInput("chunk row 5 column 3 is NaN")`).
- `otto ci` exit 0, `All CI checks passed!`; 141 lib tests, 11 parity (determinism and golden digests ok), 4 quality, 1 doctest. `rg -n 'gather_into' src/config.rs` prints nothing (exit 1). `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`.
- `otto bench` before / after: `train/n100000_d32_k256_t5` 41.879 -> 29.230 ms; `train/n100000_d8_k256_t5` 97.075 -> 95.025 ms; `Model::assign` benches 7-59% faster (the serial finite scan left the critical path). Full tables in the design doc, Phase 4.
- `otto h2h` after: every d=32 cell passes (train 1.089 / 1.155 / 1.078 at 1 / 8 / 64 threads; assign 1.097 at 1 thread, 1.237 at 64); the 5 d=8 default cells still miss (Direct is Phase 6). `assign-d8-default-1t` printed 52.03 ms in the full run, which is a load mode: interleaved re-runs gave 44.2 ms for both `684cfe0` and this phase.
