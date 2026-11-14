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

## Phase 5: Cluster-partitioned reduction

### Design decisions
- `Trainer::reduce(blocks, labels, scores)` (`src/trainer.rs`) is now a dispatch: `labels.len() * d >= PARALLEL_REDUCE_VALUES` -> `rayon::current_num_threads()` owners, else 1. The work is in `Trainer::reduce_partitioned(.., owners)`, which the oracle test drives at fixed owner counts.
- Owners split `self.batch.counts` / `self.batch.sums` with `split_at_mut` into disjoint contiguous slices and add in place (`add_owned_rows`, a free fn): every owner walks all blocks in their order (the carry block first when there is one) and every label in row order, skipping rows outside its range with one `wrapping_sub` + bounds compare. No partial is ever zero-started and merged.
- One owner (serial path) calls the same `add_owned_rows` over the whole range, so the serial and parallel paths share one add loop.
- The objective (with the spherical `2 * rows` per block ahead of its scores and the f32 `-2.0 * score`), `pass_labels` and `batch.assigned` run on the calling thread, in row order, after the owners' region joins. `apply` only runs later in `update`.
- `reduce` takes `&[&[f32]]` instead of an iterator: each owner walks the blocks, so they must be re-walkable and shared. `accumulate_indexed` collects its `gathered.chunks(..)` (at most 32 slices per batch); `assign_blocks` already had the slice.
- `PARALLEL_REDUCE_VALUES = 1 << 16`, a `pub(crate)` const next to `BLOCK_ROWS`, with the measured reason in its doc comment.
- `oracle_reduce`: `3564435`'s `Trainer::reduce` verbatim, as an `impl Trainer` block in `src/trainer/tests.rs`. Tests `partitioned_reduce_matches_the_oracle_bitwise` (owners 1, 2, 7, 64) and `reduce_on_both_sides_of_the_parallel_threshold_matches_the_oracle` (dispatch in 1- and 8-thread pools, below and above the threshold). Fixture `ReduceCase`: k=100, a carry block from its own buffer, middle blocks, a short final block, nonzero starting counts/sums/objective/`assigned`/captured labels, both metrics, and six clusters (0, 1, 14, 49, 50, 99: at or next to owner boundaries) that receive only the rows 2^60, -2^60, 1 in column 0 from a start of 0.5. The fixture asserts the oracle lands on 1.0 there, so the cancellation is real.

### Deviations
- Owner ranges are `owner * k / owners` boundaries (sizes differ by at most one) instead of "`k / P` each, the last takes the remainder". Same bits (each cluster still has one owner walking in row order); avoids one owner taking 37 of 100 clusters at P=64. Owners are clamped to `1..=k`, so k < P never leaves empty owners.
- The serial path is two passes over the labels (adds, then the objective) instead of today's one fused loop. Bits are unchanged (independent accumulators, each in row order); the d=8 bench, which takes this path, did not regress.

### Tradeoffs
- `PARALLEL_REDUCE_VALUES = 1 << 16` vs the edges of the passing window (8193..=262144 at these two benches): the window's ends are pinned by one bench each, and the break-even between them was not measured, so the value sits inside it with margin on both sides rather than at an edge.
- P = thread count vs fewer owners: P owners read P x rows labels. At 64 threads and d=32 that still won ~11%; no other P was tried, per the doc's partition policy.
- Collecting block slices into a `Vec` per batch vs a `Clone + Sync` iterator bound: one small allocation per batch, simpler signature.

### Open questions
- None.

#### Recorded observations
- Mutations, run once and restored, each failing both oracle tests: (a) an owner adding into a zero-started partial merged at the end: `sums, L2 d=3 rows=805 owners=2`; re-run with every random starting sum zeroed so only the cancellation clusters' 0.5 starts remain: `sums[0] (cluster 0) got 1.5 want 1`. (b) an owner walking the blocks in reverse: `sums, L2 d=3 rows=805 owners=2` and `sums, L2 rows=8485 threads=8`.
- Bench, `3564435` re-measured interleaved with this phase, two rounds (load 13.9-25.7): d=32 27.971, 27.643 -> 24.799, 24.806 ms; d=8 84.168, 88.278 -> 82.543, 87.108 ms. Threshold probe at `1 << 13`: d=8 199.39, 196.89 ms (2.1x worse), so d=8 must stay serial. Full table in the design doc, Phase 5.
- `otto ci` exit 0, `All CI checks passed!`; 143 lib tests, 11 parity (determinism and golden digests ok), 4 quality, 1 doctest. `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`.

## Phase 6: Lane-vectorized Direct

### Design decisions
- `direct_lanes` (`src/assign.rs`) replaces `direct_l2` and `direct_inner_product`: one generic kernel taking the metric's identity (`+inf` / `-inf`), its lane fold (`sum + diff * diff` with `diff = x - c`; `sum + x * c`), its scalar fold for the tail (`squared_l2` / `inner_product`, today's functions) and its strict order (`<` / `>`). Monomorphized per metric in `nearest_capped`'s match.
- Tiles cover only the `k - k % LANES` centroids in full groups, width `LANES * max(1, tile_floats / (LANES * d))` clamped to that count, so every tile is whole groups. The `k mod LANES` tail runs after the last tile through the scalar fold with the strict order, ids above every tile's, as the doc's pseudocode orders it.
- Loop order is tile outer, rows inner: each tile is transposed once per call (`transpose_into_lanes`) and every row of the block scans it. Each row's running best lives in its `labels` / `scores` slots; a tile replaces it only on the strict order.
- `lane_groups_best` mirrors `lane_scan`'s merge: per-lane best under the strict order (lane keeps its lowest id), lanes merged in lane order with `v == best && v != worst && id < best_id`. The inner loop is `acc: [f32; LANES]` over `as_chunks::<LANES>()` coordinates, so lane `l` runs exactly `squared_l2(row, centroid)`'s ops from 0.0, coordinate 0 first. No FMA, no `mul_add`, no `target_feature`, no new `unsafe`.
- Scratch grows by `reserve_exact` only when shorter than the layout, so its capacity never exceeds `max(earlier capacity, tile layout)`; never shrunk, so a task's later blocks reuse it without refilling. The Direct path now uses the `for_each_init` scratch the earlier phases plumbed through.
- Tests: `lane_direct_equals_the_oracle_bitwise_on_random_and_quantized_data` (d in {1, 2, 3, 8, 16, 33} x k in {1, 15, 16, 17, 33, 256}, random and quantized-to-0.5 data, both metrics, fresh and reused dirty scratch), `tiled_lane_direct_equals_the_oracle_bitwise_with_five_or_more_tiles` (k in {80, 100, 256}, tiles of 1-3 groups leaving 5+ tiles, cap 0 included), `tiled_lane_direct_tie_straddling_a_tile_boundary_goes_to_the_lower_as_in_the_oracle` (k=100: ties across 16- and 32-wide tile boundaries, tile-vs-tail and tail-vs-tail), `tiled_lane_direct_all_infinite_and_nan_rows_match_the_oracle`, `direct_scratch_stays_under_the_tile_cap_at_d64_k65536_and_matches_the_oracle` (also d = TILE_FLOATS / LANES + 1: capacity exactly `LANES * d`), and two `transpose_into_lanes` tests (placement, partial-group panic).

### Deviations
- Doc pseudocode signature `direct_lanes(rows, scratch, j0)` became `direct_lanes(rows, centroids, d, labels, scores, scratch, tile_floats, worst, fold, scalar, better)` owning the tile loop, with `lane_groups_best(row, layout, ..)` as the per-row scan of one tile. Same effect, correct seam: one function for both metrics, the tile cap a parameter as in Phase 3.
- `nearest`'s doc comment now states the Direct scratch bound `max(TILE_FLOATS, LANES * d)` beside the Gemm one.

### Tradeoffs
- One generic kernel with closures vs two hand-written metric copies: one copy of the tile, lane and merge logic; the closures are `Copy` and monomorphized, and the bench shows the inner loop vectorizes (d=8 train 82.5-87.1 -> 42.9 ms).
- Reusing `lane_scan`'s merge rule by a sibling function instead of calling `lane_scan` on a materialized row of k scores: materializing would need a `rows x k` buffer or a per-row k-wide one, which is the unbounded scratch change 2 removed.
- Grow-only scratch vs `resize` to the exact tile every call: no zero-fill per block; the bound still holds because the Direct layout size is fixed per call.

### Open questions
- `train-d8-default-1t` sits exactly on its 1.6 bound (1.603 and 1.600 in two rounds); it is gated at Phase 9 and Phase 7 may move it either way. Worth watching, not acted on here.

#### Recorded observations
- `bin/h2h`, Phase 5 (`3468734`) vs Phase 6 interleaved, two rounds, load 2.27-2.55: train d=8 1t 4.235/4.227 -> 1.603/1.600; 8t 5.935/4.533 -> 1.969/2.185 (Phase 7 gate); 64t 2.087/2.133 -> 0.988/1.011; assign d=8 1t 4.643/4.679 -> 1.736/1.737; 64t 4.192/4.024 -> 1.662/1.578. Full table in the design doc, Phase 6.
- `cargo bench --bench train -- n100000_d8`: 42.880 ms [42.206, 44.317] (Phase 5: 82.543, 87.108 ms).
- Mutations, run once and restored: cross-tile merge on equal (`|| best == *score`) fails the three `tiled_lane_direct_*` tests; a lane keeping the later id on equal fails `tie_between_two_centroids_in_one_lane_goes_to_the_lower` and three more.
- `otto ci` exit 0; 150 lib tests, 11 parity (determinism and golden digests ok), 4 quality, 1 doctest. `cargo test oracle`: 11 passed. `rg -c '\bunsafe (\{|fn|impl|trait|extern)' src/`: `src/assign.rs:1`.

## Phase 7: Task split, pool-thread loop, permutation overlap

Status: implemented in the working tree on `d8b88b4`, `otto ci` green, NOT committed. The Phase 7 gate (`otto h2h --cell train-d8-default-8t`, bound 1.35) misses: 1.32-1.67 over 20 runs, median ~1.47.

### Design decisions
- `task_rows(blocks, threads)` (`src/trainer.rs`) is the one split rule: `TASK_ROWS` (64) when `blocks < threads && threads <= SPLIT_MAX_THREADS` (32), else `BLOCK_ROWS`. A `const` assert pins `BLOCK_ROWS % TASK_ROWS == 0`, so a task never straddles a block.
- `accumulate_indexed` and `Model::assign` chunk the whole call at `task_rows(n.div_ceil(BLOCK_ROWS), current_num_threads())`. Blocks are cut at multiples of 256 from the start, so 64-row chunks land inside blocks; the reduction still walks `gathered.chunks(BLOCK_ROWS * d)`. The `fetch_min` offset is `index * task * d + at`.
- `assign_blocks` cuts each block (carry block included) into `task`-row pieces with their own label/score slices; `reduce` still gets the 256-row blocks.
- `run_passes` (`src/config.rs`) owns the pass loop: pass 0's order drawn up front, then per pass `rayon::join(batches of pass p, draw pass p+1 if any)`, `fed?` after the join, `end_epoch`. `train_rows` calls it inside `rayon::scope` and hands its returned last order to `scatter_labels`.
- Tests: `split_rule_cuts_blocks_into_tasks_only_below_the_thread_count_and_cap` (the seven doc cases plus 1 block at 1 and 2 threads), `assign_blocks_split_into_tasks_matches_one_task_per_block` (carry block + 3 blocks, 1/8/64-thread pools, bitwise counts, sums, objective, labels). `train_matches_caller_loop_in_pools` pins the loop and the overlap; the Phase 4 NaN tests now cross 64-row tasks at 8 threads.

### Deviations
- None from the design. The gate criterion is not met (see Open questions).

### Tradeoffs
- Chunking the whole batch at `task` rows vs nesting a per-block split: same tasks (64 divides 256), one zip chain, no extra job list.
- `run_passes` as a free function vs a closure inside `scope`: names the loop, keeps `train_rows` short, and is the seam the scratch instrumentation hooked.

### Open questions
- The 8t gate. Measured root cause (full numbers in the design doc, Phase 7): the S5 prototype itself measures 1.35-1.54 (median ~1.42) in `bin/h2h`, against 1.12-1.20 in its own harness the same hour. The S5 harness ran emeans from the main thread on the global pool, so emeans paid a per-region handoff (18.3-20.3 ms) that `bin/h2h`'s `pool.install` removes (12-16.5 ms). This build is within ~1 ms of the prototype's absolute time. What remains is the lane Direct kernel's per-row cost (parked non-goal), serial reduce + update (~2.2 ms) and region sync (~5 ms). Scott's call: rebound the 8t cell on `bin/h2h` evidence, or reopen the kernel non-goal.
- `train-d8-default-1t` measured 1.617 and 1.613-1.618 (bound 1.6, Phase 9). Same code path as Phase 6 at 1 thread (no split, `join` inline); Phase 6 had 1.600-1.603.

## Phase 7 (orchestrator addendum, 2026-11-14): gate resolution

### Design decisions
- Scott chose option A: reset the d=8 default bounds from `bin/h2h` measurements (`bin/h2h/src/main.rs` `GRID`): train-d8-default 1t 1.6 -> 1.75, 8t 1.35 -> 1.8; assign-d8-default 64t 1.6 -> 1.8. Recorded in the design doc's Acceptance Criteria and Resolved Decisions.

### Deviations
- Supersedes the "NOT COMMITTED" status above: Phase 7 is committed with the reset bounds. The criterion was a doc defect: the S5 prototype it came from misses 1.35 in `bin/h2h` (median ~1.42) because the S5 harness charged emeans a calling-thread-to-pool handoff `bin/h2h` does not.

### Tradeoffs
- Bounds at ~8% over the worst observed value vs. the median: the gate is a median of 15 pairs that still swung 1.32-1.67 over 20 runs, so a median-based bound would flake.

### Open questions
- None.

## Phase 3 (orchestrator addendum, 2026-11-14)

### Design decisions
- `TILE_FLOATS = 1 << 18` kept by the doc's "faster at 64 threads" rule; the doc's Data Model and Resolved Decisions lines were updated to say so. 2^16's 1-thread (~11% faster) and memory (~35% lighter) edge was surfaced to Scott; no change requested.

### Deviations
- None.

### Tradeoffs
- None.

### Open questions
- None.
