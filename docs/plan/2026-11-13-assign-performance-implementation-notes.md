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
