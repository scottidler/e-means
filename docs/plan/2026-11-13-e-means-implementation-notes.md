## Phase 1: Repo and scaffold
### Design decisions
- Full `Error` enum (all 11 variants from the doc's API section) defined now, with payloads: src/error.rs. The doc scopes Phase 1 to `error.rs` and lists the variants once; later phases construct them without reopening the enum.
- `.otto.yml` `test` task runs plain `cargo test`: e-means has no features, so expand-tilde's default/all-features split is dropped.
- `mutants` task keeps expand-tilde's body verbatim, with the TMPDIR renamed to `/var/tmp/e-means-mutants`.
- `Cargo.lock` is committed: the repo's Rust `.gitignore` does not ignore it, and it keeps CI and mutants builds reproducible.
### Deviations
- `ci.yml` keeps expand-tilde's `rust-version: "1.99.1"` input; it matches the installed toolchain (cargo resolved against 1.99.1).
### Tradeoffs
- Payload variants vs. unit variants: payloads give actionable messages; Phase 3 may adjust fields if the trainer needs different data.
### Open questions
- None.

## Phase 2: Recipe
### Design decisions
- `pub mod recipe` with every item `pub`: src/lib.rs. `#![deny(dead_code)]` fails the build on constants and laws that Phases 3-6 first consume; a private module cannot hold them yet.
- `RecipeSpec` input struct and `Recipe::resolve(&RecipeSpec) -> Result<Recipe>`: src/recipe.rs. Eight inputs exceed clippy's argument limit; the spec struct is what `EMeans` (Phase 3) will fill.
- `Recipe` fields: `batch_size`, `alpha0`, `batch_law: Option<BatchLaw>` (resolved, never `Auto`, `None` for a caller `batch_size`), `unused_threshold: f32`, `epoch_rows`, `total_steps`. The minimum the trainer reads; the name and reason strings of the Python recipe dict are dropped.
- `resolve` validates epochs (finite, > 0, rows sum > 0), `alpha0` in (0, 1], `batch_size >= 1`, threshold >= 0 and not NaN: src/recipe.rs `Recipe::resolve`, from the design's edge-case table. `d`, `k`, `n` checks stay in Phase 3's `begin`.
- `batch_from_law` and `step_law` are public so tests assert B and alpha0 separately.
- `INIT_USAGE`, `SPLIT_EPSILON`, `UNUSED_THRESHOLD` are `f32` (state arrays are f32); `ALPHA_MIN_FACTOR` is `f64` (the schedule runs in f64).
### Deviations
- Faiss `_resolve_recipe` is split into `batch_from_law` + `step_law` + `Recipe::resolve`. Same effect, correct seam: Rust has no dict recipe, and the pieces are separately testable.
- Ported tests that go through the `Emeans` class (`:138-142`, `:167-186`, `:188-192`, `:194-206`) are ported at the `Recipe::resolve` seam; the training calls arrive in Phase 5. `:92-98` (verbose output) is covered on the row-count side only.
### Tradeoffs
- Rust `powf`/`log2` vs Python `**`/`math.log2` can differ in the last ULP. Criteria use the specified 1e-12 tolerance, and no fixture sits near a `round_pow2` tie.
### Open questions
- None.

## Phase 3: Trainer core
### Design decisions
- `BLOCK_ROWS = 256`: src/trainer.rs. Small enough that the chunk-invariance fixture (B = 2 * 256 + 37) spans 3 blocks cheaply; Phase 8/9 can retune it, since results never depend on it.
- Batch counts, sums, objective and assigned-row count live in one `BatchAccumulator`; `carry` holds rows of a partial block, flushed when the block (cut from the batch start) fills: src/trainer.rs `Trainer::accumulate`, `assign_block`. A full block arriving inside one chunk is assigned straight from the chunk, no copy.
- Reduction in row order on one thread: `counts[l] += 1`, `sums[l] += f64(row)`, `objective += f64(distance)`, block by block: src/trainer.rs `assign_block`.
- EMA order exactly as the Data Model: pass counts, last-pass sums, decay all masses by `f32(1 - alpha)` unless `alpha == 1` (tested on the f64 alpha), then per touched cluster `mass = retained + f32(alpha) * n`, `weight = f32(alpha) * n / mass`, mean in f64 rounded to f32, `weight == 1` takes the mean: src/trainer.rs `Trainer::apply`.
- `pass_sums` is a `Vec<f64>` allocated at `begin` and added to only in the last pass, not `Option<Vec<f64>>`: src/trainer.rs. The memory budget already counts it; it removes an invariant `expect` in `finalize`.
- Last-pass labels are captured in feed order and handed to a `Model::labels()` accessor now: src/model.rs. `#![deny(dead_code)]` fails a captured-but-unread field, and the Phase 3 criterion asserts labels `[0, 0, 1, 1]`.
- `update_with` validates lengths, count total against the due batch, finite objective and sums, zero sums for zero counts, means that fit f32, and no accumulated rows, all before any change: src/trainer.rs `check_statistics`. A last-pass `update_with` drops the labels (`Model::labels()` is `None`): those rows have no label.
- `apply` checks every moved centroid for finiteness and poisons with `NonFiniteState` on overflow (finite rows near `f32::MAX` with an opposite-signed centroid overflow `mean - c`). Fail closed instead of carrying inf/NaN.
- Revival is `Trainer::revive`, called at `(step + 1) % CHECK_UNUSED_EVERY == 0`, returning 0 splits until Phase 4.
- `Kernel` enum with only `Direct`, fixed at `Trainer::new`; `assign::nearest` asserts buffer lengths: src/assign.rs.
- Logging: `debug!` on `begin`, `end_epoch` (with the epoch stats), `finalize`; `trace!` on `accumulate`, `update`, `update_with` (per-batch hot path, per refs/logging.md).
- Bite check (Testing Strategy): moved the `usage *= gamma` decay loop in `Trainer::apply` to after the touched-cluster EMA. `ema_matches_f64_scalar_oracle_over_400_steps` failed (`e_means_test_utils.py:979` assertion) and `initial_mass_first_update_and_zero_mass_reentry` failed (`:945`); restored, 49/49 pass.
### Deviations
- `Error` payloads adjusted: `EpochIncomplete(String)`, `ScheduleComplete(String)`, `NonFiniteState(String)` (were `{ missing }`, unit, unit). Several cases carry no meaningful count ("pass done, end_epoch not called"), and the reason string names the pass. Phase 1's error test was rewritten for the new messages.
- `update()` / `accumulate()` on a pass whose rows are all applied (before `end_epoch`) return `ScheduleComplete("pass i of n has all r rows applied; call end_epoch")`. The contract does not name this case; no step remains in that pass.
- `end_epoch()` after the last pass ended returns `ScheduleComplete`, not `EpochIncomplete`. Mid-schedule "called twice in a row" is `EpochIncomplete` as specified (the next pass has 0 rows applied).
- Builder knobs `seed`, `spherical`, `max_points_per_centroid`, `capture_labels`, and `train`/`train_from` are not in `EMeans` yet: nothing consumes them before Phases 4-6, and `deny(dead_code)` rejects unread fields. Phase 3 has `new`, `epochs`, `batch_law`, `batch_size`, `alpha0`, `unused_threshold`, `begin`.
- Faiss tests that drive `train(x, init)` (`test_e_means.py:300-316`, `:379-389`, `:444-453`) are ported through `begin` + a row-order feed loop. Same effect: each case's result does not depend on row order (one batch per pass, or k = 1). The `assign` half of `:314-316` is checked with `assign::nearest` against the final centroids.
- The Faiss multi-chunk test (`e_means_test_utils.py:486-613`) compares chunked and whole runs with `assert_allclose(1e-6)`; the port asserts bitwise equality, which the reduction-order design guarantees.
### Tradeoffs
- Snapshot copy per batch (O(k*d)) vs assigning against `centroids` directly in Euclidean mode — kept the snapshot: it is the designed state, and Phase 6 turns it into the spherical readout.
- Zeroing batch sums per batch (O(k*d)) vs tracking touched clusters — zeroing, as the reference does; assignment is O(B*k*d).
### Open questions
- None.

## Phase 4: Revival
### Design decisions
- `Stream` wraps `ChaCha8Rng` and owns the rest of the contract: key = four SplitMix64 outputs of the seed, little-endian (`expand_seed`), `set_stream(id)` per use, `uniform` = `(u64 >> 11) * 2^-53`, `below` = Lemire multiply-shift with rejection, `permutation` = Durstenfeld Fisher-Yates from the top: src/rng.rs. `rand_core` 0.10 / `rand_chacha` 0.10 (what `cargo add` resolved; 0.10 renames `RngCore` to `Rng`).
- Stream ids as `StreamId { Pass(i) -> i, Init -> u64::MAX, Revival -> u64::MAX - 1 }`: src/rng.rs `StreamId::id`. Pass ids can never reach the reserved ones (a pass count is a `Vec` length).
- Golden vectors for seed 1234 in src/rng/tests.rs: SplitMix64 words `[13478418381427711195, 10936887474700444964, 3728693401281897946, 5648149391703318579]`; revival stream first 8 u64 `[7839892025930919283, 4898396843259606927, 9463475350526664508, 9108384216797613808, 15029637854104988123, 15329761231136882789, 4486195400716396707, 7132333170260767736]`; first 4 uniforms `[0.4250013983283042, 0.26554262495791003, 0.5130160267152014, 0.49376649778423254]` (compared by bits); pass-0 permutation of 10 `[5, 3, 7, 1, 2, 8, 0, 6, 9, 4]`; first u64 of pass 0 `7751969689126549159`, of init `13412373498528894678`. SplitMix64 and the permutation were cross-checked against independent Python transcriptions (the permutation over the same pass-0 u64 stream).
- Fisher-Yates and `below` ship now because the Phase 4 bullet asks for a permutation golden vector; nothing outside tests calls them until `train` in Phase 5, so `deny(dead_code)` would reject them. They carry `#[cfg_attr(not(test), expect(dead_code, reason = ...))]`, as do `StreamId::Pass` and `StreamId::Init`. `expect`, not `allow`: once Phase 5 calls them the expectation goes unfulfilled and the build fails until the attribute is removed.
- Revival as free functions over slices (`threshold`, `revive`, `separate`, `choose_donors`): src/revival.rs. The trainer passes `usage`, `centroids`, `d`, tau and its `split_rng`; tests drive the functions directly with controlled masses, which is what the Faiss test does through private fields.
- Threshold = sequential f32 fold, `* tau`, `/ k as f32` (D8). Dead = `usage < threshold` (strict); donor = `usage >= threshold && usage > 0` (`is_donor`). `nsplit == 0` returns before any draw.
- Donor key = `f64::from(usage.ln())` (ln in f32, widened) `- (-u.ln()).ln()` in f64; one uniform drawn per cluster, donor or not, floored at `f64::MIN_POSITIVE`; sort descending by `total_cmp`, ties to the higher index (what a stable ascending argsort reversed gives). Targets are the first `nsplit` dead indices.
- Split: donor `c * (1 - offset)`, child `c * (1 + offset)` with `offset = +/-SPLIT_EPSILON` alternating from coordinate 0, all f32; ULP separation via `next_up`, falling back to `next_down` at `f32::MAX`; donor and child each get `usage[source] / 2`. All donor and child rows are built in scratch and checked finite before anything is written: src/revival.rs `revive`.
- `Trainer::finish_step` sets `poisoned` when revival returns an error (only `NonFiniteState`), so the next call returns `Poisoned`: src/trainer.rs.
- `EMeans::seed(u64)` added, default `DEFAULT_SEED = 1234` (`REF/e_means.py:169`), passed to `Trainer::new`, which seeds `split_rng = Stream::new(seed, StreamId::Revival)`: src/config.rs, src/trainer.rs.
- Bite check (Testing Strategy): changed `usage[j] < threshold` to `<=` in `revive`. Failed: `revival::tests::mass_at_the_threshold_is_alive` (Ok(2) instead of Ok(1)), `revival::tests::split_revives_the_dead_cluster_and_drops_its_mass` (`e_means_test_utils.py:1050`), `revival::tests::no_pair_draws_no_random_number`, and `trainer::tests::ema_matches_f64_scalar_oracle_over_400_steps` (tau 0 at d = 1 gives threshold 0; an underflowed mass of 0 became dead and split). Restored; 75/75 pass.
### Deviations
- Phase 3's `revival_hook_runs_without_splitting` pinned the no-op hook on the `test_e_means.py:417-429` fixture; it is replaced by `revival_starts_on_fifth_update` on the same fixture, asserting 0 splits after 4 updates and 1 after 5.
- `test_e_means.py:431-442` (`test_revival_follows_current_batch_update`) runs with the d = 1 default tau of 0, so revival cannot fire in the reference test whatever the order. The port keeps that case and adds the same fixture with tau = 0.1, where revival before the fifth update would split cluster 1. The row order is fed directly (the 10.0 row last) instead of drawn from `_epoch_seed(7, 0)`, which is the order the reference constructs.
- `e_means_test_utils.py:1037-1084` infers the donor rather than pinning NumPy's RNG; the port does the same with the crate's stream (seed 123), so which donor is picked differs from Faiss by D1.
- The Faiss threshold test's Torch half and the object-deletion half of `:1112-1148` (`weakref` release, second `finalize` returns the same array) have no Rust counterpart; ownership covers both.
- `Stream::below(0)` panics (assert) instead of returning a `Result`: an internal precondition, never reachable from caller input (Fisher-Yates passes `i + 1`).
### Tradeoffs
- Scratch `Vec`s per revival event vs preallocated buffers on the trainer: events with `nsplit > 0` are rare and the reference allocates too; the scratch keeps the no-commit-on-overflow rule local to `revive`.
- Gumbel top-k by a full sort of donor keys (O(k log k)) vs a partial select: revival runs every 5th update and k is at most thousands; the sort makes the tie rule explicit.
- `cfg_attr(not(test), expect(dead_code))` on four Phase 5 items vs making `rng` a public module (the Phase 2 precedent for `recipe`): the RNG is not caller API, and `expect` removes itself loudly once Phase 5 reads the items.
### Open questions
- None.

## Phase 5: train, Model, labels
### Design decisions
- `train` / `train_from` are `train_subset` -> `train_rows`, and `train_rows` is the caller-driven loop: `begin`, then per pass a permutation from `Stream(seed, Pass(i))`, the first `rows_i` entries cut into batches of B, each gathered into one reused buffer and fed as one `accumulate` + `update`, then `end_epoch`, `finalize`: src/config.rs `EMeans::train_rows`. No trainer internals are touched; every error is the trainer's or `begin`'s.
- Init: caller init as given; else `n == k` uses `x` itself (no draw); else the first k of `Stream(seed, Init).permutation(n)`: src/config.rs `train_rows` (`REF/e_means.py:593-600`).
- Row-order labels by one helper, `Model::scatter_labels(rows, n)` (label at feed position p goes to row `rows[p]`, others `None`), applied once with the last pass's order and once more with the kept-row ids under `max_points_per_centroid`: src/model.rs. The caller-driven path never calls it, so its labels stay in feed order, all `Some`.
- `max_points_per_centroid`: `limit == 0` is `InvalidParameter` at `train` (the builder setter cannot fail); `k * limit` saturates; when `n > k * limit`, the kept rows are the first `k * limit` of `Stream(seed, Subset).permutation(n)`, sorted, gathered into a copy, and trained exactly as a `train` on that copy would be (the ported test asserts bitwise equality with a direct `train` on the kept rows): src/config.rs `training_subset`. It applies to `train_from` too, as Faiss `train(x, init)` does.
- `capture_labels(bool)` is a builder field handed to `Trainer::new`; `start_pass` allocates the last-pass label buffer only when it is set, so it also governs the caller-driven path: src/trainer.rs.
- `Model` now carries `d` and the run's `Kernel`; `Model::assign` uses that kernel, returns `Assignment { labels: Vec<u32>, distances: Vec<f32> }` (squared L2), rejects partial rows with `DimensionMismatch` and NaN/inf with `NonFiniteInput` (a NaN row would otherwise land on centroid 0 with distance inf): src/model.rs.
- The four `cfg_attr(not(test), expect(dead_code))` attributes in src/rng.rs are gone; `train` reads `Pass`, `Init`, `below` and `permutation`.
- Logging: `debug!` on `train`, `train_from`, `Model::assign` entry/exit, and the subset draw.
- Bite check: shifted the pass stream to `StreamId::Pass(pass + 1)` in `train_rows`. Failed: `caller_batches_match_matrix_training`, `train_matches_a_caller_loop_from_the_seeded_init`, `fractional_epochs_preserve_batches_and_closing_labels`. Restored; 89/89 pass.
### Deviations
- New reserved stream id `StreamId::Subset = u64::MAX - 2` for the `max_points_per_centroid` draw; the Dependencies section names only two reserved ids (init, revival). The subset needs its own stream so that training on the subset draws init and pass orders exactly as a `train` on those rows would. Golden first u64 for seed 1234 pinned in src/rng/tests.rs: `17003396118023392089`.
- Faiss draws the subset with `Generator.choice(n, limit, replace=False, shuffle=False)`; the port takes the first `limit` of a full Fisher-Yates permutation. Same contract (seeded uniform subset, kept in row order), different numbers by D1; O(n) indices instead of O(limit).
- Ported tests use in-test LCG data where Faiss uses `RandomState(...).randn/rand` (D1); the bitwise comparisons are against this crate's own caller-driven loop, not against Faiss values. Faiss's caller-vs-matrix tolerances (`atol 1e-6`) are asserted bitwise here.
- Not ported: `test_e_means.py:488-498` (global RNG untouched; the crate has no global RNG), the `mock.patch` label-mapping failure and Torch/negative-seed halves of `e_means_test_utils.py:615-664`, and the int64-dtype check of `:666-703` (labels are `Option<u32>`, D5).
### Tradeoffs
- Per-pass permutation of all n rows (O(n) `usize`) even for a fractional pass that uses `rows_i` of them, vs a partial shuffle: the design says a fractional pass uses the first `rows_i` entries of the pass's permutation, which a full permutation states exactly; the reference permutes all n too.
- The subset is copied into a new `Vec<f32>` (`k * limit * d`) vs indexing through `keep` on every batch: the copy makes the subset run literally `train` on the kept rows, with no second code path.
- One `accumulate` per batch in `train` vs chunking: chunking is result-invisible (Phase 3 gate), so the simplest feed is used.
### Open questions
- None.

## Phase 6: Spherical
### Design decisions
- `normalize_into(source, out, d)` and in-place `normalize_rows(rows, d)`: per row, divide by the max-abs, then by the f32 norm (sequential sum of squares); a zero divisor becomes 1, so a zero row stays zero: src/spherical.rs (`REF/e_means_spherical.py:32-42`). Module is `pub(crate)`; the design does not list it as API.
- `Metric::{L2, InnerProduct}` in src/assign.rs; `nearest` takes the metric and writes a per-row score: squared distance under L2, inner product under InnerProduct. Largest inner product wins with a strict `>` from `-inf`, so ties (the zero row) go to the lowest index.
- Trainer: `metric` field fixed at `begin`. `Trainer::new` normalizes the init in place (`REF/e_means.py:322-328`), then `refresh_readout` projects the EMA state into `snapshot`; `finish_step` calls `refresh_readout` after the EMA and revival, before the step advances (`REF/e_means.py:217-241`, invalidation at `:299` and `:505`). `accumulate` skips the first-chunk centroid copy in spherical mode, because the snapshot already is the readout: src/trainer.rs.
- Batch objective in spherical mode: `2 * block_rows` (f64) added per block, then per row `f64(-2 * ip)` with the product in f32 (`REF/e_means.py:303-307`, `:432-434`): src/trainer.rs `assign_block`.
- Closing: means in f64 rounded to f32 (unchanged), then `normalize_rows` over all k rows, so clusters with no last-pass rows keep their EMA position, normalized: src/trainer.rs `finalize`.
- `Model` carries the metric; `Model::assign` maps the inner product to `2 - 2 * ip` in f32 (D10): src/model.rs.
- `EMeans::spherical(bool)`, default false, passed to `Trainer::new` as the metric: src/config.rs.
- Bite check: removed the `self.refresh_readout()` call in `finish_step`. Failed: `trainer::tests::spherical_deferred_ema_keeps_raw_state_and_refreshes_readout` ("tests/e_means_test_utils.py:212 readout refreshed after the update: value 0 is 0.6, want 0.000027536757") and `trainer::tests::spherical_readout_follows_the_update` (`e_means_test_utils.py:319`). Restored; 106/106 pass.
### Deviations
- The snapshot in spherical mode is refreshed right after each update instead of frozen at the next batch's first chunk. Same effect, correct seam: centroids change only in `update` / `update_with`, so both give the same snapshot for every row of a batch; the refresh is the "readout refreshed after each update" the design names.
- `nearest` returns the inner product, not Faiss's `-2 * ip` (`REF/e_means.py:303-307`); the trainer forms `-2 * ip` for the objective and `Model::assign` forms `2 - 2 * ip`.
- The `2 * rows` objective term is added per 256-row block; Faiss adds it per `assignment_chunk_size` chunk and sums each chunk's distances pairwise. Objectives agree to f64 rounding, not bitwise.
- `e_means_test_utils.py:260-271` (wide closing) sets f64 pass sums of 1e300, so the reference projects before narrowing. The design rounds the mean to f32 first, and in this crate that state is unreachable: pass sums come from finite f32 rows (mean within f32 range) or from `update_with`, which rejects a mean that overflows f32. The port uses sums `[6e38, -6e38]` over 2 rows (mean 3e38, representable), which still needs the max-abs prescale, and also pokes the empty cluster's raw position to `[-3, 0]` to show it is normalized.
- Not ported: the `assertIs` cache-identity and Torch version-counter halves of `:168-226` and `:292-325` (no borrowed cache handle exists here), the Torch backends of every test, and `:163` (input untouched; `&[f32]` guarantees it). `:327-388` uses in-test LCG rows and the crate's pass orders (D1), and asserts bitwise where Faiss uses atol 2e-7.
### Tradeoffs
- Snapshot doubles as the readout (no separate `readout` buffer) vs a dedicated cache: one k*d buffer instead of two; the design's state table already calls the snapshot "unit-norm readout in spherical mode".
- `metric` as a separate enum next to `Kernel` vs folding it into `Kernel`: Phase 9's Gemm kernel needs both metrics, so the two axes stay independent.
- Sequential f32 sum of squares in `normalize_rows` vs a pairwise sum like NumPy's einsum: rows are already prescaled to max-abs 1, so the norm is in `[1, sqrt(d)]` and the order only moves the last bits.
### Open questions
- None.

## Phase 7: Faiss parity and quality
### Design decisions
- `compat/generate.py` drives Faiss `NumpyEmeans` (seed 1234) through its caller-driven API (`begin`, `accumulate`, `update`, `end_epoch`, `finalize`) with an explicit init and explicit pass orders. The inputs are then exactly what the Rust side replays. `compat/requirements.txt` pins `faiss-cpu==1.16.0`, `numpy==2.5.4`, `scipy==1.18.1`; the venv lives at `compat/.venv` (gitignored). Otto tasks: `compat-setup` (`uv venv --clear`, `uv pip install -r`) and `compat` (refuses without the venv). Neither is in `ci`.
- Two kinds of draw, named per fixture in the manifest:
  - `faiss` (the n=4000 fixtures): the init and orders `NumpyEmeans.train(x)` draws itself (`RandomState(seed).permutation(n)[:k]`, `RandomState(_epoch_seed(seed, p)).permutation(n)[:rows_p]`, `REF/e_means.py:594-614`), stored as files. The rows are NumPy `randn` blobs in f32, off-grid, so per-cluster f32 sums round and D2 is exercised. Driving the caller API with these draws gives the same objectives as `train(x)`, to every printed digit (separated_d8 227625.092 both ways).
  - `splitmix` (the n=100000 fixtures): rows, init, and orders come from an integer-only recipe. SplitMix64 seeded `stream_seed + stream`; Fisher-Yates from the top with `next() % (i + 1)`; rows `grid_blobs`: Irwin-Hall noise (12 ten-bit uniforms, centered) on a 2^-10 grid, centers `3 * draw`, row i in blob `i % 256`. `tests/parity.rs` regenerates all three bit for bit, so no input file is committed for them.
- Every fixture carries FNV-1a 64 hashes of its rows, init, and concatenated orders. `tests/parity.rs::inputs` recomputes the hashes and fails loudly on a mismatch, so a drift in either generator cannot pass silently.
- Format: flat little-endian `f32` / `u32` files plus `tests/fixtures/manifest.json` (recipe, per-pass objectives and split counts, file entries, hashes). The writer refuses an array whose values change on conversion.
- Fixtures. Separated (16 blobs at scale 10, n=4000, k=16, revival off): `separated_d8` (quality law, B=32, 625 steps), `separated_d32` (throughput, B=256, 80 steps), `separated_d8_t2_5`, `separated_spherical_d32` (rows normalized in f32 by the generator; Faiss does not normalize), `separated_d2`. Loose: `oversplit` (d=32, k=64, revival off), `large` (n=100000, d=32, 256 blobs at scale 3, k=256, revival off), `large_revival` (same, default tau 0.1).
- Shared inputs: `large` and `large_revival` share one generator spec and one stream seed, so their rows, init, and orders are the same. `separated_d8` and `separated_d8_t2_5` share `blobs_d8.x.bin`. `separated_d32` and `oversplit` share `blobs_d32.x.bin`.
- The generator checks the separated class itself: it reruns each separated fixture in Faiss with `distance_compute_blas_threshold = 0` and stops if any label moves. Measured (centroid / objective / labels): d8 0 / 7.9e-8 / 100%; d32 0 / 5.3e-8 / 100%; d8 T=2.5 0 / 2.5e-8 / 100%; spherical 0 / 7.4e-9 / 100%; d2 0 / 5.6e-8 / 100%. Stored per fixture as `faiss_blas_forced`.
- `tests/parity.rs`: `inputs` builds the rows, init, and orders. `rust_run` replays them through `EMeans::begin`, one `accumulate` + `update` per logical batch, `end_epoch`, `finalize`, and maps the feed-order labels back to rows through the last pass's order. Before training it asserts the Rust recipe equals Faiss's: batch size, alpha0 (bitwise), epoch rows, total steps. `metrics` computes the design's three metrics. One `#[test]` per fixture, plus `manifest_names_every_fixture`. No test name contains `determinism`, so Phase 8's `cargo test --test parity determinism` selects only its own test. `inputs`, `rust_run`, `faiss_run`, and `metrics` are the pieces it reuses.
- `tests/quality.rs` builds a mixture from an in-test SplitMix64 + Box-Muller generator: 256 equal-size components (row i from component `i % 256`), d=32, centers `N(0, 3^2)`, unit noise, 100000 train rows plus 20000 held-out rows. `EMeans::train` with `.seed(seed)` runs against an in-test Lloyd: 25 full iterations, init = k distinct random rows drawn from the same seed, f64 means, and an empty cluster keeps its position. Held-out MSE is computed in f64 by direct difference. One `#[test]` per seed, so the harness runs them in parallel; Lloyd runs on a scoped thread next to e-means.
- `[profile.test] opt-level = 3` in Cargo.toml. Rust does no fast-math or FMA contraction, so results are bitwise the same as debug. Debug assertions and overflow checks stay on (inherited from dev). In `otto ci`: parity 0.9 s, quality 5.5 s.
- Dev-dependencies: `serde` (derive) and `serde_json` with `float_roundtrip`. Root cause of the first parity failure: alpha0 differed in the last bit (0.10500128138887335 in both Faiss and Rust, parsed from the manifest as ...337). serde_json's default float parser is not correctly rounded; `float_roundtrip` makes it exact.

Measured parity (Rust `Kernel::Direct` against Faiss 1.16.0; centroid = worst per-centroid relative L2, objective = final-pass relative, labels = last-pass agreement):

| fixture | centroid | objective | labels | splits rust / faiss |
|---|---|---|---|---|
| separated_d8 | 3.071e-7 | 1.012e-9 | 100% | 0 / 0 |
| separated_d32 | 1.013e-7 | 3.160e-9 | 100% | 0 / 0 |
| separated_d8_t2_5 | 2.976e-7 | 9.618e-10 | 100% | 0 / 0 |
| separated_spherical_d32 | 1.587e-7 | 7.417e-9 | 100% | 0 / 0 |
| separated_d2 | 4.663e-7 | 7.673e-9 | 100% | 0 / 0 |
| oversplit | 1.051e-7 | 2.831e-9 | 100% | 0 / 0 |
| large | 0 | 5.107e-8 | 100% | 0 / 0 |
| large_revival | 1.952 (not asserted) | 2.741e-2 | 78.698% (not asserted) | 47 [35, 11, 1, 0, 0] / 46 [34, 10, 2, 0, 0] |

Final-pass objectives: separated_d8 Rust 227625.09253954887, Faiss 227625.09230928123. large Rust 6095657.641466828, Faiss 6095657.330136299. large_revival Rust 4925685.168298721, Faiss 5064512.466234207.

Measured quality (held-out MSE, e-means T=5 against Lloyd-25): seed 10 51.6994 against 57.6729; seed 11 52.9388 against 65.5743; seed 12 51.4942 against 57.7037.

Committed fixtures: 2,623,001 bytes over 33 files (32 binaries plus `manifest.json`). The largest is 512,000 bytes (`blobs_d32.x.bin`, `blobs_d32_unit.x.bin`).
### Deviations
- The design lists `x`, `init`, and per-epoch order as fixture contents. For the n=100000 fixtures they are a generator spec plus hashes, not files. The first export stored them (6.4 MB `x` as i16 grid steps, 2 MB orders), and the commit hook refused it: blobs over 1 MB in a public repo need Scott's confirmation or a move to `scottidler/keep`. I did not split files to slip under the limit. Same effect, correct seam: Faiss still trains on exactly the rows, init, and orders the Rust side replays, and the hashes prove it.
- The large fixtures' rows sit on a 2^-10 grid. Per-cluster f32 sums over them are exact, so `large` does not exercise D2 (its centroid metric is exactly 0). The separated fixtures still exercise D2 (off-grid f32 rows).
- `faiss`-draw fixtures stay on Faiss's own MT19937 draws, not the SplitMix recipe. Measured: with a SplitMix init, `separated_d8` stopped being separated in Faiss itself (forced BLAS moved 1.2% of labels, centroid 2.1e-2), and the generator's class check refused it. Separation depends on the init. The MT draws are the ones the design's noise floor was measured with, and they pass the check.
- `separated_d8_t2_5` resolves B=16, not B=32: T=2.5 changes the batch law. The design names B=32 only for `separated_d8`.
- The quality mixture is my choice; the design says only "seeded Gaussian mixture". The first version drew each row's component uniformly at random. On it, seed 12 failed: e-means 57.585 against Lloyd 57.430 (seed 10: 52.41 against 58.67; seed 11: 52.77 against 65.77). Root cause, measured on the seed-12 rows: an unlucky init and order draw from the crate's stream (D1), not a defect. Rust e-means over crate seeds 0..19 ranged 47.2-57.6 with seed 12 the worst of the 20; the next worst was 55.0. On the same rows, Faiss `NumpyEmeans` (seeds 1000-1005) ranged 50.9-54.6 and Faiss `Kmeans(niter=25)` ranged 56.2-61.7. Rust over crate seeds 1000-1005 ranged 48.4-51.7, so its spread matches Faiss's. I then switched to equal-size components (row i from component `i % 256`, the shape of the `large` fixture), and it passes all three seeds. This choice came after the failure: see Open questions.
- Faiss's self-noise objective on `separated_d8` under a forced BLAS kernel is 7.9e-8, above the design table's "at most 2.5e-8" for this class. The design measured on its own fixtures; these are new ones. Labels and centroids hold (100%, 0), and the Rust bound (1e-6) is 12x above it.
### Tradeoffs
- Integer generator for the large inputs vs storing them: about 8.5 MB fewer committed bytes and no hook exception, at the cost of a generator mirrored in Python and Rust. The hashes catch any drift between the two.
- Two draw kinds vs one: SplitMix everywhere would drop the stored small orders (80 KB each), but the n=4000 separated class then fails Faiss's own check (Deviations).
- `[profile.test] opt-level = 3` vs smaller test data or `#[ignore]`: the design fixes n=100000 for both the large fixtures and the quality test, and ignoring them would take them out of `otto ci`. Optimized test builds compile slower; every test runs faster, the mutants task included.
- Lloyd baseline without empty-cluster splitting (weaker than `faiss.Kmeans`, which splits) vs porting Faiss's split: the design says "in-test Lloyd, random-row init", and a split rule would be a second clustering algorithm to verify.
### Open questions
- The quality mixture changed after seed 12 failed on the first one (Deviations). Both choices fit the design's wording. The claim is a single run per seed, and e-means's run-to-run spread on this data (about 47-58) overlaps Lloyd-25's lower tail (about 56), so a single-seed comparison can fail on an unlucky init on any mixture. Keep the three-seed form, or move to a statistic over several crate seeds per mixture?
- The large fixtures replay a generated init and orders, not the ones Faiss `train(x)` draws. If Scott wants Faiss's own `train` draws for them too, the 2 MB orders file needs his confirmation for the public repo (or a home in `scottidler/keep`).

## Phase 7 addendum (orchestrator): seed-12 quality miss, root cause

### Design decisions
- Root cause of the first-mixture seed-12 failure (e-means 57.585 vs Lloyd 57.430): the crate's own seed-12 draw (init + orders) is the worst of 20 seeds. Not a defect.
  - Faiss init and orders fed to the Rust Trainer, revival off: held-out MSE 56.9986 on both sides; objective differs by 2.658e-9 relative.
  - Revival on, Faiss seeds 0..19 paired on the same inputs: paired MSE difference mean -0.250, std dev 2.168, t = -0.52 (no shift). Only revival donor choice differs (D1).
  - Each side with its own RNG, 20 seeds: Rust mean 51.212 / median 51.200 / sd 2.569; Faiss mean 51.164 / median 51.047 / sd 2.473. Faiss max 56.813, Rust max 57.585.
  - Faiss seed-12 init: Rust Lloyd-25 55.7964, Rust e-means 52.8728.

### Deviations
None.

### Tradeoffs
None.

### Open questions
- The quality test now uses equal-size mixture components (chosen after the random-size mixture failed seed 12). Both shapes fit the doc's "seeded Gaussian mixture". On the random-size mixture, one run per seed can fail on an unlucky draw (Rust range 47.2-57.6 against Lloyd-25 about 56-62). Keep the equal-size three-seed gate, or average several seeds per mixture?

## Phase 8: Parallel assignment

### Design decisions
- `accumulate` gathers every block the chunk completes (the filled `carry` block first, then whole blocks cut straight from the chunk, all relative to the batch start) and hands them to `assign_blocks` in one call: `src/trainer.rs:Trainer::accumulate`. A whole-batch chunk spreads its blocks over the pool instead of assigning one block per call.
- `assign_blocks` splits the label/score scratch per block with `split_at_mut`, runs `assign::nearest` per block with `into_par_iter`, then adds counts, sums, objective and pass labels on the calling thread in row order: `src/trainer.rs:Trainer::assign_blocks`. This is the design's "Reduction order": parallel work writes only per-block outputs, every float addition stays sequential.
- The spherical `2 * rows` term is still added once per block, before that block's rows, in block order. Same sequence of f64 additions as Phases 6 and 7, so results are bitwise unchanged from Phase 7.
- `Model::assign` runs `par_chunks(BLOCK_ROWS * d)` zipped with `par_chunks_mut(BLOCK_ROWS)` on labels and distances: `src/model.rs:Model::assign`. Each row's output depends only on the row, so there is no reduction.
- Both paths use the ambient `rayon` pool, so a caller picks the thread count with `pool.install`. The determinism test does exactly that.
- Determinism test `determinism_across_thread_counts_and_chunkings` in `tests/parity.rs`. `rust_run` became a wrapper over `rust_run_chunked(fx, x, init, orders, chunks)`, which feeds each batch in chunks cycling through `chunks` (`[usize::MAX]` = whole batch, `[1]`, `[3, 5, 7]`; the Phase 3 unit test's chunking convention). The 1-thread whole-batch run is the reference; the 8-thread whole, row-at-a-time and 3/5/7 runs must match its centroid bits, labels, objective bits and split counts on all 8 fixtures. Runtime: about 7 s for the whole test.
- Unit test `assign_over_many_blocks_is_thread_count_invariant` in `src/model/tests.rs`: 3 blocks + 17 rows, 1 and 8 threads, bitwise equal to a serial `assign::nearest`.

### Deviations
None.

### Tradeoffs
- One `rayon` task per 256-row block vs larger tasks: the block is the unit the design already fixes for chunk invariance, and at k=256, d=32 a block is about 2M multiply-adds, enough to amortize task overhead. Row-at-a-time feeding hands rayon one block per completed block, same work as before.
- Scratch kept in the trainer (`block_labels`, `block_scores`, grown to the chunk's complete rows) vs per-block allocations: one buffer, no allocation in steady state. The `Vec<&[f32]>` of block slices is allocated per `accumulate` call and holds at most `B / 256 + 1` pointers.

### Open questions
None.

Parity metrics after the change (bitwise identical to Phase 7; the reduction order did not change). Centroid / objective / labels:
- separated_d8: 3.071e-7 / 1.012e-9 / 100%
- separated_d32: 1.013e-7 / 3.160e-9 / 100%
- separated_d8_t2_5: 2.976e-7 / 9.618e-10 / 100%
- separated_spherical_d32: 1.587e-7 / 7.417e-9 / 100%
- separated_d2: 4.663e-7 / 7.673e-9 / 100%
- oversplit: 1.051e-7 / 2.831e-9 / 100%
- large: 0 / 5.107e-8 / 100%
- large_revival: 1.952 / 2.741e-2 / 78.698%, splits rust [35, 11, 1, 0, 0] faiss [34, 10, 2, 0, 0]

## Phase 9: Gemm kernel and benches

### Design decisions
- `Kernel::Gemm` and `BLAS_THRESHOLD = 128_000` in `src/assign.rs`; the predicate is `Kernel::for_batch(batch_size, d)` (`B * d < BLAS_THRESHOLD` -> Direct, else Gemm, product saturating). `Trainer::new` calls it once from `recipe.batch_size` and logs the choice in its DEBUG line; `finalize` already handed `self.kernel` to `Model`, so `Model::assign` uses the run's kernel with no change.
- `gemm_nearest` (`src/assign.rs`): one `rows x k` f32 tile of inner products per block (at most `BLOCK_ROWS x k`, the only per-thread scratch), allocated per block call. L2 score is `row_norm + centroid_norm - 2 * ip`, set to 0 when below 0, in that order (Faiss `exhaustive_L2sqr_blas`); the `if < 0` form keeps a NaN a NaN, as Faiss does, where `f32::max` would turn it into 0. Inner-product mode scans the tile for the largest product. Ties go to the lowest index in both, as in the direct kernel. Centroid norms are recomputed per block: O(k * d) against the block's O(BLOCK_ROWS * k * d).
- `inner_products` (`src/assign.rs`) is the safe wrapper around the crate's one `unsafe` block (`matrixmultiply::sgemm`): asserts `d > 0`, whole rows in both inputs, `tile.len() == m * k` (checked multiply), the tile disjoint from both inputs (`disjoint`, pointer ranges, empty is disjoint), and strides that fit `isize`; returns before the call when the tile is empty. Centroids are passed as B = their transpose with row stride 1, column stride d, so no transpose copy. beta = 0, so `sgemm` never reads the tile. `matrixmultiply` runs without its `threading` feature: one block is single-threaded and deterministic, parallelism stays at the rayon block level.
- `rg -l '\bunsafe\b' src/` prints `src/assign.rs` only: the word appears in its module doc, the SAFETY comment and the block, nowhere else in `src/`.
- Tests (`src/assign/tests.rs`): the margin criterion on random LCG data for d = 1, 3, 8, 32, 100 (k=64, n=700), with a floor that at least 75% of rows clear the margin so the check is not vacuous (d=1 clears 88%), plus Gemm distance within `1e-4 * (||x||^2 + ||c1||^2)` of direct; the same margin check in inner-product mode; the offset fixture (`common_offset_direct_is_right_and_gemm_cancels`, which replaces `direct_has_no_common_offset_cancellation` and keeps its direct assertions); exact small-integer cases and tie order; non-negative distances on identical rows; zero-row inner product; empty block; the tile layout; the wrapper's length panics; `disjoint` on overlap, adjacency and empty. `src/model/tests.rs::model_keeps_the_kernel_picked_from_batch_times_d`: d=32, B=3999 -> Direct and B=4000 -> Gemm, Euclidean and spherical, the model keeps the kernel and assigns every row of a separated dataset correctly.
- Benches: `benches/train.rs` (`EMeans::train`, T=5, n=100000, k=256, d=32 and d=8, `BatchLaw::Auto`, so B=8192 Gemm and B=1024 Direct), `benches/assign.rs` (`Model::assign` over the same 100000 rows, both kernels picked through the public API by training with B on either side of the threshold). Shared data in `benches/common/mod.rs`: the `tests/quality.rs` mixture generalized to any d (256 equal-size components, centers N(0, 3^2), unit noise, seed 10). Otto task `bench` runs both; not in `ci`.

Parity metrics with the predicate live (centroid / objective / labels; `large` and `large_revival` now run `Kernel::Gemm`, B * d = 262144; the separated and oversplit fixtures stay Direct, so they are bitwise unchanged from Phase 8):
- separated_d8: 3.071e-7 / 1.012e-9 / 100%
- separated_d32: 1.013e-7 / 3.160e-9 / 100%
- separated_d8_t2_5: 2.976e-7 / 9.618e-10 / 100%
- separated_spherical_d32: 1.587e-7 / 7.417e-9 / 100%
- separated_d2: 4.663e-7 / 7.673e-9 / 100%
- oversplit: 1.051e-7 / 2.831e-9 / 100%
- large: 0 / 5.588e-8 (Phase 8 Direct: 5.107e-8) / 100%. Centroid 0 is real, not a fold artifact: last-pass labels match Faiss on every row and the grid-blob rows are integers, so the closing means are exact on both sides.
- large_revival: 1.952 / 2.741e-2 / 78.290% (Phase 8: 78.698%), splits rust [35, 11, 1, 0, 0] faiss [34, 10, 2, 0, 0]
- `determinism_across_thread_counts_and_chunkings` passes with the large fixtures on Gemm.

Measured numbers for the README (Phase 10 pastes these). This machine, 64 cores, 2026-11-13, `cargo bench`, criterion medians:

| bench | kernel | median | `NumpyEmeans` | bound |
|---|---|---|---|---|
| train n=100000 d=32 k=256 T=5 (B=8192) | Gemm | 44.4 ms | 0.41 s | < 0.41 s |
| train n=100000 d=8 k=256 T=5 (B=1024) | Direct | 99.9 ms | 4.35 s | < 4.35 s |
| assign 100000 rows d=8 k=256 | Direct | 2.22 ms | | |
| assign 100000 rows d=8 k=256 | Gemm | 1.21 ms | | |
| assign 100000 rows d=32 k=256 | Direct | 7.46 ms | | |
| assign 100000 rows d=32 k=256 | Gemm | 2.01 ms | | |

### Deviations
- The design's success criterion names `Kernel::Direct` labelling the offset fixture 1 and `Kernel::Gemm` labelling it 0; the existing Phase 3 test `direct_has_no_common_offset_cancellation` pinned only the first half. It is renamed `common_offset_direct_is_right_and_gemm_cancels` and asserts both, plus the Gemm distance 0 the design's `[0, 0]` expansion implies.
- The bench data is my choice (the design gives only n, d, k, T): the quality-test mixture in d=32 and d=8. The `NumpyEmeans` reference times were measured on the design author's data, not necessarily this mixture.

### Tradeoffs
- Tile allocated per block call vs a thread-local or per-trainer buffer: one 256 KB allocation (k=256) per block against about 2M multiply-adds; the train bench is 9x under its bound at d=32, so a buffer pool buys nothing measurable and would add state shared across rayon tasks.
- Centroid norms recomputed per block vs cached with the snapshot: caching needs the trainer to own a second derived buffer that must be refreshed at every snapshot change (Euclidean freeze, spherical readout refresh) and in `Model`; the per-block cost is 1/256 of the block's work.
- Sequential `x . x` for norms vs SIMD-summed norms like Faiss's `fvec_norms_L2sqr`: results differ in the last bits from Faiss either way (sgemm's summation order already differs), and one helper serves both kernels.

### Open questions
None.

## Phase 10: README, example, release
### Design decisions
- README numbers are pasted from the Phase 7 and Phase 9 notes, none re-measured: README.md. The parity table drops the `large_revival` row (not asserted) and describes it in prose with the Phase 9 figures (78.290%).
- Quality numbers name the baseline in the same sentence and table header (random-init Lloyd-25, no k-means++): README.md "Quality".
- `examples/blobs.rs`: four 2-d blobs from a fixed LCG, `EMeans::train` then `Model::assign`; asserts each blob lands in one cluster with its centroid within 1.0 of the blob center, so it exits nonzero if it regresses.
- Crate doc in `src/lib.rs` gained one doctest (it had none): a 1-d two-blob `train` and `assign`.
- Consumer instructions use a git tag dependency, `tag = "v0.1.0"` (Cargo.toml version 0.1.0).
### Deviations
- The README deviation table keeps the design's wording, shortened where the design cited reference line numbers (D2, D4, D8); the `rg -c` row count is 10.
### Tradeoffs
- The caller-driven README snippet is schematic (`order_for_this_pass`, `rows` are the caller's) vs a full runnable program: the first form is the runnable example, and a second example would duplicate Phase 8's parity driver.
### Open questions
- Pending the finalization checkpoint: `bump release` (version, tag, push, install) and the scratch-crate-from-tag success criterion were NOT run in this phase; they run after the user approves.
