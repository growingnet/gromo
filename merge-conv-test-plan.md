## Goal

Raise line and branch coverage of newly introduced code in `Conv2dMergeGrowingModule` (in `src/gromo/modules/conv2d_growing_module.py`) to 100% by enriching the existing `TestConv2dMergeGrowingModule` test class (in `tests/test_conv2d_growing_module.py`). Keep existing tests intact and extend them with targeted, minimal new cases.


## Scope and requirements

- Cover every property/method/branch added in `Conv2dMergeGrowingModule`:
  - Properties: `input_volume`, `out_channels`, `in_features` (warns), `out_features`, `output_size`, `padding`, `stride`, `dilation`, `unfolded_extended_activity`.
  - Methods: `set_next_modules`, `set_previous_modules`, `construct_full_activity`, `compute_previous_s_update`, `compute_previous_m_update`, `compute_s_update`, `update_size`.
- Exercise all warning paths, assertion errors, and NotImplementedError branches reachable without changing production code.
- Keep tests deterministic (seed, fixed sizes) and device-aware via `global_device()`.
- Don’t change public APIs or core module logic.
- Validate shapes and a few value invariants; prefer shape and type checks over brittle value equality.
- Ensure tests are isolated; avoid mutating shared state across tests.


## Environment and tooling

- Always run shell commands inside the conda environment named `gromo` (it contains all required project packages). Activate it before running tests or coverage.
- Always prioritize relevant MCP tool use to maximize precision and efficiency (for reading/editing files, searching, running tests, capturing coverage, focusing editors, and checking diagnostics). Avoid assumptions—gather context with tools first, then change code.


## High-level test matrix (what to cover and how)

Below, each row is a concrete test to add or extend. Use new test methods inside the existing `TestConv2dMergeGrowingModule` class unless stated otherwise.

1) input_volume full coverage
- Case A: With `previous_modules` set (already covered). Assert equals first previous `out_features` and equals `out_features` of merge.
- Case B: With `previous_modules = []`: expect `UserWarning` and return `-1`.
- Case C: With `_input_volume` explicitly set: returns the override without warnings.

2) out_channels and in_features
- Add a test asserting:
  - `merge.out_channels == merge.in_channels`.
  - Accessing `merge.in_features` emits `UserWarning` and equals `merge.in_channels`.

3) output_size
- Assert `merge.output_size == merge.input_size` (after init). Use square inputs to avoid ambiguity.

4) padding/stride/dilation – all branches
- Baseline: with next `Conv2dGrowingModule` (default in setUp), `padding == next.layer.padding`, `stride == next.layer.stride`, `dilation == next.layer.dilation`.
- No next: `merge.set_next_modules([])` → each property warns and returns fallback: `padding=0`, `stride=1`, `dilation=1`.
- Linear next: set next to a `LinearGrowingModule(merge.out_features, 1)` → `padding=0`, `stride=1`, `dilation=1`.
- NotImplementedError branches: create a fresh `Conv2dMergeGrowingModule` instance and directly assign `merge.next_modules = [torch.nn.ReLU()]` (or another dummy object that is neither `Conv2dGrowingModule` nor `LinearGrowingModule`), then calling each property should raise `NotImplementedError`. Do NOT use `set_next_modules` here (it rejects invalid types); assign directly to cover the else-branch. Clean up by using a fresh instance per property to avoid side effects.

5) unfolded_extended_activity – conv vs linear branches
- Conv branch (default next is conv): ensure `torch.nn.functional.unfold` path is used and a bias column is concatenated; assert shape `(N, D_with_bias, L)` and last channel equals ones.
- Linear branch: set next to `LinearGrowingModule`, set `merge.activity = merge.activity.flatten(1)` before access; assert shape `(N, D_with_bias)` and last column equals ones. Existing test already hits this; extend with a one-check that the last column is all ones.

6) set_next_modules – type and shape validation
- Already tested: invalid types (string, torch.nn.Linear) raise `NotImplementedError` and wrong `LinearGrowingModule` input size raises `AssertionError`.
- Add: valid next `Conv2dGrowingModule` updates that module’s `input_size` to `merge.output_size` (assert equality). Also test passing a next `Conv2dMergeGrowingModule` (channels aligned) updates its `input_size` accordingly.
- Add: re-calling `set_next_modules` after `merge.tensor_s` has collected samples should emit `UserWarning` (trigger by running a tiny forward/backward + update_computation before calling again).

7) set_previous_modules – type/shape and warnings
- Already tested: wrong type raises `TypeError`, channel mismatch raises `ValueError` and totals updated with one and two previous modules.
- Add: after an initial compute/update, re-calling `set_previous_modules(...)` should warn for non-empty `previous_tensor_s` and `previous_tensor_m`. Use `assertWarns(UserWarning)` and check both warnings by ensuring both previous tensors have non-zero samples before the call.
- Add: when `previous_modules` is empty, both `previous_tensor_s` and `previous_tensor_m` are set to `None`.

8) construct_full_activity
- Already tested for a single previous module. Add multi-previous case: set two identical conv previous modules; check resulting shape equals `(N, sum(in_features + bias), L)` and that slices match each module’s `unfolded_extended_input` (validate a small slice equality to ensure correct concatenation order).

9) compute_previous_s_update and compute_previous_m_update
- Already covered for shape and batch size. Add one check that S is symmetric: `S.T == S` (within atol) for the typical case to sanity check einsum indices. For M, check dtype and shape only.

10) compute_s_update – conv vs linear vs invalid
- Already covered conv and linear branches. Add an invalid-type next module via direct `merge.next_modules = [object()]` and assert `NotImplementedError` is raised when calling `compute_s_update()` (ensure `merge.store_activity=True` and `merge.activity` set appropriately to satisfy asserts).

11) update_size – reallocate tensors and channel propagation
- Start with one previous conv; capture `previous_tensor_s._shape` and `previous_tensor_m._shape` and `in_channels`.
- Then set two previous conv modules (same channels), call `update_size()` and assert:
  - `merge.in_channels` updated to `previous_modules[0].out_channels`.
  - `previous_tensor_s._shape` updated to `(sum_in_features_with_bias, sum_in_features_with_bias)`.
  - `previous_tensor_m._shape` updated to `(sum_in_features_with_bias, merge.in_channels)`.
- Finally, set `previous_modules=[]`, call `update_size()` and assert `previous_tensor_s is None` and `previous_tensor_m is None`.


## Concrete edits to implement (step-by-step)

Add the following new test methods to `tests/test_conv2d_growing_module.py` within `class TestConv2dMergeGrowingModule`:

- `test_out_channels_and_in_features_warning`
- `test_output_size_property`
- `test_padding_stride_dilation_notimplemented_for_invalid_next`
- `test_unfolded_extended_activity_has_bias_column`
- `test_set_next_modules_updates_input_size_and_warns_when_already_sampled`
- `test_set_previous_modules_warns_on_existing_prev_stats_and_handles_empty`
- `test_construct_full_activity_with_two_prev_modules`
- `test_compute_previous_s_update_symmetry`
- `test_compute_s_update_invalid_next_raises`
- `test_update_size_reallocates_previous_stats`

Notes and hints per test:

- Always create fresh module instances inside each test when you need to directly tamper with `next_modules` to reach invalid branches.
- Use `with self.assertWarns(UserWarning):` around calls expected to warn.
- For invalid next types in property tests, use a vanilla `torch.nn.Module` like `torch.nn.ReLU()` or a minimal dummy class so that `isinstance(..., LinearGrowingModule)` is false. Assign via `merge.next_modules = [dummy]` (avoid `set_next_modules`).
- When testing `compute_s_update` invalid branch, set `merge.store_activity=True` and provide a valid `merge.activity` (e.g., reusing prior forward pass or a shaped tensor) to satisfy asserts before the NotImplementedError branch triggers.
- When verifying the bias column in `unfolded_extended_activity`, check `(unfolded[:, -1] == 1).all()` (or the appropriate dim depending on shape).


## Implementation outline (GPT-5 Mini friendly)

Follow this minimal recipe for each new test:

1) Arrange
- Build small conv previous/next modules as needed. Use the same sizes as existing setup to stay consistent: input_size=(8,8), kernel=3, in_channels=2, hidden=3, out=2, batch=5.
- Link modules: `prev -> merge -> next`, and optionally a small linear head.
- Forward a batch `x` of random data, compute a tiny loss, call `backward()`, then `update_computation()` on all custom modules to populate stats.

2) Act
- Call the property/function under test.
- For branches, mutate `merge.next_modules` directly or rebuild a fresh `merge` as explained above.

3) Assert
- Use `assertEqual`/`assertTrue`/`assertAllClose` to validate shapes, dtypes, and selected invariants.
- Use `assertWarns(UserWarning)` and `assertRaises` for warning/error branches.

Keep each test short and focused on one behavior.


## Example patterns to reuse

- Building a second previous module for multi-prev scenarios:
  - Create another `Conv2dGrowingModule` with the same signature as `self.prev` and assign `merge.set_previous_modules([prev_a, prev_b])`.
- Bias-column check in unfolded:
  - For conv next: `unfolded.shape == (N, D_with_bias, L)` and `(unfolded[:, -1, :] == 1).all()`.
  - For linear next: `unfolded.shape == (N, D_with_bias)` and `(unfolded[:, -1] == 1).all()`.


## Coverage check and guardrails

- After implementing tests, run the suite and verify coverage exceeds 93.16% for the newly introduced lines and aim at 100%. If any missed branches remain, inspect uncovered lines in `conv2d_growing_module.py` around `Conv2dMergeGrowingModule` and add a narrow test.
- Keep running time short (< 5s) by using small tensors/batch sizes and avoiding loops.

Optional commands to run locally:

```bash
# Activate project environment
conda activate gromo

# Run tests
pytest -q

# Generate coverage (XML/HTML already configured in repo)
pytest -q --cov=src/gromo/modules/conv2d_growing_module.py --cov-report=term-missing
```


## Acceptance criteria

- All added tests pass on macOS with torch device resolved by `global_device()`.
- All `Conv2dMergeGrowingModule` branches enumerated above are executed at least once; warning/exception paths asserted where applicable.
- Overall coverage for newly added lines in this PR is >= 93.16%; target 100% for `Conv2dMergeGrowingModule`.
- No flaky behavior; tests are deterministic with fixed seeds where random tensors are created.
