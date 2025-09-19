# Refactor plan: Ban MergeGrowingModule from MergeGrowingModule.previous_modules / next_modules

One-sentence task receipt

- I'll prepare a concrete, step-by-step refactor plan to change the API so that a `MergeGrowingModule` cannot accept another `MergeGrowingModule` in its `previous_modules` or `next_modules` lists, update child classes and tests, and provide a validation checklist and test commands.

Checklist (requirements)

- [ ] Change the signature of `MergeGrowingModule` constructor to forbid `MergeGrowingModule` in `previous_modules` and `next_modules` (type hints and runtime checks).
- [ ] Update `set_next_modules` and `set_previous_modules` methods in `MergeGrowingModule` to reject `MergeGrowingModule` instances and provide useful errors.
- [ ] Propagate signature and runtime-check changes to `LinearMergeGrowingModule` and `Conv2dMergeGrowingModule` (constructors and `set_*_modules` methods).
- [ ] Audit and update tests that depend on the previous mixed-allowing behavior so CI remains relevant.
- [ ] Run full test-suite inside the `gromo` conda environment and iterate until green.

High-level plan

1. Inspect current implementations and tests (done).
2. Implement signature/type-hint changes and robust runtime checks in the following modules:
   - `src/gromo/modules/growing_module.py` (base `MergeGrowingModule` class)
   - `src/gromo/modules/linear_growing_module.py` (`LinearMergeGrowingModule`)
   - `src/gromo/modules/conv2d_growing_module.py` (`Conv2dMergeGrowingModule`)
3. Run tests; collect failures. Adjust tests that used `MergeGrowingModule` inside `previous_modules`/`next_modules` to either use `GrowingModule` instances or assert that passing a `MergeGrowingModule` raises the new error.
4. Iterate until test-suite passes.

Files to change (concrete edits)

- `src/gromo/modules/growing_module.py`
  - Constructor signature currently:
    - previous_modules: list["MergeGrowingModule | GrowingModule"] | None = None
    - next_modules: list["MergeGrowingModule | GrowingModule"] | None = None
  - Change to:
    - previous_modules: list["GrowingModule"] | None = None
    - next_modules: list["GrowingModule"] | None = None
  - Update `self.previous_modules` and `self.next_modules` annotations likewise to `list[GrowingModule]`.
  - Update `add_next_module` / `add_previous_module` and `set_next_modules` / `set_previous_modules` implementations (the latter two are priority) to check and raise `TypeError` or `AssertionError` if any element is a `MergeGrowingModule`. Example message: "MergeGrowingModule cannot be preceded or followed by another MergeGrowingModule; pass GrowingModule instances instead."
  - Keep other usages that refer to `GrowingModule | MergeGrowingModule` (for `next_module` / `previous_module` attributes in `GrowingModule`) as-is where appropriate, but avoid changing the semantics of single-link `previous_module`/`next_module` attributes in `GrowingModule` unless needed.

- `src/gromo/modules/linear_growing_module.py`
  - Constructor and the methods `set_next_modules` & `set_previous_modules` currently accept mixed lists. Modify their signatures and runtime checks to accept only `GrowingModule` elements (not `MergeGrowingModule`). Update any local isinstance checks that assume `MergeGrowingModule` may appear in those lists.

- `src/gromo/modules/conv2d_growing_module.py`
  - Same as `linear_growing_module.py`: modify constructor type hints and `set_next_modules` / `set_previous_modules` to accept only `GrowingModule` in lists.

Rationale and specific runtime behavior

- Why this change: A `MergeGrowingModule` is a node that logically merges multiple previous (or fans-out to multiple next) regular growing modules; allowing a `MergeGrowingModule` to be connected to another `MergeGrowingModule` makes graph semantics ambiguous (two merges chained) and is not supported by code paths. Banning this at the API/signature level removes a whole class of errors.

- Runtime checks: prefer explicit TypeError over assert (since asserts can be stripped). Example check inside `set_previous_modules`:

    for module in previous_modules:
        if isinstance(module, MergeGrowingModule):
            raise TypeError("MergeGrowingModule.previous_modules must contain GrowingModule instances, not MergeGrowingModule")

- Backwards-compatibility: This change is breaking for code that passed `MergeGrowingModule` instances into those lists. Tests and any user code using that must be updated to either:
  - connect underlying `GrowingModule` nodes instead of `MergeGrowingModule`, or
  - raise an explicit test expectation that passing a `MergeGrowingModule` raises a `TypeError`.

Tests to inspect and update (priority list)

I scanned the repo and found these test files referencing the affected APIs. Update them as part of the refactor.

- tests/test_conv2d_growing_module.py
  - Several tests call `m.set_previous_modules([self.prev])` and `m.set_next_modules([self.next])` where `self.prev` and `self.next` are likely `Conv2dGrowingModule` or `LinearGrowingModule` instances — these are OK. But the file also contains places where `set_previous_modules` or `set_next_modules` are called with `MergeGrowingModule` instances or where tests purposely try to set `[]` and later reassign; search hits show a couple of 'bypass the validation' comments — inspect and either update the tests to expect the new `TypeError` or adjust the test input to use `GrowingModule` instances.

- tests/test_growing_module.py
  - `TestMergeGrowingModule` and several other tests create `LinearMergeGrowingModule` and call `set_previous_modules`. Update tests that assume that `previous_modules`/`next_modules` can contain `MergeGrowingModule` to instead use appropriate `GrowingModule` instances or expect an error.

- tests/test_linear_growing_module.py
  - Many tests instantiate `LinearMergeGrowingModule` and call `set_previous_modules`/`set_next_modules`. Check whether any tests intentionally pass `LinearMergeGrowingModule` instances into these methods (that would now be disallowed). Update either the test inputs to make them `LinearGrowingModule` or update assertions to expect the explicit TypeError message.

How to update tests (recommendations)

- Option A (preferred): Where tests previously connected two merge nodes directly (rare), change them to connect the underlying growing modules instead (e.g., use a `LinearGrowingModule` instead of `LinearMergeGrowingModule` as an item in `previous_modules`/`next_modules`). This preserves the original test intent (validate shape inference, etc.) and preserves guarantees.

- Option B (explicit failure): Where the test is intended to validate error handling, replace the old behavior with an assertion that passing a `MergeGrowingModule` raises `TypeError` (use pytest.raises).

- Option C (if semantic intent was chaining merge modules, and this pattern must be kept): Introduce a small shim constructor or helper that extracts the internal node(s) to connect instead of allowing chains. This is more invasive and not recommended unless you want to reintroduce that feature.

Concrete test edits to consider (examples)

- Example (convert to using a GrowingModule instance):
  - Before: my_merge.set_previous_modules([some_merge])
  - After: my_merge.set_previous_modules([some_merge_underlying_growing_module])

- Example (expecting error):
  - Before: my_merge.set_previous_modules([some_merge])  # allowed previously
  - After:
    with pytest.raises(TypeError):
        my_merge.set_previous_modules([some_merge])

Validation and verification (commands)

- Activate environment and run the test-suite (run these commands in macOS bash):

```bash
conda activate gromo
pytest -q
```

- Run a targeted test file (example):

```bash
conda activate gromo
pytest -q tests/test_conv2d_growing_module.py::TestConv2dMergeGrowingModule::test_set_previous_modules_and_shapes -q
```

Implementation steps (concrete order)

1. Create a small branch and commit (or ensure current PR branch is used). Save work.
2. Edit `src/gromo/modules/growing_module.py`:
   - Change the constructor type-hints.
   - Update `set_previous_modules` and `set_next_modules` runtime checks to reject `MergeGrowingModule` instances.
   - Update internal `self.previous_modules` / `self.next_modules` type annotations.
3. Edit `src/gromo/modules/linear_growing_module.py` and `src/gromo/modules/conv2d_growing_module.py`:
   - Update constructors and `set_*_modules` signature/type-hints to reflect only `GrowingModule` in lists.
   - Adjust any isinstance checks that assumed merged nodes could appear (for example, checks that branch on `MergeGrowingModule` vs `GrowingModule`) to handle the new invariants.
4. Run full test-suite. Expect failures from tests that relied on the old behavior.
5. Update tests as described above. For each failing test, prefer changing inputs to `GrowingModule` instances if the test logic intended to test shape inference or other functional behavior; prefer asserting TypeError if the test was intentionally exercising invalid wiring.
6. Re-run tests; iterate until green.
7. Add a small note in `docs/source/whats_new.rst` documenting the API change.

Edge-cases and extra checks

- There are places where code checks whether previous_module is instance of MergeGrowingModule; those checks are still valid for single-link fields (`GrowingModule.previous_module` may still be a `MergeGrowingModule` if a non-merge module is preceded by one), but you should not accidentally break those checks. Only change the semantics inside `MergeGrowingModule.set_*_modules` and the subclass overrides.

- The codebase uses string-based forward references for type-hints in some places. Update literal type hints carefully to avoid circular import issues.

- Prefer raising `TypeError` (with clear message) to making the argument silently ignored or falling back.

Quality gates

- After changes, run:
  - `pytest -q` (all tests)
  - If runtime/typing errors appear because of typing imports, run flake/mypy if present (project uses pyproject.toml; if mypy is configured, run it).

Requirements coverage mapping

- Change signature/type hints: Planned (modify 3 files).
- Change `set_next_modules`/`set_previous_modules`: Planned (modify base class and subclasses).
- Propagate to `LinearMergeGrowingModule` and `Conv2dMergeGrowingModule`: Planned.
- Update tests in `tests/` using the API: Planned; list of candidate files included above.

Next steps (if you approve)

If you confirm this plan I'll:
- Implement the signature and runtime-check changes in the three source files mentioned.
- Run the full test-suite in the `gromo` conda environment.
- Edit tests iteratively to reflect the new API and re-run tests until green.

Notes

- I scanned the workspace for references; the tests that will most likely need edits are `tests/test_conv2d_growing_module.py`, `tests/test_linear_growing_module.py` and `tests/test_growing_module.py`. I included example edits and a preferred approach (change tests to use `GrowingModule` inputs where the intention is functional behavior; use `pytest.raises` where the test was exercising illegal wiring).

- If you want, I can implement the changes now and run the test-suite. This will require activating the `gromo` conda environment locally; the repo already documents that environment. If tests fail due to environment setup or missing packages, I'll ensure the environment is active and re-run.

---
Generated by automated code review tools and local repo scan; if you want small cosmetic changes (e.g., exact error class/message or prefer ValueError vs TypeError), tell me which preference to use and I'll follow it in the implementation.
