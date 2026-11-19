<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 16: the framework tests](#slice-16-the-framework-tests)
  - [Findings](#findings)
    - [16.1 The serial `file` runs never read an input lazily, and `test_convert_file_input` asserts the behaviour that the units rule forbids](#161-the-serial-file-runs-never-read-an-input-lazily-and-test_convert_file_input-asserts-the-behaviour-that-the-units-rule-forbids)
    - [16.2 No guard of `ModelMPI._build` has a test, except the units checks; the rule on a `parameter` on `time` is one of them](#162-no-guard-of-modelmpi_build-has-a-test-except-the-units-checks-the-rule-on-a-parameter-on-time-is-one-of-them)
    - [16.3 Twelve guards of the serial path have no test showing that they raise](#163-twelve-guards-of-the-serial-path-have-no-test-showing-that-they-raise)
    - [16.4 The buffer-sharing checks compare a dataset variable with itself, and the MPI probe reads `_obj`, not the `_refs` the kernels use](#164-the-buffer-sharing-checks-compare-a-dataset-variable-with-itself-and-the-mpi-probe-reads-_obj-not-the-_refs-the-kernels-use)
    - [16.5 The conversion of a parameter at the Model boundary is checked through its attrs in serial and not at all under MPI: the converted parameter is read by no kernel](#165-the-conversion-of-a-parameter-at-the-model-boundary-is-checked-through-its-attrs-in-serial-and-not-at-all-under-mpi-the-converted-parameter-is-read-by-no-kernel)
    - [16.6 `test_finalized_output_is_silent` cannot fail on the warning it guards against, and the unfinalized test does not meet the case of finding 2.2](#166-test_finalized_output_is_silent-cannot-fail-on-the-warning-it-guards-against-and-the-unfinalized-test-does-not-meet-the-case-of-finding-22)
    - [16.7 `tests/test_units_check_mpi.py` runs its collective calls inside the test bodies, and `assert all(comm.allgather(True))` cannot fail](#167-teststest_units_check_mpipy-runs-its-collective-calls-inside-the-test-bodies-and-assert-allcommallgathertrue-cannot-fail)
    - [16.8 `Model.input_spec` is never compared with what assembly resolves; four of its branches are unrun, and the "partition" assert checks less than its comment says](#168-modelinput_spec-is-never-compared-with-what-assembly-resolves-four-of-its-branches-are-unrun-and-the-partition-assert-checks-less-than-its-comment-says)
    - [16.9 `config.load_model_yaml` is tested only under `mpirun`, through one yaml](#169-configload_model_yaml-is-tested-only-under-mpirun-through-one-yaml)
    - [16.10 The lazy-file case of `test_ref_behaviors.py` asserts `is not`, which cannot tell a copy from a second view of one array; nothing pins alignment by label](#1610-the-lazy-file-case-of-test_ref_behaviorspy-asserts-is-not-which-cannot-tell-a-copy-from-a-second-view-of-one-array-nothing-pins-alignment-by-label)
    - [16.11 The budget tests: one term per component, one leaking rank, `budget_rtol` never set, the reported positions never read](#1611-the-budget-tests-one-term-per-component-one-leaking-rank-budget_rtol-never-set-the-reported-positions-never-read)
    - [16.12 `test_units.py` checks that each vocabulary expression parses, not what it means; no test compares two declarations of one name; the registry walk can miss a package class](#1612-test_unitspy-checks-that-each-vocabulary-expression-parses-not-what-it-means-no-test-compares-two-declarations-of-one-name-the-registry-walk-can-miss-a-package-class)
    - [16.13 What the toy regressions cannot show](#1613-what-the-toy-regressions-cannot-show)
    - [16.14 `test_time.py` has no value for `current_epiweek` and no case of a time axis the class should refuse](#1614-test_timepy-has-no-value-for-current_epiweek-and-no-case-of-a-time-axis-the-class-should-refuse)
    - [16.15 The contract-graph tests assert substrings of the text; a sub-model is never drawn](#1615-the-contract-graph-tests-assert-substrings-of-the-text-a-sub-model-is-never-drawn)
    - [16.16 `topological_order` is tested for a valid order, not for pywatershed's order](#1616-topological_order-is-tested-for-a-valid-order-not-for-pywatersheds-order)
    - [16.17 Comments and docstrings that do not match the code, duplicated setup, style](#1617-comments-and-docstrings-that-do-not-match-the-code-duplicated-setup-style)
  - [Code findings of slices 1-4: why did no test catch this?](#code-findings-of-slices-1-4-why-did-no-test-catch-this)
  - [Checklist T, item by item](#checklist-t-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 16: the framework tests

- pws_phoenix commit: 01dcff2 (branch `review_001`; read from
  `.git/HEAD` and `.git/refs/heads/review_001` as text). The earlier
  slices record that the commits since e33e217 are review commits.
  The 18 files total 3,737 lines, as in the plan.
- Paths (under `tests/`): `test_budget.py` (342),
  `test_budget_mpi.py` (133), `test_discretization.py` (199),
  `test_input_spec.py` (314), `test_model_contract_graph.py` (233),
  `test_model_step.py` (100), `test_output_finalize.py` (79),
  `test_process_fields.py` (110), `test_ref_behaviors.py` (271),
  `test_time.py` (73), `test_two_grid.py` (150),
  `test_two_grid_mpi.py` (417), `test_units.py` (187),
  `test_units_check.py` (405), `test_units_check_mpi.py` (229),
  `test_up_low_regression.py` (202),
  `test_up_low_regression_mpi.py` (181),
  `test_benchmark_coverage.py` (112)
- Checklist: T. Covers code slices 1-4. Reviewed 2026-11-19.
- Method: `/code-review high` on the 18 files (10 candidates, and 4
  it ranked below its cap of 10). Kept: all 10, as 16.4 (the
  candidate on the identity checks and the one on `UpperProbe`,
  merged), 16.5 (two candidates on the converted parameter, merged),
  16.6, 16.7, 16.8, 16.12 (the name-keyed registry) and 16.17 (the
  toys that read `self._obj[name].values`, the duplicated setup, the
  lower-ranked style items; the lower-ranked budget item is in
  16.11). Added by the read: 16.1, 16.2, 16.3,
  16.9, 16.10, 16.11, 16.13, 16.14, 16.15, 16.16. Then every file of
  the slice, whole, against checklist T.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/process.py:138-154`, `:253-479`;
    `pws_phoenix/model.py:96-205`, `:330-419`, `:700-766`,
    `:812-943`, `:1020-1026`, `:1046-1052`, `:1090-1116`,
    `:1270-1298`, `:1396-1422`, `:1468-1476`, `:1604-1634`,
    `:1674-1731`; `pws_phoenix/data_io.py:28-38`, `:50-66`,
    `:76-101`, `:148-156`, `:174-188`, `:268-282`;
    `pws_phoenix/discretization.py:80-112`, `:156-163`;
    `pws_phoenix/maps.py:106-114`; `pws_phoenix/units.py:122-192`,
    `:270-281`; `pws_phoenix/budget.py:139-190`;
    `pws_phoenix/clock.py:87-114`;
    `pws_phoenix/model_contract_graph.py:150-168`, `:395-401`;
    `pws_phoenix/processes_concrete.py:150-182`;
    `pws_phoenix/prms_translate/assemble.py:68-90`;
    `tests/conftest.py:219-222`, `:299-300`, `:330-465`;
    `pytest.ini`; the import lines of `benchmarks/_common.py`.
  - Before the review: `coverage_serial.txt` (whole); the findings
    files of slices 1-4 in full; the notes for slice 16 at the end
    of the slice 15 file.
- Run: the `/code-review` session ran `tests/test_ref_behaviors.py`
  and `tests/test_output_finalize.py` in env pwpx2 (17 passed). No
  test was changed to see whether it then fails, so a finding that
  says a test "cannot fail" rests on reading and is marked "not
  run". Statements about text in the tests come from text scans
  (grep), listed with the finding.

Findings are ordered by consequence: first the tests that pass for
a reason other than the one they name (16.1, 16.4-16.8), with the
guards that have no test (16.2, 16.3) beside them; then what the
slice does not reach (16.9-16.16); then comments and style (16.17).

## Findings

### 16.1 The serial `file` runs never read an input lazily, and `test_convert_file_input` asserts the behaviour that the units rule forbids

- Where: `tests/test_up_low_regression.py:31-65`;
  `tests/test_units_check.py:245-265`;
  `pws_phoenix/data_io.py:61-63`, `:83`; `pws_phoenix/model.py:107`
- Code:

      `memory` passes in-memory DataArrays/Dataset; `file` writes one NetCDF
      per input (so the serial Model's path-opening + shared-file dedup are
      exercised) and passes the paths.

      """A file-backed read-only input is converted too: Input.__init__
      sets the read-only flag through `.values`, which materializes the
      lazily opened array (pre-existing behavior), so it is resident at
      the time of the check.
      ...
      got = _run(stamped, two_grid_weights, {"units": "convert"}, n_time)
      np.testing.assert_allclose(got["flow"], answers["flow"], rtol=1e-12)

- Consequence: `load_all` defaults to False
  (`self._load_all = control.get("load_all", False)`), and no test
  names `load_all` (text scan of `tests/`). A file given as a `Path`
  is therefore opened lazily, and
  `self.data.values.flags.writeable = False` then reads the whole
  variable (finding 2.4). So:
  - The `file` half of `test_model_regression` runs the in-memory
    branch of `Input.advance`. The lazy branch
    (`pws_phoenix/data_io.py:83`) and both `load=True` paths
    (`:37`, `:61`, `pws_phoenix/model.py:109`) are run by no serial
    test (`coverage_serial.txt`). What `file` adds over `memory` is
    the opening of the files and the shared `parameters.nc`.
  - `test_convert_file_input` asserts that a file input in other
    units IS converted under `"convert"`. `CLAUDE.md` ("Units
    mode"): "A streamed or lazy input that would need converting
    RAISES: pre-process the file." The test pins the opposite, and
    says why in its docstring. When 2.4 is fixed this test fails,
    and it is the only serial test of a file input under units
    mode.
  - In the serial `Model` no test reaches `convertible=False`. The
    refusal is tested on the function (`test_reconcile_messages`,
    `match="streamed/lazy"`) and under MPI
    (`tests/test_units_check_mpi.py:175`).
- Source: coverage, checklist T (a test that cannot fail for the
  reason it names)
- Outcome: open

### 16.2 No guard of `ModelMPI._build` has a test, except the units checks; the rule on a `parameter` on `time` is one of them

- Where: `tests/*_mpi.py` (text scan for `pytest.raises`: three
  matches, all in `tests/test_units_check_mpi.py:175`, `:180`,
  `:199`); `pws_phoenix/model.py:1286-1292`, `:1303-1312`,
  `:1401-1406`, `:1416-1422`, `:1471-1476`, `:1608-1613`,
  `:1615-1619`, `:1629-1634`
- Code:

      raise ValueError(
          f"discretizations {bad_discs}: pass Discretizations for "
          f"SERIAL grids only ({sorted(serial_grids)}); the "

      f"Map '{map_name}' targets the distributed grid "
      f"'{mpi_grid}': serial->distributed maps (scatter) "
      "are not implemented (Step B is distributed->serial)."

      f"Map '{map_name}' crosses the parallel boundary "
      f"from '{mpi_grid}': use MapMPI, not Map."

- Consequence: `coverage_serial.txt` shows `:1270-1662` run by no
  serial test, and no MPI test file expects any of these raises.
  Not shown to raise: a `parameter` declared on the model `time`
  axis (`CLAUDE.md`: "is a misdeclaration and raises in
  `ModelMPI._build`"); a `Discretization` passed for the
  distributed grid (4.4); an input absent from `input_file`; a
  streamed input on the wrong dims; an output variable owned by no
  grid or by two; a Map that targets the distributed grid; a plain
  `Map` from the distributed grid; a serial-grid output variable
  without `output_serial_zarr`. Each of these depends on the
  declarations, `control` and file metadata only, so every rank
  raises together and a test would not hang (the slice 3 checklist
  checked this). A change that removes one of them passes the whole
  sweep. Also without a test under MPI: `ModelMPI.run(dt,
  n_steps=...)` and a second `run()` (3.13); `test_model_step.py`
  tests `step` and `run` for the serial `Model` only, and its one
  MPI case calls `ModelMPI.step(None, DT)` on the class.
- Source: checklist T (a guard with no test showing that it raises)
- Outcome: open

### 16.3 Twelve guards of the serial path have no test showing that they raise

- Where (the raise, then what it guards; all unrun in
  `coverage_serial.txt`, and a text scan of `tests/` for a phrase
  of each message found none):
  - `pws_phoenix/model.py:157`: `output_var_names` without
    `output_serial_zarr`, or the reverse.
  - `pws_phoenix/model.py:715`: a declared parameter that is in
    neither the grid's parameters nor the supplied ones.
  - `pws_phoenix/model.py:760`: an input on a spatial dim other
    than the grid's.
  - `pws_phoenix/model.py:826`, `:833`, `:850`, `:860`, `:875`: in
    `_resolve_maps`, an unknown grid, a missing source variable, a
    weights shape that does not fit, an unused Map, a source
    variable with two owners.
  - `pws_phoenix/model.py:934`: no time-dimensioned input to set
    the clock from.
  - `pws_phoenix/data_io.py:153`: an output store without the
    `.zarr` suffix. `:186`: an output variable that no process
    owns.
  - `pws_phoenix/maps.py:112`: a `grid=` or `variable=` dict with
    more than one entry.
  - `pws_phoenix/discretization.py:85`: `topo_order` without
    `parameters`.
  - `pws_phoenix/units.py:279`: `"convert"` on an array that is not
    float.
- Code:

      raise ValueError(
          f"process '{proc_name}': input '{ii}' arrives on "
          f"dims {current.dims} but grid '{grid}' expects "

      f"Map '{map_name}' is unused: '{mm.target_var}' on grid "

- Consequence: `CLAUDE.md` states several of these as rules ("A
  mismatched input dim raises at build"; "`_resolve_maps` also
  raises on a weights shape that does not fit the grid sizes, an
  unknown grid, a missing source variable, and an unused Map").
  Of the seven raises of `_resolve_maps`, two have a test: the
  writer placed after the first consumer
  (`tests/test_two_grid.py:99-115`) and the units of a Map
  (`tests/test_units_check.py:398-405`). Also unrun: the warning
  for a missing `time_chunk_size` (`pws_phoenix/model.py:164-165`).
  Guards that do have a test in this slice: an unresolved input, an
  unknown `Discretization`, the values of `control["budget"]` and
  `control["units"]`, a step past the end, a step after `finalize`,
  the budget declaration errors, a missing to-index variable.
- Source: coverage, checklist T (a guard with no test showing that
  it raises)
- Outcome: open

### 16.4 The buffer-sharing checks compare a dataset variable with itself, and the MPI probe reads `_obj`, not the `_refs` the kernels use

- Where: `tests/test_up_low_regression.py:160-167`;
  `tests/test_up_low_regression_mpi.py:136-140`;
  `tests/test_two_grid_mpi.py:310-349`;
  `pws_phoenix/process.py:332-335`, `:360-363`;
  `pws_phoenix/model.py:1713-1715`
- Code:

      assert (
          model.model_dict["upper"]["param_shared_name"].values
          is model.model_dict["lower"]["param_shared_name"].values
      ), "Shared parameter references broken"

      "shared_flow": upper._obj["flow"].values is lower._obj["flow"].values,

      def __getitem__(self, key: str) -> xr.DataArray:
          return self._obj[key]

      self.addresses.setdefault(name, set()).add(
          self._obj[name].values.ctypes.data
      )

- Consequence: `Upper` and `Lower` are bound to one dataset (serial:
  the grid's; MPI: the last `step`, `for proc in mpi_procs:
  proc._obj = step`). Each check is `ds[name].values is
  ds[name].values`, which is true for any in-memory variable. A
  kernel reads `self._refs[name]`, captured once in
  `Process.__init__`. If the variable were replaced in the grid
  dataset between the binding of `Upper` and of `Lower` (label
  alignment, 3.3; a supplied input ahead of its producer, 3.11),
  the two `_refs["flow"]` would be different arrays and these
  asserts would pass. For `flow` the answers would show it; for
  `param_shared_name`, which no kernel reads (15.17), nothing
  would. `UpperProbe` records, every step, the address of
  `self._obj[name].values`: it shows that `step` exposes one array
  per name for the whole run. It does not compare that array with
  `self._refs[name]`, the one captured at build before `iterate`
  began, which is the invariant the comment in
  `pws_phoenix/process.py:330-331` states ("The MPI rebind does NOT
  recapture -- the arrays behind the names never move"). The
  comment block at `tests/test_two_grid_mpi.py:310-328` still
  describes `self._obj[name].values` as "the normal path" and a
  reference cache as a future step. A fix: compare
  `upper._refs[name] is lower._refs[name]`, and add
  `self._refs[name].ctypes.data` to the probe's set. Not run.
- Source: /code-review
- Outcome: open

### 16.5 The conversion of a parameter at the Model boundary is checked through its attrs in serial and not at all under MPI: the converted parameter is read by no kernel

- Where: `tests/test_units_check.py:230-242`;
  `tests/test_units_check_mpi.py:133-158`
- Code:

      up["param_up_0"] = (up["param_up_0"] * 100.0).assign_attrs(units="percent")
      ...
      assert up["param_up_0"].attrs["units"] == "fraction"

      """param_up_0 supplied x100 as 'percent': converted in place on each
      rank's decomposed slice -> the declared-units answers."""

- Consequence: the toy kernels read `param_up_1` and the forcings
  (15.17), so "the same answers" holds whatever is done to
  `param_up_0`. Serial: the test would pass if the Model set the
  attrs and left the values, or scaled them by 100 instead of 0.01;
  `forcing_up` does get a value check (`:239-241`). MPI: nothing
  about `param_up_0` is asserted, so the test shows that
  `"convert"` does not raise. The function-level
  `test_reconcile_converts_in_place_and_restores_flag` checks
  values, for one call of `units.reconcile`; the code that decides
  which arrays are passed to it, and how many times
  (`pws_phoenix/model.py:708-729`, `:1430-1440`), is what 1.1 and
  3.6 are about. A fix: convert `param_up_1`, or assert the values.
  Related, by text scan: `units_patch` appears twice in `tests/`
  (`tests/test_units_check.py:207`, `:294`), both with `"check"`
  and a patch equal to the declaration. No test combines a patch
  with `"convert"`, and none builds twice from one dataset.
- Source: /code-review
- Outcome: open

### 16.6 `test_finalized_output_is_silent` cannot fail on the warning it guards against, and the unfinalized test does not meet the case of finding 2.2

- Where: `tests/test_output_finalize.py:74-77`, `:38-62`;
  `pws_phoenix/data_io.py:269-282`; `pytest.ini`
- Code:

      with warnings.catch_warnings():
          warnings.simplefilter("error")
          del model
          gc.collect()

      tail = n_time % TIME_CHUNK_SIZE
      assert tail > 0, "test needs a partial trailing chunk"
      ...
      assert not np.allclose(out[-1], flow_last)

- Consequence:
  - The warning is issued inside `Output.__del__`. With the filter
    set to `"error"` it becomes an exception raised in `__del__`,
    which Python reports as unraisable and does not propagate.
    pytest turns that into a warning of its own after the test, and
    `pytest.ini` has no filter that makes a warning an error. So if
    `finalize()` stopped setting `_finalized` and `__del__` warned,
    this test would pass. The sibling test's form
    (`catch_warnings(record=True)`, then an assert on the messages)
    would fail. Not run.
  - `test_unfinalized_output_warns` needs a partial last chunk
    (chunk 61, 365 steps, tail 60). The case of 2.2, a run that
    stops on a chunk boundary and gets no warning, has no test.
    The last assert says the tail is not the data; it does not say
    what the tail is (0.0, which 2.2 reports as indistinguishable
    from data).
  - No test of this slice reads the store's `time` coordinate or
    its attrs (2.1, 2.13), and no test file uses `restart_read`
    with `output_serial_zarr` (text scan), which is why 2.1 was
    not caught.
- Source: /code-review (the first point), checklist T
- Outcome: open

### 16.7 `tests/test_units_check_mpi.py` runs its collective calls inside the test bodies, and `assert all(comm.allgather(True))` cannot fail

- Where: `tests/test_units_check_mpi.py:144-158`, `:174-178`,
  `:198-229`
- Code:

      with pytest.raises(ValueError, match="streamed input.*'forcing_up'"):
          _build(data_dir, two_grid_toy, two_grid_weights, "feet")
      raised_everywhere = comm.allgather(True)
      assert all(raised_everywhere)

- Consequence: every rank that reaches the `allgather` sends
  `True`, so the assert is true whenever the line returns. A rank
  on which `_build` did not raise fails in `pytest.raises` and
  leaves the test; the other ranks then wait in `allgather` (no
  launch has a time limit, 15.13). The same holds for the asserts
  placed between `model.run`, `model.finalize()` and
  `comm.Barrier()` (`:150`, `:221`). The other MPI files of the
  slice keep every collective call in a module fixture and say why
  ("a failing rank can never interrupt a collective and hang the
  others"). In practice the raises here depend on metadata and the
  asserted array is the replicated segment state, so the ranks
  fail together; the form still differs from the rule the sibling
  files state.
- Source: /code-review
- Outcome: open

### 16.8 `Model.input_spec` is never compared with what assembly resolves; four of its branches are unrun, and the "partition" assert checks less than its comment says

- Where: `tests/test_input_spec.py:96-111`, `:258-269`;
  `pws_phoenix/model.py:353-356`, `:361-365`, `:393-397`,
  `:403-414`
- Code:

      # exactly what test_prms_stream_temp.py supplies from disk
      assert set(seg["external_inputs"]) == {
          "seg_outflow",

      # the categories partition the declared input names
      ...
      assert len(cats) == (
          len(req["external_inputs"])
          + len(opt["internal_inputs"])
          + len(opt["map_fed_inputs"])
      )

- Consequence:
  - `input_spec` is called in `tests/test_input_spec.py` and,
    through the graph, in `tests/test_model_contract_graph.py`
    (text scan); both work from declarations alone. The two lists
    said to be "exactly what" two other tests supply are typed by
    hand. No test builds a `Model` and compares what it resolved
    with the specification, which is the question the slice 3
    notes ask. The cases where the two differ by construction are
    3.11 (a supplied input ahead of its producer) and 3.17 (an
    input satisfied by a same-named parameter).
  - Unrun (`coverage_serial.txt`): the content of
    `required_initial_state` (the test asserts the key exists,
    `:84`); a parameter one process declares with a derivation and
    another without; an input satisfied by an internal parameter;
    an input satisfied by a same-named supplied parameter.
  - The assert shows the three categories do not overlap. A
    declared input that is in none of them passes. Comparing the
    union with `get_inputs()` plus `get_mutable_inputs()` of the
    grid's processes would do what the comment says.
- Source: /code-review (the third point), coverage
- Outcome: open

### 16.9 `config.load_model_yaml` is tested only under `mpirun`, through one yaml

- Where: `tests/test_two_grid_mpi.py:124-161`, `:178-181`;
  `pws_phoenix/config.py` (17% in `coverage_serial.txt`:
  `:89-90`, `:96-99`, `:106-115`, `:130-172` unrun)
- Code:

      process_dict, control, maps, _ = load_model_yaml(
          mpi_paths["yaml_file"]
      )

- Consequence: the loader has no MPI code, yet its one test needs
  four ranks. That yaml has no `discretizations:` section, file
  weights, no `restart_read`, and the fourth return value is
  thrown away. Not reached by any test: the body of
  `_load_discretizations`, inline weights, both raises (`:141`,
  `:162`). This is why 4.1 (a `restart_read` not resolved against
  the yaml's directory) and 4.4 (loaded discretizations that
  `ModelMPI` rejects) were not caught. A serial test that loads a
  yaml into `Model` would cover the same lines without `mpirun`.
  This confirms the slice 4 note.
- Source: coverage, checklist T
- Outcome: open

### 16.10 The lazy-file case of `test_ref_behaviors.py` asserts `is not`, which cannot tell a copy from a second view of one array; nothing pins alignment by label

- Where: `tests/test_ref_behaviors.py:126-138`, `:16-21`, `:94-103`
- Code:

      # Dataset. Each .values access reads from disk -- fresh numpy array
      # each time, so buffer identity is NOT preserved across the selection.
      ...
      assert subset["v1"].values is not ds["v1"].values

- Consequence: the run recorded in 2.4 (JLM, 2026-11-15) showed,
  for a lazily opened array, that after one `.values` the data stay
  in memory (`_in_memory` is True, later `.values` share memory)
  while `a is da.values` is still False. So `is not` is true here
  whether or not a second read from disk happens, and the comment
  "reads from disk ... each time" is not what that run found. The
  assert that would separate the two is `np.shares_memory`. What
  the file does establish is that `.load()` is needed for object
  identity (`:140-172`), which is what `Process._refs` and the
  identity checks in the other tests rely on. Not run for the
  `Dataset` selection case. Separately: the file pins that
  `Dataset.__setitem__` keeps the buffer for a new key, an existing
  key, another dtype, and a non-index coordinate
  (`space_coord`). It has no case with an index coordinate on the
  dim whose labels differ from the dataset's, where xarray
  reindexes and stores a new array (3.3). This is the slice 2 note,
  answered.
- Source: checklist T (a test that cannot fail)
- Outcome: open

### 16.11 The budget tests: one term per component, one leaking rank, `budget_rtol` never set, the reported positions never read

- Where: `tests/test_budget.py:44-50`, `:169-193`, `:299-311`,
  `:338-342`; `tests/test_budget_mpi.py:34-41`, `:113-121`
- Code:

      "inputs": ("inflow",),
      "outputs": ("outflow",),
      "storage_changes": ("storage_change",),

      _terms(inputs=("nope",), outputs=(), storage_changes=()),

      expected = f"{runs['leak_count']} location(s)"

      def test_leaky_none_is_silent():

- Consequence:
  - Every component of every toy budget has exactly one term, a
    writable float64 `(space,)` array. `_sum_terms` over two or
    more terms, and over a mix of read-only and writable arrays
    (1.6), is not run by this slice; whether the port tests run it
    is for slices 19 and 20.
  - An empty tuple appears only in declarations that fail for
    another reason. No test runs a valid budget with an empty
    required component (1.5).
  - Under MPI the bucket leaks on one rank, so the largest count
    on one rank and the total are the same number: the test cannot
    tell `MPI.MAX` from `MPI.SUM` (3.20). The warning's "first
    indices" are asserted nowhere (1.7); serial asserts the date
    and `max |residual| = nan`.
  - `budget_rtol` is set by no test (text scan); `budget_atol` by
    one. The term `rtol * |outputs + storage_changes|` is run and
    never decides a result.
  - `test_leaky_none_is_silent` asserts the residual values. It
    does not assert that no warning was issued, and pytest does
    not fail on a warning.
  - `test_setup_is_a_subset_rank_leak` asserts three facts about
    module constants; `assert N_TIME >= 1` is true by definition.
- Source: checklist T; the fifth point from /code-review
- Outcome: open

### 16.12 `test_units.py` checks that each vocabulary expression parses, not what it means; no test compares two declarations of one name; the registry walk can miss a package class

- Where: `tests/test_units.py:92-97`, `:132-154`;
  `tests/test_units_check.py:108-119`;
  `pws_phoenix/process.py:323`;
  `tests/test_prms_stream_temp_full_chain.py:119`,
  `tests/test_restart_processes.py:282`
- Code:

      def test_vocabulary_parses(key):
          qq = quantity(key)
          assert qq.magnitude > 0

      for cls in Process._registry.values():
          if not cls.__module__.startswith("pws_phoenix."):
              continue  # a test module's own toy / carrier class

- Consequence:
  - A key whose pint expression is the wrong unit, or carries the
    wrong magnitude, parses. In this slice three conversions are
    pinned by value (`feet` to `inches`, `percent` to `fraction`,
    `degF` to `degC`). The kernel units tests check dimensions, so
    a wrong scale (in `MCM`, `Mcf`, `acres`) would show only in a
    `"convert"` run of that unit. Whether
    `tests/test_prms_units.py` pins more is for slice 22.
  - `test_affine_and_same` asserts `same("fraction", "none")` and
    has no case for `radians` or `degrees` (1.3).
  - No test compares the units that two classes declare for one
    name on one grid (3.16). The slice 3 finding left this to be
    read here: the bodies confirm it.
  - `Process._registry` is keyed by class name. Two test modules
    of the serial sweep, and
    `tests/test_prms_stream_temp_full_chain_mpi.py:108`, define
    `class HumidityCarrier(Process)`, the name of the class in
    `pws_phoenix/prms_translate/assemble.py:71`. In the serial
    sweep the test class registered last replaces the package
    class, the `__module__` filter skips it, and the package class
    is not visited by `test_registered_float_fields_declare_units`.
    Today its one field is a literal `DataArrayMeta(...,
    units="percent")`, which the source scan checks, so nothing is
    missed; a package class built through a helper and shadowed
    the same way would be. `tests/test_benchmark_coverage.py:73-77`
    and `tests/test_restart.py:334-349` filter the same registry.
    Walking `Process.__subclasses__()` recursively does not depend
    on names. This is finding 1.9 seen from the tests: they rely on
    the silent replacement.
- Source: checklist T; the fourth point from /code-review
- Outcome: open

### 16.13 What the toy regressions cannot show

- Where: `tests/test_two_grid.py:73-90`;
  `tests/test_up_low_regression_mpi.py:124-128`, `:176-181`;
  `tests/conftest.py:222`, `:299-300`;
  `pws_phoenix/processes_concrete.py:160-181`
- Code:

      np.testing.assert_allclose(
          model.model_dict["lower"]["storage"].values,
          answers["storage"][-1],
          rtol=1e-12,
      )

      storage_prev_final = np.concatenate(
          comm.allgather(model._ds_mpi_stream["storage_previous"].values.copy())
      )

- Consequence: the toy tests are the ones written to test the
  framework. Beyond 15.16 (no Dec 31) and 15.17 (no static
  parameter reaches an answer):
  - `test_up_low_regression_mpi.py` checks `flow` at every step
    and `storage_previous` at the last, which is `storage` of the
    step before. The last `Lower.calculate` of the run is not
    checked. The docstring gives the reason as "the
    one-`to_file`-var caution"; `tests/test_prms_groundwater_mpi.py`
    streams two variables.
  - The serial two-grid test compares the last step only. The
    state carries the history, so an error at an early step would
    still show unless it decays (factor 0.95 per step over 365
    steps is about 1e-8).
  - The distributed toy grids are 20 and 8 elements on 4 ranks:
    even blocks. Only the budget toy (5 on 4) is uneven. Uneven
    blocks of the full framework path rest on the port tests.
  - All toy state is float64 on `(space,)`, so the float64
    declaration of every state variable under MPI (3.9) changes
    nothing here. `DerivedToy.counts` (int64) is serial only.
  - The toy kernels take `dt` and do not read it.
  - Every toy input shares one time axis and one set of labels on
    the grid dim (3.2, 3.3), and every parameter arrives on the
    declared dims (3.1).
- Source: checklist T (a comparison of only some variables, time
  steps)
- Outcome: open

### 16.14 `test_time.py` has no value for `current_epiweek` and no case of a time axis the class should refuse

- Where: `tests/test_time.py:14-21`, `:57-73`;
  `pws_phoenix/clock.py:87-95`
- Code:

      # (date, year, month, day_of_month, doy, dowy)  -- 2000 is a leap year
      KNOWN = [
          ("2000-01-01", 2000, 1, 1, 1, 93),

- Consequence: `year`, `month`, `day_of_month`, `doy`, `dowy` and
  `jsol` are compared with values typed by hand. `current_epiweek`
  is run by the STARFIT tests and compared with a known week
  nowhere (text scan: the word appears in `tests/` only in two
  comments of `tests/test_mixed_channel_starfit.py`). `dowy` is
  pinned on Oct 1 (1) and not on Sep 30, the last day of a water
  year (365 or 366). Nothing passes an integer or float axis, or
  an index outside the axis (1.4). `Time.__repr__`
  (`pws_phoenix/clock.py:111`) is unrun.
- Source: checklist T
- Outcome: open

### 16.15 The contract-graph tests assert substrings of the text; a sub-model is never drawn

- Where: `tests/test_model_contract_graph.py:103-109`, `:134-136`,
  `:176-180`; `pws_phoenix/model_contract_graph.py:156`, `:166`,
  `:180`, `:188`, `:399`
- Code:

      assert '    upper -. "flow" .-> lower' in mermaid

      # 13 dashed map EDGES (the Time node is also style=dashed)
      ...
      assert len(dashed_edges) == 9  # aggregated pairs (13 variables)

- Consequence: no test hands the dot text to graphviz or the
  mermaid text to a parser, so text that is not valid in either
  language passes (4.7, 4.8). Every graph in the tests is a whole
  model in which each Map source has an owning process: the return
  of `"({grid})"` (`:188`) and the Map whose target grid hosts no
  process (4.9) are not reached. Also unrun: a Map with a
  derivation (`:156`, `:399`), an input satisfied by a parameter
  (`:166`). `test_two_grid_toy` does not look at how `param_up_1`
  on `("month", "space")` is classified (4.10). The comment at
  `:176` says 13 edges and the assert says 9. This confirms the
  slice 4 note.
- Source: coverage, checklist T
- Outcome: open

### 16.16 `topological_order` is tested for a valid order, not for pywatershed's order

- Where: `tests/test_discretization.py:161-174`;
  `pws_phoenix/discretization.py:156-161`
- Code:

      for iseg in range(6):
          if to_seg[iseg] >= 0:
              assert position[iseg] < position[to_seg[iseg]]

- Consequence: `CLAUDE.md`: "`Discretization.topological_order`
  replicates pywatershed's construction EXACTLY (networkx): a
  different valid order changes float accumulation downstream."
  The test accepts any valid order, plus one feature of the
  construction (the isolated segment comes first). No test in
  `tests/` calls `topological_order` outside this file (text
  scan); the others pass `topo_order=` and compare model answers.
  Whether those answers hold the order exactly (the channel at
  its tolerance) is for slice 19; `tests/test_prms_channel.py:391`
  describes its own check as "a valid ordering". Not tested here:
  a one-element grid (`order = [0]`, `:161`), and the inputs of
  2.7 (a to-index outside the grid, an unsigned type, nan, a
  cycle).
- Source: checklist T
- Outcome: open

### 16.17 Comments and docstrings that do not match the code, duplicated setup, style

- Where: `tests/test_time.py:1`;
  `tests/test_up_low_regression_mpi.py:14-16`, `:134-135`;
  `tests/test_up_low_regression.py:106-133`;
  `tests/test_two_grid_mpi.py:192-193`, `:224-226`;
  `tests/test_budget.py:52-71`; `tests/test_discretization.py:54-66`;
  `tests/test_input_spec.py:43-64`, `:126-167`;
  `tests/test_model_contract_graph.py:26-87`;
  `tests/test_benchmark_coverage.py:43-46`;
  `tests/test_ref_behaviors.py:259`; `tests/test_units.py:67`
- Code:

      """Unit tests for the Time clock (globals.py).

      def advance(self) -> None:
          self._obj["storage_prev"].values[:] = self._obj["storage"].values

      pytestmark = pytest.mark.skipif(
          bool(_missing),

- Consequence:
  1. `test_time.py` names `globals.py`; the module is `clock.py`
     (the same slip as 1.14).
  2. `test_up_low_regression_mpi.py`: "the one-`to_file`-var
     caution" is out of date (16.13); "deletes/closees".
  3. `test_zero_copy_inputs` checks the parameters. No input is
     checked for identity or for a read-only flag in any test
     (text scan for `writeable`: parameters and internal
     parameters only), which is why 2.3 and 3.10 were not caught.
  4. Two `TODO: JLM` comments in `test_two_grid_mpi.py` ask open
     design questions (is the `files` style worth keeping; one
     `input_file` for more than one distributed grid).
  5. The toy processes of `test_budget.py` and
     `test_discretization.py` read and write
     `self._obj[name].values` inside `advance`, `calculate` and
     `initialize`. Prime directive 4 forbids that in a step. The
     results are right because these are the arrays `_refs`
     holds; the toys do not use the access path of a real
     process, and they are what a reader may copy.
  6. `test_input_spec.py` and `test_model_contract_graph.py` each
     build the same ten-process dict and thirteen Maps, each with
     its own copy of the humidity carrier, declared without
     `units` (the package class declares `"percent"`).
     `PARAM_NAMES` and the Upper/Lower process dict are repeated
     in three files.
  7. `test_benchmark_coverage.py` skips the module when a
     benchmarked data file is missing. Under
     `tests/run_tests.sh` a skip is an error; in the CI job
     without the data it is a skip (15.12).
  8. Single-letter names (`t`, `k`); 11 lines longer than the 79
     of `ruff.toml` in `test_up_low_regression.py` (3),
     `test_up_low_regression_mpi.py` (5), `test_ref_behaviors.py`
     (3), counted with awk.
- Source: /code-review (items 5, 6, 8), checklist T
- Outcome: open

## Code findings of slices 1-4: why did no test catch this?

One line per finding. "No test" means no test in `tests/`, by the
text scans and the coverage report named above.

Slice 1:
- 1.1: no test uses `units_patch` with `"convert"`, and none builds
  twice from one dataset (16.5).
- 1.3: `same` is tested for `fraction` / `none`, `inches` / `feet`
  and `dday` only (16.12).
- 1.4: every `Time` in the tests has a datetime64 axis and an index
  inside it (16.14).
- 1.5: an empty component appears only in declarations that fail
  for another reason (16.11).
- 1.6: one term per component, all writable float64 (16.11).
- 1.7, and 3.20: one leaking rank; the indices are not read
  (16.11).
- 1.8: `tests/test_units.py` covers the package's classes; no test
  declares a wrong `kind`, a `dtype` outside `_FILL_VALUE`, or
  `restart=True` on an input.
- 1.9: no test defines two classes of one name on purpose; three
  do by accident and depend on the replacement (16.12).
- 1.10: `test_kernels_jitted` asserts the kernels are compiled; no
  test measures an allocation.
- 1.11, 1.12, 1.13, 1.14: cost and documentation; no test bears on
  them.

Slice 2:
- 2.1: no test file has both `restart_read` and
  `output_serial_zarr`; no test of the slice reads the store's
  `time` (16.6).
- 2.2: the unfinalized test needs a partial chunk (16.6).
- 2.3: no test writes to an input buffer or reads its flag
  (16.17 item 3).
- 2.4: `test_convert_file_input` pins the behaviour (16.1).
- 2.5: no MPI test passes a store without `.zarr`; it would hang
  (15.13). The serial raise has no test either (16.3).
- 2.6: the `file` runs use a four-variable parameter file; no test
  checks that a file is closed after `finalize`.
- 2.7: 16.16.
- 2.8: no test measures memory or reads `Map.weights` after
  `bind`.
- 2.9: no test reads the coordinates, attrs or dtype of an input's
  buffer in the grid dataset.
- 2.10: no test builds a `Discretization` from a `Path` and checks
  the file; `test_zero_based_convention` reads the variable that
  `topo_order` wrote into `parameters`, which is the write 2.10
  describes.
- 2.11: no test gives a `MapMPI` to the serial `Model`.
- 2.12: `tests/test_two_grid_mpi.py` compares with a dense product
  at 1e-12, looser than the last bit.
- 2.13: no test reads the attrs of the store (16.6).
- 2.14: no test passes `time_chunk_size=0`, weights that are not
  2-D, or a Map source that is not 1-D.

Slice 3:
- 3.1: every toy parameter arrives on its declared dims, and three
  of four reach no answer (15.17, 16.13).
- 3.2, 3.3: one time axis and one set of labels in every toy
  (16.13); `test_ref_behaviors.py` has no label case (16.10).
- 3.4, 3.5, 3.8, 3.9: slice 17.
- 3.6: the MPI units test has one process on the distributed grid
  and no patch (16.5).
- 3.7: 15.13.
- 3.10: no test reads the write flag of a distributed-grid
  parameter (16.17 item 3).
- 3.11: no test misspells a key, supplies a value for a variable
  without `restart=True`, or supplies an input that a later
  process produces.
- 3.12: the tests of `control` pass a wrong VALUE for `budget` and
  `units`; none passes an unknown key.
- 3.13: 16.2.
- 3.14, 3.19, 3.23, 3.24: cost and structure; no test bears on
  them.
- 3.15: every Map in the tests reads a variable that a process
  owns; no test chains two Maps.
- 3.16: 16.12.
- 3.17: 16.2 (the guards under MPI), 16.8 (the cases where
  `input_spec` and the MPI build differ).
- 3.18: no test supplies an input of one name on two grids.
- 3.21: no test raises during a build and then looks at the open
  files.
- 3.22: no test passes a file as `str`, a class by name to
  `Model`, or a `dt` that disagrees with the axis; the toy kernels
  do not read `dt` (16.13).

Slice 4:
- 4.1, 4.4, and 4.2, 4.3, 4.5, 4.6, 4.12: one yaml, under MPI,
  with none of these features (16.9).
- 4.7, 4.8, 4.9, 4.10: 16.15.
- 4.11: `test_supply_side` asserts the graph's own classification
  per node; no test compares it with
  `spec["required"][grid]["parameters"]`.

## Checklist T, item by item

- A raise or a branch that no test reaches: 16.2, 16.3 (guards);
  16.1 (the lazy and the `load=True` paths); 16.8 (`input_spec`);
  16.9 (`config.py`); 16.15 (the graph). Reached in this slice and
  100% in the report: `pws_phoenix/budget.py`,
  `pws_phoenix/processes_concrete.py`; `pws_phoenix/process.py`
  but for `:474->exit` (`set_restart_state` with empty state,
  slice 17).
- A test that cannot fail: 16.4, 16.6, 16.7; in part 16.1, 16.5,
  16.8, 16.10. Weak by design and saying so:
  `test_declarations_found` (`> 400`), `test_vocabulary_parses`
  (16.12), `test_default_mode_ignores_units` and
  `test_patch_supplies_missing_units` (they assert only that
  nothing raises). The MPI tests that read a global file return
  without an assert on ranks other than 0
  (`if MPI.COMM_WORLD.rank != 0: return`): one rank checks the
  whole file, which is enough.
- A tolerance looser than the process's standard: does not apply
  (no port). The toy comparisons use `rtol=1e-12` against a numpy
  loop written apart from the kernels (`tests/conftest.py:379-463`),
  and `assert_array_equal` between two runs of the model. The
  units tests take their expected values from a run of the model
  in declared units; that is the comparison a conversion test
  needs.
- A comparison where nan on both sides passes:
  `assert_allclose` and `assert_array_equal` treat nan in the same
  place as equal. In this slice the expected side is always a
  finite numpy answer or a finite earlier run, so a nan in the
  result fails. `test_closed_budget` uses
  `np.all(np.abs(res) < 1e-14)`, which a nan fails.
- A comparison of only some output variables, time steps or
  elements: 16.13.
- A serial test with no MPI counterpart where the MPI code
  differs: `test_model_step.py` (16.2, 3.13);
  `test_output_finalize.py` (the rank-0 `Output` without
  `finalize`); `test_units_check.py` against its MPI file, which
  has no patch, no restart file, no missing-attrs case and no Map
  units case, where the MPI units loop is separate code
  (`pws_phoenix/model.py:1430-1440`). `test_discretization.py`:
  grid-owned parameters of a serial grid under MPI are used by
  `tests/test_prms_channel_mpi.py` (slice 19). No MPI code behind
  `test_input_spec.py`, `test_model_contract_graph.py`,
  `test_process_fields.py`, `test_time.py`, `test_units.py`,
  `test_ref_behaviors.py`.
- A guard with no test showing that it raises: 16.2, 16.3. No test
  asserts `pws_phoenix.NUMBA_CACHE is False` (the slice 15 note;
  text scan).

## Notes for later slices

- Slice 17: `tests/test_restart.py:334-349` walks the name-keyed
  registry (16.12). No MPI test file expects a raise outside
  `tests/test_units_check_mpi.py` (16.2), so
  `ModelMPI._raise_if_bad_initial_state` has no test of its
  collective raise unless `tests/test_restart_mpi.py` catches it
  by hand, as `tests/test_budget_mpi.py:94-97` does. Restart with
  units under MPI. `set_restart_state` with empty state
  (`pws_phoenix/process.py:474->exit`).
- Slice 18: `tests/test_mixed_channel_starfit.py` is the only place
  an epiweek matters (16.14); does it pin the week?
- Slice 19: does any channel test hold `segment_order` to
  pywatershed's exact order (16.16)? Do the port tests under MPI
  cover an uneven decomposition with int and bool state (16.13,
  3.9)? `tests/test_prms_channel_mpi.py:323` is the one MPI check
  of a write flag, on an internal parameter.
- Slices 19, 20: which port budgets have two or more terms in one
  component, and whether any mixes read-only and writable arrays
  (16.11, 1.6).
- Slice 22: does `tests/test_prms_units.py` pin the value of more
  conversions than the three of `test_affine_and_same` (16.12)?
- Fix phase, runs that confirm before fixing: remove
  `self._finalized = True` from `Output.finalize` and run
  `tests/test_output_finalize.py` (16.6); in
  `tests/test_up_low_regression.py`, compare `_refs` in place of
  the dataset lookups (16.4); `np.shares_memory` in the lazy case
  of `tests/test_ref_behaviors.py` (16.10); the serial sweep with
  a print of `Process._registry["HumidityCarrier"].__module__`
  inside `test_registered_float_fields_declare_units` (16.12);
  `tests/test_units_check.py` with the conversion of `param_up_0`
  disabled in `Model._add_process_fields` (16.5).
