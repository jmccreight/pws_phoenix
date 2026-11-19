<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 15: the test harness](#slice-15-the-test-harness)
  - [Findings](#findings)
    - [15.1 With `func=`, the "compiled" run is the same Python function: seven njit kernels are never run compiled by their units test](#151-with-func-the-compiled-run-is-the-same-python-function-seven-njit-kernels-are-never-run-compiled-by-their-units-test)
    - [15.2 A kernel's return values are thrown away by both runs](#152-a-kernels-return-values-are-thrown-away-by-both-runs)
    - [15.3 A units test compares a kernel with itself, and `coverage_serial.txt` counts the lines only it runs as covered](#153-a-units-test-compares-a-kernel-with-itself-and-coverage_serialtxt-counts-the-lines-only-it-runs-as-covered)
    - [15.4 The units test takes units from the kernel's argument NAMES; nothing compares the positional argument lists in `calculate()` with the signature](#154-the-units-test-takes-units-from-the-kernels-argument-names-nothing-compares-the-positional-argument-lists-in-calculate-with-the-signature)
    - [15.5 A line marked `# units-test: unreachable` is exempted without a check that it did not run](#155-a-line-marked--units-test-unreachable-is-exempted-without-a-check-that-it-did-not-run)
    - [15.6 The harness requires every statement to run, not every branch (known before the start)](#156-the-harness-requires-every-statement-to-run-not-every-branch-known-before-the-start)
    - [15.7 For a field declared `fraction` or `none` the Quantity run checks much less, and no tested kernel has an absolute temperature](#157-for-a-field-declared-fraction-or-none-the-quantity-run-checks-much-less-and-no-tested-kernel-has-an-absolute-temperature)
    - [15.8 No kernel units test for snow, atmosphere, the stream-temperature group, the FlowGraph node types, or hydraulic geometry (known before the start)](#158-no-kernel-units-test-for-snow-atmosphere-the-stream-temperature-group-the-flowgraph-node-types-or-hydraulic-geometry-known-before-the-start)
    - [15.9 The composed FlowGraph class does not store its kernel, and the kernel's arguments are not fields (known before the start)](#159-the-composed-flowgraph-class-does-not-store-its-kernel-and-the-kernels-arguments-are-not-fields-known-before-the-start)
    - [15.10 `check_oracle_pins` passes answer data that carries no marker file, and no test calls the check](#1510-check_oracle_pins-passes-answer-data-that-carries-no-marker-file-and-no-test-calls-the-check)
    - [15.11 On a platform with no pin, marked data stops the whole session, including the tests that read no answers](#1511-on-a-platform-with-no-pin-marked-data-stops-the-whole-session-including-the-tests-that-read-no-answers)
    - [15.12 The no-skips rule exists in `tests/run_tests.sh` alone; the fast CI job runs without it and without the data](#1512-the-no-skips-rule-exists-in-testsrun_testssh-alone-the-fast-ci-job-runs-without-it-and-without-the-data)
    - [15.13 An MPI test that hangs stops the sweep with no result: no launch has a time limit](#1513-an-mpi-test-that-hangs-stops-the-sweep-with-no-result-no-launch-has-a-time-limit)
    - [15.14 Every MPI file runs at one rank count](#1514-every-mpi-file-runs-at-one-rank-count)
    - [15.15 `run_tests.sh`: a pytest filter counts every unselected MPI file as a failed suite, and both `-only` flags together run nothing and print "ALL SUITES PASSED"](#1515-run_testssh-a-pytest-filter-counts-every-unselected-mpi-file-as-a-failed-suite-and-both--only-flags-together-run-nothing-and-print-all-suites-passed)
    - [15.16 The toy time axis stops at Dec 30: 365 steps of the leap year 2000](#1516-the-toy-time-axis-stops-at-dec-30-365-steps-of-the-leap-year-2000)
    - [15.17 Three of the four toy parameters are read by no kernel, so no toy answer depends on a static `(space,)` parameter](#1517-three-of-the-four-toy-parameters-are-read-by-no-kernel-so-no-toy-answer-depends-on-a-static-space-parameter)
    - [15.18 `match="ts"` matches every assertion message, so two must-fail cases do not show which array failed](#1518-matchts-matches-every-assertion-message-so-two-must-fail-cases-do-not-show-which-array-failed)
    - [15.19 The mypy pre-commit hook reports 34 errors at e33e217 (known before the start)](#1519-the-mypy-pre-commit-hook-reports-34-errors-at-e33e217-known-before-the-start)
    - [15.20 Docstrings and comments that do not match the code](#1520-docstrings-and-comments-that-do-not-match-the-code)
  - [Code findings and the harness: why did no test catch this?](#code-findings-and-the-harness-why-did-no-test-catch-this)
  - [Checklist T, item by item](#checklist-t-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 15: the test harness

- pws_phoenix commit: 6a211c3 (branch `review_001`; read from
  `.git/HEAD` and `.git/refs/heads/review_001` as text). The earlier
  slices record that the commits since e33e217 are review commits.
  The four files total 2,554 lines against the plan's 2,560; the
  difference was not explained (no git command was run).
- Paths: `tests/conftest.py` (465), `tests/kernel_units.py` (322),
  `tests/run_tests.sh` (87), `tests/test_kernel_units.py` (1,680)
- Checklist: T. Reviewed 2026-11-18.
- Method: `/code-review high` on the four files (10 candidates).
  Kept: all 10, as 15.1, 15.2, 15.5, 15.10, 15.11, 15.15 (two
  candidates on `run_tests.sh`, merged), 15.16 (two candidates on
  the toy axes, merged), 15.18. Added by the read: 15.3, 15.4, 15.6,
  15.7, 15.8, 15.9, 15.12, 15.13, 15.14, 15.17, 15.19, 15.20. Then
  every file of the slice, whole, against checklist T.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/hydrology/prms_soilzone_ag.py:1726-1737`,
    `:1834-1880`, `:1997-2196`;
    `pws_phoenix/hydrology/prms_canopy.py:539-569`;
    `pws_phoenix/hydrology/prms_soilzone.py:174-199`;
    `pws_phoenix/hydrology/prms_runoff.py:455-486`;
    `pws_phoenix/hydrology/prms_hydraulic_geometry.py:188-200`;
    `pws_phoenix/processes_concrete.py` (whole);
    `pws_phoenix/maps.py:65-141`; `pws_phoenix/flow_graph.py:265-347`,
    `:471-530`; `pws_phoenix/units.py:63-107` (by grep);
    `tests/prms_oracle.toml`; `pytest.ini`;
    `.github/workflows/ci.yaml`, `ci_ecosystem.yaml`;
    `tests/test_prms_snow_ab_numpy.py:76-99`;
    `docs/design_description.md:859-886`.
  - Before the review: `coverage_serial.txt` (whole); the heading of
    every finding in the 14 code-slice files; in full, every passage
    of those files that names slice 15, `tests/kernel_units.py`,
    `tests/test_kernel_units.py`, `tests/conftest.py` or
    `tests/run_tests.sh` (findings 1.13, 3.7, 6.6, 6.7, 8.2 and the
    notes for later slices of slices 1, 3, 6-13).
- Not read: the bodies of the other code findings (headings only).
  The question "why did no test catch this?" is answered here for
  the findings the harness bears on (section below); the rest belong
  to the test slice that covers each code slice (16-22).
- Not run: nothing was executed. Statements about pint (what it
  accepts, what it converts) come from its documented behaviour and
  from the must-fail tests in `tests/test_kernel_units.py`, which
  show the errors pint raises.

## Findings

### 15.1 With `func=`, the "compiled" run is the same Python function: seven njit kernels are never run compiled by their units test

- Where: `tests/kernel_units.py:173-180`;
  `tests/test_kernel_units.py:1185`, `:1475`, `:1499`, `:1595`,
  `:1626`, `:1641`, `:1678`
- Code:

      (func if func is not None else cls._calculate)(*arrays.values())

      out = assert_kernel_dimensionally_consistent(
          PRMSRunoffAg, data, func=PRMSRunoffAg._post_areas.py_func
      )

- Consequence: `func=` was written for a plain-numpy function (the
  module docstring: "an init-time derivation such as
  `muskingum_mann_coefficients`"; there "the plain-float run of the
  same function stands in for the compiled run"). Seven `@numba.njit`
  kernels are passed the same way, as `.py_func`:
  `PRMSRunoffAg._post_areas`, `PRMSSoilzoneAg._update_areas`, `_pre`,
  `_post`, `PRMSSoilzoneAgObsET._prep_aet`, `_adjust_irrigation`,
  `_iter_diag`. For these, `run_compiled` runs the Python body on
  floats. The bit-for-bit comparison is Python on floats against
  Python on Quantities, and the asserts that follow
  (`out["ag_area"][1] == 20.0`, `status[0] == -9`, ...) pin the
  Python body's values. A difference between numba and Python in
  these kernels is not seen (6.9 names one candidate form,
  `max(x, 0.0)` with nan). A fix: pass the Dispatcher and let
  `_body` take `.py_func` when the object has one.
- Source: /code-review
- Outcome: open

### 15.2 A kernel's return values are thrown away by both runs

- Where: `tests/kernel_units.py:179`, `:263`;
  `pws_phoenix/hydrology/prms_soilzone_ag.py:2026`, `:2079`,
  `:2149-2173`
- Code:

      (func if func is not None else cls._calculate)(*arrays.values())
      _body(cls, func)(*arrays.values())

      return add_estimated_irrigation, num_hrus_ag_iter, unsatisfied_big
      ...
      if not add_estimated_irrigation:
          keep_iterating = False

- Consequence: `_adjust_irrigation` returns the flag that decides
  whether the ObsET loop makes another pass, and `_prep_aet` returns
  the count that ends it (`if n_transp == 0`). Neither run keeps
  them, so they are not compared, and `unsatisfied_big` (inches,
  started as a bare `0.0`) is not unit-checked on the way out. The
  fixture has the capped HRU of 10.4 (h3: `unused_ag_et` 5.0 above
  `ag_soil_moist_max` 4.0) and pins
  `outs[1]["ag_irrigation_add"][3] == pytest.approx(0.1 + 4.0)`: the
  test runs the branch 10.4 is about and cannot see the flag 10.4 is
  about.
- Source: /code-review
- Outcome: open

### 15.3 A units test compares a kernel with itself, and `coverage_serial.txt` counts the lines only it runs as covered

- Where: `tests/kernel_units.py:305-319`;
  `coverage_serial.txt:2-3`, `:63-121`
- Code:

      compiled = run_compiled(cls, data, func)
      ...
      np.testing.assert_array_equal(got, compiled[name], err_msg=name)

      # Run 2: numba on, tests/test_kernel_units.py alone, appended to run 1

- Consequence: the comparison shows that the units are consistent.
  It says nothing about the values: both sides are the code under
  test. The asserts after it pin one to three outputs per fixture
  HRU against literals. The coverage table is the sum of both runs,
  so a line reached by run 2 alone reads as covered, and the table
  cannot say which lines those are. Known cases, from the plan and
  the code slices:
  - `if sink > stor:` (`pws_phoenix/hydrology/prms_groundwater.py:224`,
    `:308`): the plan's "known before the start" item;
  - the preferential-flow reservoir of both soil zones
    (`coverage_serial.txt:93-94`): drb_2yr and fgr_ag_2yr have
    `pref_flow_den == 0` on every HRU (notes of slices 9 and 10);
  - the Ag runoff kernel with `intcp_changeover_in_net_rain` False
    (8.2).
  The 88% total therefore overstates what is compared with
  pywatershed's answers. A table from run 1 alone would give the
  lines that only the units tests reach.
- Source: checklist T (the expected value comes from the code under
  test), coverage
- Outcome: open

### 15.4 The units test takes units from the kernel's argument NAMES; nothing compares the positional argument lists in `calculate()` with the signature

- Where: `tests/kernel_units.py:130-156`;
  `pws_phoenix/hydrology/prms_canopy.py:539-569`;
  `pws_phoenix/hydrology/prms_soilzone_ag.py:1726-1737`,
  `:2153-2165`
- Code:

      return list(inspect.signature(_body(cls, func)).parameters)

      self._calculate(
          refs["net_ppt"],
          refs["net_rain"],
          refs["net_snow"],
          ...

      def _hold(self, *names: str) -> tuple:
          return tuple(self._refs[nn] for nn in names)

- Consequence: the test builds each argument from the field of the
  same name and calls the kernel itself. The model calls the kernel
  from a list written by hand, in position (27 arrays for the
  canopy; "~80 arguments" for the Ag family, by its docstring). No
  test reads a signature except the harness (`inspect.signature`
  appears in `tests/` only in `tests/kernel_units.py`). Two float
  arrays swapped in a `calculate()` list compile, run, and pass every
  units test. A regression against pywatershed's answers catches the
  swap only where its domain runs the lines and the two arrays
  differ; 15.3 lists lines no regression runs. A check is cheap: for
  each kernel call built with `_hold` or a `refs[...]` list, compare
  the names with the signature.
- Source: checklist T
- Outcome: open

### 15.5 A line marked `# units-test: unreachable` is exempted without a check that it did not run

- Where: `tests/kernel_units.py:183-194`, `:270-284`
- Code:

      marked = {
          first + offset
          for offset, text in enumerate(source)
          if UNREACHABLE_MARKER in text
      }
      return lines - marked
      ...
      never = sorted(executable_lines(code) - lines)

- Consequence: a marker on a line that does run, or that an edit
  makes reachable, never fails: the line leaves the required set and
  nothing looks for it in `fired`. Seven lines carry the marker
  today (`pws_phoenix/hydrology/prms_runoff.py:470`, `:483`, `:895`,
  `:908`, `:965`; `pws_phoenix/hydrology/prms_soilzone.py:193`,
  `:194`). Slice 8 already doubts one reason: the comment at
  `prms_runoff.py:963-964` does not cover a negative or nan
  `dprst_depth_avg`, which reaches `:965`. The harness matches the
  prefix only, so five of the seven have no `(<reason>)` on the
  line. A fix: in `assert_all_lines_ran`, fail when a marked line is
  in `fired`.
- Source: /code-review
- Outcome: open

### 15.6 The harness requires every statement to run, not every branch (known before the start)

- Where: `tests/kernel_units.py:23-30`;
  `tests/test_kernel_units.py:851-852`
- Code:

      for review by eye. Statement coverage, not branch coverage: a skipped
      ``if`` body is required, the paths between statements are not.

      19 empty depressions: the area stays the bare-zero initial value
         (_ZERO_ACRES -- statement coverage alone would not need this hru)

- Consequence: for an `if` with no `else`, the path that skips the
  body need not run under pint. That path has no operation of its
  own, but it decides which value reaches the later lines: a
  variable that keeps its bare initial value on the skipped path is
  checked only if a fixture HRU takes that path. HRU 19 of the
  runoff fixture was added by hand for one such case; nothing finds
  the others.
- Source: plan ("known before the start"), checklist T
- Outcome: open

### 15.7 For a field declared `fraction` or `none` the Quantity run checks much less, and no tested kernel has an absolute temperature

- Where: `pws_phoenix/units.py:103-107`;
  `tests/kernel_units.py:105-110`
- Code:

      "fraction": "dimensionless",
      "none": "dimensionless",
      "percent": "percent",
      "degrees": "degree",
      "radians": "radian",

- Consequence:
  - pint accepts a bare number added to, compared with, or assigned
    into a dimensionless Quantity. The six tested kernel modules
    hold `units="fraction"` 62 times and `units="none"` 7 times
    (grep; a units key passed through a helper is not counted). For those
    fields the named-zero rule is not enforced by the test, and
    `fraction` declared where `none` is meant (or the reverse)
    cannot be seen. `radians` is dimensionless in pint as well. This
    is the test-side form of 1.3.
  - The question left by 1.13 (does the bit-for-bit comparison catch
    a product with an absolute temperature, now that
    `autoconvert_offset_to_baseunit=True` stops pint raising?): no
    tested module declares `degF` or `degC`, so no test meets the
    case. It opens with snow, atmosphere and stream temperature.
    There the comparison catches the kelvin product only if it
    reaches an output array; a product used only in a condition
    (`if slope * tmax > limit`) changes no magnitude unless the
    branch flips.
- Source: checklist T (a check that cannot fail)
- Outcome: open

### 15.8 No kernel units test for snow, atmosphere, the stream-temperature group, the FlowGraph node types, or hydraulic geometry (known before the start)

- Where: `tests/test_kernel_units.py:48-74` (the imports: groundwater,
  canopy, channel, soilzone, soilzone Ag, runoff)
- Code:

      from pws_phoenix.hydrology import (
          prms_canopy,
          prms_channel,
          prms_runoff,
          prms_soilzone,
          prms_soilzone_ag,
      )

- Consequence: the declared units of these kernels are checked by no
  test. `CLAUDE.md` ("Units mode") lists snow, atmosphere, stream
  temperature and the FlowGraph node types as "not yet"; hydraulic
  geometry is in neither its "done" nor its "not yet" list. Also
  without a test: the toy `Upper` / `Lower` kernels and the budget
  kernels (`pws_phoenix/budget.py:56-67`), which are simple sums.
  What each needs first, from the code slices:
  - STARFIT types: 6.6 (four declarations that do not match the
    values) and 6.7 (bare `0.0` for MCM and m^3/s);
  - hydraulic geometry: bare `0.0` into five dimensional outputs
    (`pws_phoenix/hydrology/prms_hydraulic_geometry.py:193-197`);
  - snow: bare `0.0` and `1.0` into dimensional outputs, bare albedo
    values (note of slice 11);
  - atmosphere: 12.2, `1/delta_degF` times an absolute `degF`, the
    Jensen-Haise line, the literals listed in the note of slice 12;
  - stream temperature: the marker for the unreachable branch of
    `_twavg` (13.6), bare sentinels (`-99.9`, `-98.0`), `_equilb`'s
    empirical constants;
  - any kernel with absolute temperatures: 15.7.
- Source: plan ("known before the start")
- Outcome: open

### 15.9 The composed FlowGraph class does not store its kernel, and the kernel's arguments are not fields (known before the start)

- Where: `pws_phoenix/flow_graph.py:474`, `:505-509`, `:511-530`;
  `tests/kernel_units.py:124-133`
- Code:

      kernel = _build_graph_kernel(prepare_fns, substep_fns, finalize_fns)
      ...
      kernel(self._graph_state, dt, np.int64(n_substeps), tctx)

      return func if func is not None else cls._calculate.py_func

- Consequence: `kernel` is a local of `make_flow_graph`, reached only
  through the closure of `calculate`; `class_attrs` gets no entry
  for it. A test cannot reach `kernel.py_func` from the class.
  Storing it is not enough for the harness as written: the kernel
  takes `(state, s_per_time, n_substeps, tctx)`, where `state` is a
  namedtuple of every field array, and `kernel_arg_names` assumes
  one argument per field. The harness needs a second way to build
  the arguments (a namedtuple of Quantity arrays) before a FlowGraph
  units test can exist.
- Source: plan ("known before the start"), checklist T
- Outcome: open

### 15.10 `check_oracle_pins` passes answer data that carries no marker file, and no test calls the check

- Where: `tests/conftest.py:98-141`
- Code:

      found = _marker(root / domain / ".oracle_ikey")
      if found is not None and found != ikey:
      ...
      okey = _marker(out_dir / ".oracle_okey")
      if okey is None:
          continue

- Consequence: the docstring excuses a pinned DIRECTORY that does
  not exist. The code also excuses a directory that exists with no
  `.oracle_okey`, and a domain with no `.oracle_ikey`: a copy of the
  data made without dotfiles (`cp -r test_data/*`), or data from
  before the markers existed, passes, and the parity tests run on
  answers nothing has identified. Only the directories named in
  `tests/prms_oracle.toml` are looked at; nothing compares that list
  with the directories the tests read. `check_oracle_pins` and
  `oracle_platform_tag` are named in no file under `tests/` but
  `conftest.py`: no test shows that a wrong key, or a
  `.derived_okey` that differs, stops the session.
- Source: /code-review; checklist T (a guard with no test showing
  that it raises)
- Outcome: open

### 15.11 On a platform with no pin, marked data stops the whole session, including the tests that read no answers

- Where: `tests/conftest.py:80-91`, `:117-119`, `:133-141`;
  `tests/prms_oracle.toml` (every output has `macos-arm64` only)
- Code:

      system = {"linux": "linux", "darwin": "macos", "win32": "windows"}[
          sys.platform
      ]
      ...
      if tag not in by_platform:
          problems.append(f"{dir_name}: no pin for platform {tag}")

- Consequence: on linux-x86_64 with data that carries markers,
  `pytest_sessionstart` raises `pytest.UsageError` and no test runs:
  not the toy regressions, not `tests/test_kernel_units.py`, which
  read no answers. With unmarked data the same platform runs
  everything unchecked (15.10). Another `sys.platform` (for example
  `cygwin`) raises a bare `KeyError` at `:84`.
  `docs/design_description.md:884` records "Keys are pinned for
  macos-arm64 only" as unfinished.
- Source: /code-review
- Outcome: open

### 15.12 The no-skips rule exists in `tests/run_tests.sh` alone; the fast CI job runs without it and without the data

- Where: `tests/run_tests.sh:59`, `:66`;
  `.github/workflows/ci.yaml:52-55`;
  `.github/workflows/ci_ecosystem.yaml:10-11`, `:40`, `:116`;
  `tests/test_prms_snow_ab_numpy.py:83-84`
- Code:

      serial_opts=(--error-for-skips --ignore-glob='tests/*_mpi.py')

      python -m pytest tests/ -vv -n=4 \
        --ignore-glob='tests/*_mpi.py' \

      #   test file under mpirun (tests/run_tests.sh). Data-dependent
      #   suites (drb/starfit) skip cleanly: no domain data in CI.
      ...
      ECOSYSTEM_ORG: EDIT-ME-org

      # a skipif mark, not pytest.importorskip: --error-for-skips does not see
      # a collection-time module skip, so importorskip would pass the sweep

- Consequence:
  - `ci.yaml` calls pytest directly, with no `--error-for-skips` and
    no test data (`pytest_sessionstart` returns at
    `tests/conftest.py:134-135`). The test files with a data
    `skipif` (about 50; `skipif` appears in 54 files under `tests/`,
    two of them the harness itself) skip, and the job is green. That is the case
    `docs/design_description.md:862-863` gives as the reason for the
    rule ("a machine without the answer data looked green").
  - `ci_ecosystem.yaml` does call `tests/run_tests.sh`, with a
    comment that the data suites "skip cleanly". Under
    `--error-for-skips` they fail. The workflow also still holds the
    placeholder `EDIT-ME-org`. `docs/design_description.md:885`
    records "CI cannot fetch the answer data yet".
  - A module-level skip is not seen by `--error-for-skips` (the
    comment above). No test file uses `importorskip`,
    `allow_module_level` or `xfail` today (grep), and nothing checks
    that none is added.
- Source: checklist T (a guard with no test showing that it raises)
- Outcome: open

### 15.13 An MPI test that hangs stops the sweep with no result: no launch has a time limit

- Where: `tests/run_tests.sh:42`, `:70`; `pytest.ini`;
  `environment.yaml` (no `pytest-timeout`)
- Code:

      MPI_CMD=${MPI_CMD:-"mpirun -n 4"}
      ...
      ${MPI_CMD} pytest "${mpi_opts[@]}" "${ff}" "${pytest_opts[@]}" \
          || failures=$((failures + 1))

- Consequence: this is the test side of 3.7. When one rank raises
  alone (an assertion before a collective call, or one of the
  lone-rank raises 3.7 lists), the other three wait in the
  collective. The loop never reaches the next file and the summary
  line is never printed. The note of slice 3 proposes
  `python -m mpi4py`; that acts on an exception nobody catches, and
  pytest catches a test's exception and goes on to the next test, so
  it would not end the job (reasoned, not run). A time limit on each
  launch (`timeout` before `${MPI_CMD}`, or `pytest-timeout`) turns
  a hang into one failed suite that names its file.
- Source: checklist T; the note for slice 15 in `03_model.md`
- Outcome: open

### 15.14 Every MPI file runs at one rank count

- Where: `tests/run_tests.sh:42`; `tests/conftest.py:297-299`
- Code:

      MPI_CMD=${MPI_CMD:-"mpirun -n 4"}

      # N_HRU divides evenly over 2 and 4 ranks (uneven decomposition is a
      # separate concern from the Step B cross-grid comm under test).
      N_HRU = 8

- Consequence: results that depend on the number of ranks are seen
  at 4 only: the last bits of a `MapMPI` product (2.12), the ObsET
  `iter_count` (10.5). No sweep runs 1 rank, 2 ranks, or more ranks
  than the distributed grid has elements (a rank with an empty
  block). `MPI_CMD` can be set by hand; nothing does so on a
  schedule.
- Source: checklist T
- Outcome: open

### 15.15 `run_tests.sh`: a pytest filter counts every unselected MPI file as a failed suite, and both `-only` flags together run nothing and print "ALL SUITES PASSED"

- Where: `tests/run_tests.sh:46-52`, `:58-73`, `:78-79`
- Code:

      --serial-only) run_mpi=0 ;;
      --parallel-only) run_serial=0 ;;
      *) pytest_opts+=("${arg}") ;;
      ...
      if [ "${failures}" -eq 0 ]; then
          echo "==== ALL SUITES PASSED (in ${elapsed})"

- Consequence:
  - `./tests/run_tests.sh -k groundwater`: pytest exits 5 for a file
    with no selected test, so each such MPI file adds one to
    `failures`. The run ends "N SUITE(S) FAILED" with nothing
    failed. The header documents extra arguments (`-vv -x`), so a
    filter is an expected use.
  - `--serial-only --parallel-only` sets both switches to 0. No
    pytest runs, `failures` stays 0, and the script reports success.
    The script never checks that a suite ran.
- Source: /code-review
- Outcome: open

### 15.16 The toy time axis stops at Dec 30: 365 steps of the leap year 2000

- Where: `tests/conftest.py:224-228`, `:254-256`, `:261-263`, `:324`
- Code:

      end_time = np.datetime64(f"{start_year + n_years}-01-01") - np.timedelta64(
          1, "D"
      )
      time = np.arange(start_time, end_time, dtype="datetime64[D]")

      np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)

- Consequence: `end_time` is already Dec 31 and `np.arange` leaves
  its stop out, so `n_time` is 365 and no toy regression (serial,
  MPI, two-grid, restart) steps Dec 31, day of year 366. Only
  `time.month` reaches a toy kernel, so no answer changes; the axis
  is not the year its code describes. The three sine and cosine
  axes use `np.arange` with a float step, whose length is not
  guaranteed to be `n_time` for every `n_time`: with another value
  (366 after a fix of the first point) the dataset constructor can
  raise on a size mismatch. `np.linspace(..., n_time,
  endpoint=False)` has a fixed length.
- Source: /code-review
- Outcome: open

### 15.17 Three of the four toy parameters are read by no kernel, so no toy answer depends on a static `(space,)` parameter

- Where: `tests/conftest.py:270-276`, `:416-463`;
  `pws_phoenix/processes_concrete.py:82-100`, `:160-180`
- Code:

      param_up_0=(["space"], rng.uniform(0.1, 1, n_space)),
      param_low_0=(["space"], rng.uniform(0.17, 0.23, n_space)),
      param_shared_name=(["space"], np.zeros(n_space)),

      flow[:] = flow_previous * np.float64(0.95) + forcing_up * param_up_1

- Consequence: the toy kernels read `param_up_1` (on `(month,
  space)`) and the forcings. `param_up_0`, `param_low_0` and
  `param_shared_name` are declared, supplied and checked for buffer
  identity, and their values reach no result;
  `param_shared_name` is all zeros. A static parameter that lands in
  the wrong rank's block, or with its dims in another order (3.1),
  changes no toy answer. The port tests under MPI do depend on
  static parameters; the toy tests, which are the ones written to
  test the framework, do not.
- Source: checklist T (a comparison of only some variables)
- Outcome: open

### 15.18 `match="ts"` matches every assertion message, so two must-fail cases do not show which array failed

- Where: `tests/test_kernel_units.py:462-466`, `:483-489`
- Code:

      # magnitude comparison sees it (ts lands on the wrong rung)
      ("_SECONDS_PER_MINUTE", AssertionError, "ts"),
      ("_MINUTES_PER_HOUR", AssertionError, "ts"),

- Consequence: `match` is a regular-expression search. numpy's
  message holds "Mismatched elements", and the coverage message
  holds "units-test"; both contain `ts`. If dropping one of the two
  patches stopped changing `ts` and the failure came from `c0`, or
  from `assert_all_lines_ran`, the case would still pass. The
  groundwater case (`match="gwres_flow_vol"`) and the ObsET case
  (`match="AET_external"`) do name their array. An anchored pattern
  (`r"\nts\n"`) would.
- Source: /code-review
- Outcome: open

### 15.19 The mypy pre-commit hook reports 34 errors at e33e217 (known before the start)

- Where: the hook configuration; `tests/` (32 errors),
  `pws_phoenix/` (2 errors); `.github/workflows/ci.yaml:43-44`
- Code:

      run: mypy --check-untyped-defs --ignore-missing-imports pws_phoenix/ tests/ examples/

- Consequence: recorded from the plan, not re-run here. In a hook
  environment built fresh on 2026-11-15 the hook fails in 10 files,
  so commits on this branch skip it. Not yet determined: real
  errors, or stubs of the hook's unpinned xarray and pytest. CI runs
  mypy in the `pwpx` environment, where the versions are those of
  `environment.yaml`; whether that job passes was not looked at.
- Source: plan ("known before the start")
- Outcome: open

### 15.20 Docstrings and comments that do not match the code

- Where: `tests/conftest.py:189-193`, `:243-246`;
  `tests/kernel_units.py:1-4`, `:311-313`;
  `.github/workflows/ci_ecosystem.yaml:10-11`
- Code:

      weights.T -- identical math and float order to a per-step
      Map apply; Map/MapMPI wiring is proven by the PRMSChannel

      on pint Quantity arrays built from the declarations and compare it,
      magnitude for magnitude, to the compiled kernel.

      if name in PER_DAY:
          continue  # carries the 1 day factor by construction

- Consequence:
  1. `node_vol_input` is a dense product (`hru_da.values @
     weights.T`). `Map.apply` sums each row's nonzeros in column
     order, and its docstring (`pws_phoenix/maps.py:78-80`) says "a
     dense BLAS product sums in its own blocked order, so the last
     bit can differ where a row has >= 3 entries". The float order
     is not identical; the FlowGraph tests compare at 1e-10, so no
     result changes.
  2. `make_toy_input`'s docstring lists `flow_initial` and
     `storage_initial` beside the parameters with no word that three
     of the parameters are unused (15.17).
  3. The harness docstring says the comparison is with "the compiled
     kernel"; with `func=` it is not (15.1).
  4. The comparison loop skips the global `PER_DAY`, where
     `run_quantified` multiplies by its `per_day` argument. With
     another set the two disagree; every caller that passes
     `per_day` calls `run_quantified` directly, so no test meets it.
  5. `ci_ecosystem.yaml`: "skip cleanly" (15.12).
- Source: checklist T
- Outcome: open

## Code findings and the harness: why did no test catch this?

Only the findings the harness bears on. The others are for the test
slice that covers their code slice.

- 1.1, 1.5: the tests of `units.reconcile` and of the budget
  declarations are `tests/test_units*.py` and `tests/test_budget.py`
  (slice 16); the harness has no part.
- 1.3: slice 16 for `units.same`; the kernel tests share the cause
  (15.7).
- 1.13: 15.7.
- 3.7: 15.13.
- 6.6, 6.7, and the notes of slices 7 (hydraulic geometry), 11, 12,
  13: no kernel units test (15.8).
- 7.3: the canopy fixture has `pk_ice_prev` 0.0 or 1.0 and
  `freeh2o_prev` 0.0 (`tests/test_kernel_units.py:215-216`). The
  kernel has no `pkwater_ante` argument, so no fixture can put the
  two tests on opposite sides.
- 8.2: the Ag kernel is run with the flag False. The asserts on the
  changeover HRU are `out["infil"][2] > 0.0` and
  `out["infil_ag"][2] > 0.0` (`:990`, `:1155`); `dprst_in` is
  asserted on HRU 1 only (`:1008`, `:1168`). The base and Ag results
  are never compared with each other.
- 8.6: `hru_frac_perv` is 0.85 or 0.4 (base), 0.65, 0.85 or 0.2
  (Ag); `ag_frac` is 0 only on the dry HRU 0 and the lake 11.
- 8.7: the lake (HRU 11) has one assert, `out["sroff"][11] == 0.0`
  (`:1005`, `:1166`). No test sums the stores and fluxes.
- 9.3: the swale (HRU 5) has `pref_flow_infil_frac` 0 (`:644`), and
  the soil zone fixture has no lake or inactive HRU.
- 9.4: `hru_frac_perv` is 0.8 on every HRU (`:652`).
- 9.5: `pref_flow_flag` is False on every HRU (`:659`).
- 10.2: HRU 30 has `pref_flow_max = 0` (`:1299`) and the four option
  runs share one dataset, so the units test does run it with the
  flag True. No assert names HRU 30. Lost water is consistent in
  units; a test of units cannot see it. A budget check on the units
  fixtures would.
- 10.3: the family kernel runs once per call, never in the loop.
- 10.4: 15.2.
- 10.6: `test_soilzone_ag_pre_post_units` (`:1492-1500`) runs `_post`
  on data with an inactive HRU (24) and a lake (25) and has no
  assert of its own; with 15.1 it compares Python with Python.
- 10.7: in the family kernel runs `aet_external` is 0.15 everywhere
  (`:1312`); the missing value -1.0 is only in `aet_observed`, for
  `_prep_aet`.

## Checklist T, item by item

- A raise or a branch that no test reaches: `tests/` is not measured
  (`--cov=pws_phoenix`), so by reading. `check_oracle_pins`: every
  branch that appends a problem, and `oracle_platform_tag` on any
  platform but the one the tests run on (15.10, 15.11).
  `tests/kernel_units.py`: every argument is used by some test
  (`declared`, `per_day`, `patches`, `func`, `fired`,
  `require_all_lines=False`).
- A test that cannot fail: 15.1, 15.5, 15.18; in part 15.3 and
  15.7. `test_soilzone_ag_pre_post_units` can fail only on a pint
  error or an unrun line.
- A tolerance looser than the process's standard: none. The harness
  compares bit for bit; `pytest.approx` (default 1e-6) is used on
  literals that say which HRU ran which branch.
- A comparison where nan on both sides passes:
  `np.testing.assert_array_equal` (`tests/kernel_units.py:319`)
  treats nan in the same place as equal. Right for comparing a
  kernel with itself; it means an output that is nan in both runs
  passes. HRU 10 of the soil zone fixture takes a nan path on
  purpose, and its output is asserted (`slow_flow[10] == 0.0`).
- A comparison of only some variables: every argument array is
  compared; return values are not (15.2); the toy answers leave
  three parameters out (15.17).
- A serial test with no MPI counterpart where the MPI code differs:
  does not apply to the four files. The kernels are the same
  functions under MPI.
- A guard with no test showing that it raises: 15.10 (the pins),
  15.12 (no skips). `os.environ["PWS_NUMBA_CACHE"] = "0"`
  (`tests/conftest.py:34`) "must precede the first pws_phoenix
  import"; no test asserts `pws_phoenix.NUMBA_CACHE is False`. Every
  entry of every patch table has a must-fail case (counted: canopy
  4, channel 3, muskingum 7, soilzone 6, runoff 12, runoff Ag 4, the
  soilzone Ag tables 10), two of them weak (15.18).

## Notes for later slices

- Slice 16: 15.16 and 15.17 (what the toy answers cannot show: Dec
  31, a static parameter). Why no test caught 1.1, 1.3, 1.5. Whether
  any test asserts `pws_phoenix.NUMBA_CACHE is False`.
- Slices 16-22, MPI files: every one runs at 4 ranks (15.14) and
  none has a time limit (15.13). For each MPI test, ask what one
  rank failing alone would do.
- Slice 18: 15.9; `tests/conftest.py:188-204` pre-aggregates the
  lateral inflows with a dense product (15.20 item 1), so those
  tests do not run a Map.
- Slices 19, 20: for each kernel, which lines the regression domain
  runs, as opposed to the units test (15.3). The tables of
  `coverage_serial.txt` cannot tell; the notes of slices 7-10 name
  the lines by reasoning about the domains.
- Slice 22: `tests/prms_oracle.toml` pins "only the answer sets the
  tests read"; check the list against the directories the translate
  and port tests open (15.10).
- Fix phase, runs that confirm before fixing: pass the Dispatcher in
  the seven `func=` tests and see whether they still pass (15.1);
  coverage of run 1 alone, compared with the combined table (15.3);
  a test that puts a marker on a line that runs (15.5);
  `PWS_TEST_DATA` pointing at a copy without dotfiles (15.10);
  `./tests/run_tests.sh --serial-only --parallel-only` and
  `./tests/run_tests.sh --parallel-only -k groundwater` (15.15);
  `len(dimensions["time"])` (15.16); an MPI test with
  `assert rank != 1` ahead of a collective call, under `timeout`
  (15.13).
