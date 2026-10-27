# pws_phoenix performance benchmarks (ASV)

Design document for the benchmark suite in this directory. Written
before the code (Oct 2026); facts about ASV below were checked
against the installed `asv 0.6.6` + `asv_runner 0.3.2` source, with
`file:line` pointers into the `pwpx2` site-packages.

- [1. Strategy](#1-strategy)
- [2. Layout in the repo](#2-layout-in-the-repo)
- [3. ASV primer](#3-asv-primer)
- [4. Model under test and coverage](#4-model-under-test-and-coverage)
- [5. Benchmark inventory](#5-benchmark-inventory)
- [6. asv.conf.json](#6-asvconfjson)
- [7. Running and reading results](#7-running-and-reading-results)
- [8. Later](#8-later)

## 1. Strategy

**Goal.** A fine-grained, repeatable view of where pws_phoenix spends
time and memory, so a port or refactor that slows a kernel or grows
the footprint is caught. The primary unit is the **process**: one
timing per process per lifecycle method (`__init__`, `initialize`,
`advance`, `calculate`). Whole-model numbers sit beside them so the
per-process numbers can be checked against the total. Serial only to
start.

**Why ASV** (airspeed velocity) rather than pytest-benchmark:
`peakmem_*` benchmarks measure what the memory prime directive
(`CLAUDE.md`) demands; results are stored per commit and compared
with `asv compare`; `asv publish` renders a browsable history.

**Environment: the existing conda env, not an ASV-managed one.**
Benchmarks run in `pwpx`/`pwpx2` (locally) or the CI env -- the same
env the tests run in -- through ASV's "existing environment" mode
(`environment_type: "existing"`, section 3). ASV then neither
builds an env nor installs the project; it benchmarks the working
tree with the current `python`. Because pws_phoenix is an editable
install (`pip install -e .`), checking out another commit changes
the benchmarked code with no reinstall. The alternative -- an
ASV-managed conda env built from `environment.yaml`, which would
give `asv continuous` and commit ranges natively -- costs a second
copy of the heavy pinned env per machine, re-solved by ASV's conda
plugin rather than our setup; not worth it until the wrapper below
grows teeth (xarray's `asv_bench/` follows the same existing-env
pattern).

**Two uses of the suite:**

1. *PR comparison* (the minimum, local now; CI once the test data
   are fetchable there): run base and PR head in the same env in one
   sitting, then `asv compare base head`. Dependencies cannot explain
   a difference -- both sides share the env.
2. *History*: results saved per commit, on a developer machine,
   under `.asv/results/` (git-ignored for now, so they persist
   locally but are not committed). Across a long history the env's
   dependency versions ARE a confound, and existing-environment mode
   records none of them (`asv/environment.py:1062`:
   `self._requirements = {}`). Mitigation: the wrapper writes
   `conda list --export` to `.asv/env_lists/<hash>.txt` every time
   it saves a hash, so a drift can be checked against what changed
   in the env.

**The wrapper, `benchmarks/asv_run.sh`**: the hand-rolled `asv
continuous` for a fixed env -- requires a clean working tree; for
each hash given: `git checkout <hash>`, `asv run --set-commit-hash
<hash>`, record the env listing; restores the original checkout at
the end (also on error); with two or more hashes, `asv compare` of
the first and last. ASV's own `continuous`/range specs are refused in
existing-environment mode (`asv/commands/run.py:375-379`); this
script is their replacement.

## 2. Layout in the repo

```
pws_phoenix2/
├── asv.conf.json          ASV configuration (tracked)
├── benchmarks/            the suite (tracked)
│   ├── README.md          this document
│   ├── __init__.py
│   ├── _common.py         data root, model builder, shared constants (no benchmarks)
│   ├── bench_process.py   per-process lifecycle timings (the primary set)
│   ├── bench_model.py     whole-model time + peak memory
│   ├── bench_output.py    the serial zarr writer alone
│   └── asv_run.sh         the fixed-env history wrapper (section 1)
├── tests/
│   └── test_benchmark_coverage.py   every Process subclass / node type is benchmarked
└── .asv/                  ASV's working directory (git-ignored)
    ├── results/
    │   ├── benchmarks.json          the discovered benchmark list + attributes
    │   └── <machine>/<hash>-existing-<python>.json   one small JSON per saved run
    ├── html/              `asv publish` output (static site)
    └── env/               ASV-managed envs -- unused in existing-env mode
```

`benchmarks/` is ASV's default directory name (`benchmark_dir`); a
root directory named `asv/` would shadow `import asv` for the runner
itself. `.asv/` is not a separate repository -- it is ASV's scratch
and results space inside this one. Committing `.asv/results/` (the
history) is a later decision (section 8).

## 3. ASV primer

### What a benchmark is

A function, or a method of a class, in a module under `benchmarks/`.
The name prefix selects the measurement
(`asv_runner/benchmarks/time.py:75`, `peakmem.py:22`):

| prefix          | measures                                   | unit    |
| --------------- | ------------------------------------------ | ------- |
| `time_*`        | wall clock of one call                     | seconds |
| `peakmem_*`     | peak resident memory of the whole process  | bytes   |
| `mem_*`         | size of the returned object                | bytes   |
| `track_*`       | whatever number the function returns       | custom  |

A class attribute `params = [...]` + `param_names = [...]` runs every
method once per parameter value (here: once per process name).

### How a timing benchmark runs

Each (benchmark, parameter value) runs in a **fresh subprocess**
(`asv/runner.py:423-428`, launch method `spawn`). Inside it:

```
import the benchmark module
setup(*params)          # optional hook, NOT timed
<timed method>          # time_*: called repeatedly, see below
teardown(*params)       # optional hook, NOT timed
```

A `setup` hook is the counterpart of a pytest fixture: it builds the
state the measurement needs; the clock runs only over the `time_*`
method. That is what we want -- the number of interest is one
`calculate` call, not "build the model, then call".

**Samples and `number`** (Python `timeit` convention, which ASV
follows): one *sample* is the wall time of `number` back-to-back
calls of the method; the reported per-call time is sample ÷
`number`. ASV takes `repeat` samples and reports their **median**.
With `number = 0` (the default) ASV picks `number` itself so a
sample lasts at least `sample_time`.

### The rule that shapes this suite: `setup` re-runs before EVERY timed call

In `asv_runner 0.3.2`, when a `setup` hook exists, every timed call
sees freshly-set-up state (`asv_runner/benchmarks/time.py:80-81`,
`:213-233`): the automatic `number` becomes 1 (one call per sample)
and `setup` runs before each sample; an explicit `number > 1`
switches to a timer that re-runs `setup` between the
individually-timed calls (`time.py:11-40`). A `setup` that builds the
whole model (seconds) would therefore run tens of times per
benchmark.

Consequence for the design: **expensive state is NOT built in
`setup`.** The model is built the first time any benchmark in the
subprocess asks for it (`_common.get_model()`, section 5) and kept in
a module variable; later asks return that object -- a module-level
fixture, built lazily. The timed method calls `get_model()`; the
first call (model build + numba compile) lands in the warm-up period
and is discarded (`time.py:343-360`: with an explicit `number` and
`warmup_time`, calls are repeated untimed until `warmup_time` has
elapsed, THEN samples are collected). `setup` hooks are used only
where they are cheap (e.g. `tempfile.mkdtemp()` for the output
store).

`setup_cache` (run once, result pickled to disk and passed to the
benchmarks, `asv_runner/setup_cache.py:8-43`) is NOT used: a built
`Model` holds open netCDF handles (`Input` opens with
`xr.open_dataarray`, `data_io.py:56`) and numba dispatchers, and the
unpickling subprocess would have to recompile the kernels anyway.

### Timing attributes (class attributes on the benchmark; defaults in `time.py:112-116`)

| attribute      | default                   | meaning                                            |
| -------------- | ------------------------- | -------------------------------------------------- |
| `number`       | 0 = auto                  | calls per sample (per-call time = sample / number) |
| `repeat`       | 0 = auto (1..10, /rounds) | samples per round                                  |
| `rounds`       | 2                         | separate subprocesses, samples pooled              |
| `sample_time`  | 0.01 s                    | auto-`number` targets at least this per sample     |
| `warmup_time`  | 0.1 s                     | untimed calls before sampling                      |
| `timeout`      | 60 s                      | kill the benchmark subprocess after this           |
| `min_run_count`| 2                         |                                                    |

Every benchmark here sets `number`, `repeat`, `warmup_time` and
`timeout` explicitly (section 5): the auto-calibration loop would
otherwise see a multi-second first call and settle on `number = 1`,
and the 60 s default timeout is too short for a build + warm-up.

### Peak memory

`peakmem_*` runs the function once, then reads the subprocess's
maximum resident set size (`peakmem.py:67-70`, `_maxrss.py:117`:
`getrusage(RUSAGE_SELF).ru_maxrss`, bytes on macOS). It is the whole
process -- interpreter + numpy/xarray/numba imports + the measured
work -- so a `peakmem_imports` baseline (imports only) is included
and the build cost is the difference.

### Existing environment, commits, and saved results

`environment_type: "existing"` in the config = `asv run
--python=same` (`asv/environment.py:350-353`, `:1038-1040`): the
CURRENT CHECKOUT with the current `python`; nothing is installed or
built (`environment.py:1086-1096`). Consequences
(`asv/commands/run.py:279-281`, `:375-379`, `:544-547`):

1. No commit range can be given; the run is the working tree.
2. Without `--set-commit-hash <hash>` the commit is unknown and
   **results are NOT saved**. With it, the hash is recorded and the
   git repo is consulted for the commit date (so `asv publish`
   orders the history correctly).
3. `asv continuous` and `--interleave-rounds` are refused
   (`run.py:288-293`). `asv compare <hash1> <hash2>` works on saved
   results.

`asv machine --yes` must run once per machine first (records CPU/OS
into `~/.asv-machine.json`; otherwise `asv run` prompts).

## 4. Model under test and coverage

- **Data root**: the same rule as the tests -- `$PWS_TEST_DATA`, else
  `<mpix_meta>/pywatershed/test_data` (`tests/conftest.py:40-42`).
  `benchmarks/_common.py` repeats those two lines rather than
  importing `tests/conftest.py` (pytest machinery, and its oracle-pin
  check runs at session start).
- **First-cut domain / control**: `drb_2yr/nhm_stream_temp.control`
  via `pws_phoenix.prms_translate.model_from_control` -- the complete
  NHM through stream temperature: 765 HRUs, 456 segments, 9
  processes, 13 Maps. One build gives every one of those processes
  realistic state.
- **Processes in it** (`process_dict` keys, NHM schedule order,
  `prms_translate/control.py:231-260`): `prms_atmosphere`,
  `prms_canopy`, `prms_snow`, `prms_runoff`, `prms_soilzone`,
  `prms_groundwater`, `prms_channel`, `prms_hydraulic_geometry`,
  `prms_stream_temp`.
- **dt** = `np.float64(86400.0)` (seconds; the channel's
  `s_per_time`).
- **numba warm-up rule**: the first `calculate` of each process
  compiles its kernels. `get_model()` runs `model.run(DT,
  np.int32(1))` right after building, so every process is compiled
  and every input/Map buffer is filled before anything is timed. The
  warm-up period (section 3) absorbs it.
- **State drift caveat**: a per-process benchmark calls one method
  many times with no `advance` between (and `time_initialize`
  recomputes already-derived parameters). The wall-clock is right;
  the values are not a physics run. `time_chain_step` is the
  realistic-sequence number.

**Coverage rule: every `Process` subclass and every FlowGraph node
type gets benchmarked.** nhm_stream_temp covers nine classes; the
rest need other controls or a graph. A class already covered by one
model is not repeated in another. Enforced by
`tests/test_benchmark_coverage.py` (a test, not a doc promise -- the
no-skips policy's analogue): every class in `Process._registry` and
every node-type module in `hydrology/` must be named by some
benchmark. Ledger (first cut = row 1; later rows are the plan):

| model / control (`_common.CONTROLS` label)                        | classes it adds                                                                  |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| drb `nhm_stream_temp.control` (`drb_stream_temp`)                 | the nine above (PRMSAtmosphere, PRMSHydraulicGeometryWidthOnly, PRMSStreamTemp leaves) |
| drb `nhm_no_dprst.control` (`drb_no_dprst`)                       | PRMSGroundwaterNoDprst, PRMSRunoffNoDprst, PRMSSoilzoneNoDprst                   |
| fgr_ag_2yr `spinup.control` (`fgr_spinup`)                        | PRMSRunoffAg, PRMSSoilzoneAg, PRMSAtmosphereTranspFrost                          |
| fgr_ag_2yr `analysis.control` (`fgr_analysis`)                    | PRMSSoilzoneAgObsET, PRMSAtmosphereTranspFrostDyn (dyn_springfrost_flag)         |
| hand-built drb segment grid (`_common.HAND_BUILT`: `drb_seg_hg_full` = HydraulicGeometryFull -> ConstantShade with live `seg_flow_width`; `drb_seg_humid`), fed the `output_stream_temp/` answers as disk forcings -- `model_from_control` cannot build them (`resolve_classes` never picks HydraulicGeometryFull, no control sets `stream_temp_shade_flag=1`, `assemble_from_control` refuses the SegHumidity leaf) | PRMSHydraulicGeometryFull, PRMSStreamTempConstantShade, PRMSStreamTempSegHumidity |
| FlowGraphs (`_flow_graph.GRAPHS`): `drb_graph_channel` = the drb network as a pure channel graph; `ucb_graph_all_types` = the ucb network (1942 segments) with the mixed test's Big Sandy chain + a daily-STARFIT chain + a STARFIT-source-sink chain + a neutral source_sink splice + an obsin headwater (11 inserted nodes, one 24-substep cfs graph) | all seven node types: PRMSChannelFlowNode, PassThroughFlowNode, StarfitFlowNode, StarfitDailyFlowNode, StarfitSourceSinkFlowNode, ObsInFlowNode, SourceSinkFlowNode |

Rows 2-5 are `bench_process.ProcessVariants` (Oct 27, 2026): the
same four timings as `ProcessLifecycle`, `params = _common.VARIANT_ROWS`
= `"label/process"` strings naming ONLY the slots whose class row 1
lacks (e.g. `drb_no_dprst/prms_runoff`; the no_dprst control's other
six slots are the row-1 classes and are not repeated).
`test_variant_rows_add_new_classes_only` enforces the no-repeat rule.
Row 6 is `bench_flow_graph.FlowGraphLifecycle` (same four timings on
the `flow_graph` process; `_flow_graph.get_graph_model(label)`); the
all-types graph is a TIMING composition, not a validated scenario (Big
Sandy's parameters on every reservoir row). Both ledgers are now
empty: `PROCESSES_NOT_YET_COVERED` and `NODE_TYPES_NOT_YET_COVERED`
in the test are `set()`.
drb `nhm_transp_frost.control` and ucb `nhm_transp_frost.control` add
nothing the fgr controls do not. `get_model(label)` keeps one warmed
model per control in the subprocess; `bench_model.py` uses only
`drb_stream_temp`.

## 5. Benchmark inventory

Common attributes unless noted: `number = 10`, `repeat = 10`,
`rounds = 1`, `warmup_time = 2.0` (long enough to hold the build +
compile; tune after the first run), `timeout = 900`.

### `benchmarks/_common.py` (no benchmarks)

```python
PWS_TEST_DATA = pl.Path(os.environ.get("PWS_TEST_DATA", MPIX_ROOT / "pywatershed" / "test_data"))
DRB = PWS_TEST_DATA / "drb_2yr"
FGR = PWS_TEST_DATA / "fgr_ag_2yr"
CONTROL = DRB / "nhm_stream_temp.control"
CONTROLS = {"drb_stream_temp": CONTROL, "drb_no_dprst": ..., "fgr_spinup": ..., "fgr_analysis": ...}
DT = np.float64(86400.0)
PROCESS_NAMES = [...]            # the 9 keys above
VARIANT_ROWS = ["drb_no_dprst/prms_runoff", ...]   # section 4 ledger rows 2-4

def build_model(label="drb_stream_temp", control=None) -> Model:
    return model_from_control(CONTROLS[label], control)

def get_model(label="drb_stream_temp") -> Model:   # built + warmed once per subprocess per control
    if label not in _MODELS:
        model = build_model(label)
        model.run(DT, np.int32(1))
        _unfreeze_parameters_internal(model)   # see time_initialize
        _MODELS[label] = model
    return _MODELS[label]
```

### `benchmarks/bench_process.py` -- per-process lifecycle costs (the primary set)

`ProcessLifecycle`: `params = PROCESS_NAMES`, `param_names =
["process"]`. Each method fetches `model = get_model()` and `proc =
model.model_dict[name]`, then performs exactly one call
(`ProcessVariants` repeats the four for `VARIANT_ROWS` on
`get_model(label)` -- section 4):

| benchmark          | the call                                        | what it is                                                                                                                                                                    |
| ------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time_init`        | `type(proc)(proc._obj)`                         | construction = binding to the grid dataset (`model.py:440`; `Process.__init__` is `self._obj = xarray_obj`, `process.py:222-223`)                                            |
| `time_initialize`  | `proc.initialize()`                             | the init hook: `parameter_internal` derivation (Muskingum coefficients, soilzone/runoff derived params); a no-op for processes without one (e.g. groundwater) -- still recorded |
| `time_advance`     | `proc.advance()`                                | the `*_previous` copies                                                                                                                                                       |
| `time_calculate`   | `proc.calculate(DT, model.time)`                | one step of physics incl. the `.values` extraction around the njit kernel                                                                                                     |

`time_initialize` needs the derived parameters writable again: the
Model freezes them after all hooks (`model.py:451-455`:
`grid_ds[name].values.flags.writeable = False`), and a frozen array
raises `ValueError: assignment destination is read-only`.
`get_model()` therefore sets `writeable = True` back on every
process's `get_parameters_internal()` arrays once, after the warm-up
step (benchmark-only; no kernel reads the flag).

Build cost of a process = `time_init + time_initialize`; per-step
cost = `time_advance + time_calculate`. Dataset assembly
(`Model._add_process_fields`) and Map resolution are Model-side and
appear only in the whole-model numbers below.

### `benchmarks/bench_model.py` -- whole-model numbers

| benchmark             | the call                                           | notes                                                                                                                                      |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `time_chain_step`     | `get_model().run(DT, np.int32(1))`                 | one full step: all `advance` + all `calculate` + 13 Map applies + `Input` stepping; `number = 10` so a sample is a 10-day sequence         |
| `time_model_build`    | `build_model()`                                    | parameter loading, per-grid dataset assembly, binding, init hooks, Map resolution; `number = 1`, `repeat = 5`, `warmup_time = 0`         |
| `peakmem_model_build` | `build_model()`                                    | peak RSS of build (read against `peakmem_imports`)                                                                                         |
| `peakmem_imports`     | `import pws_phoenix.prms_translate` (+ numpy etc.) | the baseline the other `peakmem_*` are read against                                                                                        |
| `peakmem_chain_warm`  | `get_model()`                                      | peak RSS of build + 1 step: the numba compile of every kernel is in, no sequence yet                                                       |
| `peakmem_chain_run`   | `get_model().run(DT, np.int32(30))`                | peak RSS of build + 31 steps -- growth over `peakmem_chain_warm` is per-step allocation, but RSS noise is ~50 MB run to run (first runs: build 558/505 MB) |
| `track_step_alloc_bytes` | 5 more warm steps, then `tracemalloc` around `get_model().run(DT, np.int32(30))` | net bytes retained over 30 steps, byte-exact (numpy reports to tracemalloc) -- THE prime-directive number; expected ~0. First run (Oct 2026): 7.9 MB when measured from step 2 = late numba compiles of branches first reached after step 1; a few kB from step 6 |

`time_chain_step` minus the sum of the nine `time_advance +
time_calculate` = the framework overhead (Maps, Inputs, loop).

Step budget: the shared model cannot be rewound and drb_2yr has 731
steps, so `time_chain_step` must consume fewer than that over its
warm-up iterations + samples (`number = 10`, `warmup_time = 2 s` at
~9 ms/step ≈ 340 steps). `Model.run` past the end of the time axis
raises.

### `benchmarks/bench_flow_graph.py` -- FlowGraph lifecycle

`FlowGraphLifecycle`, `params = list(_flow_graph.GRAPHS)` (section 4,
row 6): `time_init` / `time_initialize` / `time_advance` /
`time_calculate` on `get_graph_model(label).model_dict["flow_graph"]`.
`drb_graph_channel/time_calculate` against
`ProcessLifecycle.time_calculate(prms_channel)` = the scalar-node
dispatch cost of the graph kernel over the array muskingum (same 456
segments, same inflows -- pre-aggregated hru volumes, no Maps).
`ucb_graph_all_types` is the `literal_unroll` dispatch at its widest
(seven types). Graph building lives in `benchmarks/_flow_graph.py`
(the tests' splice arithmetic, minus pytest).

### `benchmarks/bench_output.py` -- the serial zarr writer alone

| benchmark          | the call                                                                                                                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time_output_zarr` | `Output(time_chunk_size=30, variable_names=NAMES, output_store=tmp/"out.zarr", time_values=model.times.values)`; `initialize_variable_tracking(model.model_dict)`; `collect_current_timestep(tt)` × 30; `finalize()` |

`NAMES` = the control file's requested output variables (103 hru +
31 segment names), filtered to what the model has
(`PrmsRunConfig.output_var_names` through `assemble_from_control`,
`assemble.py:313-327`; `_common.get_output_var_names()`) -- the
realistic request. Model state is NOT advanced between collects (IO cost in
isolation: 30 buffer copies + one full-chunk append that creates the
store + a finalize with an empty tail). This benchmark uses a cheap
`setup`/`teardown` pair (`tempfile.mkdtemp()` / `shutil.rmtree`) --
re-run per call per the rule in section 3, which is exactly right
here: every call needs a fresh store. `number = 1`, `repeat = 10`.

The MPI output path (mpixarray's streamed NetCDF writer) is NOT
timed: the whole MPI path is outside the first cut (below).

### Not in the first cut

(The coverage ledger of section 4 is complete as of Oct 27, 2026.)

- `time_kernel_<process>`: the bare `_calculate` njit call, below the
  `.values` extraction -- kernel vs Python/xarray overhead per
  process. Needs per-process argument plumbing; add once the numbers
  above show the overhead matters.
- Individual Maps (`time_map_apply` per Map, incl. the MapMPI
  Allreduce under MPI) and Inputs (`Input.advance` / `set_index`).
- MPI: ASV runs its subprocess under plain `python`, not `mpirun`. An
  MPI number would be a `subprocess.run(["mpirun", ...])` wall-clock
  wrapper around a short `ModelMPI` run, or stays with the mpixarray
  scaling study.

### First numbers (Oct 26, 2026; LaNueva, Apple M4 Pro, env `pwpx2`)

For orientation only -- the record is `.asv/results/`. drb_2yr, 765
HRUs / 456 segments, medians:

| what                                   | value                                                      |
| -------------------------------------- | ---------------------------------------------------------- |
| `time_calculate` per process           | atmosphere 1.08 ms, canopy 0.59, snow 1.59, runoff 1.62, soilzone 1.61, groundwater 0.29, channel 0.19, hydraulic_geometry 0.09, stream_temp 0.80 (sum ≈ 7.9 ms) |
| `time_advance` per process             | 14-170 µs for processes with state; ~85 ns for the stateless three |
| `time_init` / `time_initialize`        | ~140 ns everywhere / 54 µs - 2.6 ms, stream_temp 21 ms (topology resolution) |
| `time_chain_step`                      | 9.57 ms -> ~1.2 ms/step of framework (13 Maps, Inputs, loop) |
| `time_model_build`                     | 420 ms (no kernel runs at build; the ~4 s first step is the numba compile) |
| `peakmem_imports` / `_model_build` / `_chain_warm` / `_chain_run` | 333 / 505-558 / 658 / 666-724 MB (±50 MB run to run) |
| `track_step_alloc_bytes`               | 5.6 kB over 30 steps (≈190 B/step of xarray/pandas bookkeeping; one hru array is 6 kB) |
| `time_output_zarr`                     | 204 ms for 134 variables x 30 steps                        |

## 6. asv.conf.json

```json
{
    "version": 1,
    "project": "pws_phoenix",
    "project_url": "https://github.com/EC-USGS/pws_phoenix",
    "repo": ".",
    "branches": ["main"],
    "dvcs": "git",
    "environment_type": "existing",
    "benchmark_dir": "benchmarks",
    "env_dir": ".asv/env",
    "results_dir": ".asv/results",
    "html_dir": ".asv/html"
}
```

- `environment_type: "existing"` = `--python=same` on every run
  (section 3); no `pythons`/`matrix`, no `build_command`/
  `install_command` -- the existing environment ignores them.
- `.asv/` is git-ignored (section 2).
- `project_url` is a placeholder for the HTML pages; fix when known.

## 7. Running and reading results

First time on a machine:

```
conda activate pwpx2
asv machine --yes
```

Smoke test (each benchmark called once, nothing saved):

```
cd /Users/jamesmcc/usgs/mpix_meta/pws_phoenix2
asv run --quick --show-stderr
```

A real run, saved under the current commit (`git rev-parse HEAD` is
yours to run):

```
cd /Users/jamesmcc/usgs/mpix_meta/pws_phoenix2
asv run --show-stderr --set-commit-hash <hash>
```

A PR comparison or a history, in the fixed env (the wrapper of
section 1; checks out each hash in turn and restores the original
checkout):

```
cd /Users/jamesmcc/usgs/mpix_meta/pws_phoenix2
benchmarks/asv_run.sh <base-hash> <head-hash>
asv compare <base-hash> <head-hash> --factor 1.1
```

Subsets: `-b <regex>` on the benchmark name, e.g.
`asv run --quick -b time_calculate`.

Reading: `asv show <hash>` prints one run's table; `asv compare`
the ratio table (`--factor 1.1` flags ±10 %); `asv publish` then
`asv preview` renders the saved history in a browser.

`--show-stderr` matters here: the numba experimental-feature
warnings from the FlowGraph compile and a `setup`-re-run notice from
asv (`time.py:226-230`) go to stderr and are otherwise hidden.

## 8. Later

- **CI on PRs**: a job in the existing CI env (`ci.yaml` already
  builds it from `environment.yaml`) running the wrapper on base and
  head and printing `asv compare` into the job log (optionally a PR
  comment). Blocked on the drb_2yr data being fetchable in CI (the
  oracle CI half). The runner is ephemeral, so CI keeps no history
  -- it is only the compare.
- **Committing the history**: once the first numbers are reviewed,
  commit `.asv/results/` (small JSON, one file per commit per
  machine) from the designated benchmark machine(s). Then publish:
  there is no `asv gh-pages` command in 0.6.6, so the docs deploy
  would run `asv publish` and push `.asv/html` to gh-pages beside
  the Sphinx docs (`peaceiris/actions-gh-pages`, `destination_dir:
  benchmarks`). Nothing is deployed to gh-pages today, so this waits
  for the docs deploy.
- The coverage ledger's remaining rows; kernel-level, Map, Input and
  MPI benchmarks (section 5, "Not in the first cut").
- Results are tagged by machine, not env: keep ONE benchmarking env
  per machine so the saved history is comparable.
