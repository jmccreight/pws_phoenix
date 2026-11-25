<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Developer notes](#developer-notes)
  - [Per-step data access: hold the ndarray, not the Dataset lookup](#per-step-data-access-hold-the-ndarray-not-the-dataset-lookup)
    - [What a process step cost before](#what-a-process-step-cost-before)
    - [The decision](#the-decision)
    - [The invariant it relies on](#the-invariant-it-relies-on)
    - [Rules this imposes](#rules-this-imposes)
    - [Outcome (Oct 27, 2026)](#outcome-oct-27-2026)
  - [Numba cache: on for users, off for tests, opt-in per kernel](#numba-cache-on-for-users-off-for-tests-opt-in-per-kernel)
    - [What invalidates the cache (numba 0.64, `numba/core/caching.py`)](#what-invalidates-the-cache-numba-064-numbacorecachingpy)
    - [MPI: no barrier](#mpi-no-barrier)
    - [Measured](#measured)
  - [Numba threading: `prange` over the HRU loop, 1 thread unless asked](#numba-threading-prange-over-the-hru-loop-1-thread-unless-asked)
    - [Threads vs ranks](#threads-vs-ranks)
    - [Where it pays (estimate; measurement pending)](#where-it-pays-estimate-measurement-pending)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Developer notes

Design decisions with their evidence, and the invariants they impose on
anyone writing a `Process`. (How we work -- tests, envs, review -- is
README "Installation" and `CLAUDE.md`; the pywatershed port record is
`docs/PORTS.md`.)

## Per-step data access: hold the ndarray, not the Dataset lookup

**Status (Oct 27, 2026): measured, decided, IMPLEMENTED** --
`Process.__init__` captures `self._refs: dict[str, np.ndarray]` (one
entry per declared field, `_resolved_fields`), and every
`advance()`/`calculate()` and init hook reads `refs[name]`; no
`.values` inside a step anywhere in the package. The MPI rebind does
not recapture (rule 2 below). `Model._validate_inputs_resolved` now
runs BEFORE binding in both paths -- the capture is strict, so an
unresolved input would otherwise surface as a bare `KeyError` at bind
instead of the assembly-time `ValueError`. Full serial + MPI sweep
green on the first run after the sweep. Before/after numbers: see
`benchmarks/README.md`.

### What a process step cost before

A process reaches every field it uses through its grid dataset, once
per step:

```python
# hydrology/prms_channel.py calculate()
self._calculate(
    self._obj["seg_lateral_inflow"].values,
    self._obj["seg_upstream_inflow"].values,
    ...                                   # 20 such lookups
    dt,
)
```

`self._obj[name]` is `xr.Dataset.__getitem__`: it constructs a new
`DataArray` (coords, indexes, attrs) every call, and `.values` then
returns the ndarray inside. The ASV benchmarks
(`benchmarks/bench_process.py`, `ProcessKernel.time_kernel` vs
`ProcessLifecycle.time_calculate`, drb_2yr, LaNueva) put the two
halves side by side:

| process (drb)         | calculate | kernel  | lookups | per lookup |
| --------------------- | --------- | ------- | ------- | ---------- |
| canopy                | 595 us    | 3.0 us  | 27      | 22 us      |
| snow                  | 1640 us   | 6.6 us  | 74      | 22 us      |
| runoff (no_dprst)     | 792 us    | 25.7 us | 40      | 19 us      |
| soilzone (no_dprst)   | 1380 us   | 46 us   | 70      | 19 us      |
| groundwater (no_dprst)| 226 us    | 1.7 us  | 12      | 19 us      |
| channel               | 188 us    | 34.5 us | 20      | 7.7 us     |
| hydraulic geometry    | 91 us     | 5.4 us  | 11      | 7.8 us     |

After the `_refs` cache (`asv compare 5d8e402 414ff9d`, same machine):
canopy 3.8 us, snow 16.9 us, runoff (no_dprst) 28.2 us, soilzone
(no_dprst) 48.0 us, groundwater (no_dprst) 1.9 us, channel 36.3 us,
hydraulic geometry 5.6 us -- each within a few us of its kernel (the
remainder = numba dispatch over long argument lists + scalar
parameter prep; snow's 9 us is the largest). Chain step 9.80 ms ->
1.92 ms. `time_init` grew by exactly the moved cost (one `.values`
per declared field, once per run: snow 1.63 ms). Full table:
`benchmarks/README.md` "After the `_refs` cache".

The physics is 0.5-18% of a step. The remainder is the lookups, at a
cost that is constant within a grid (~20 us on the hru dataset, ~8 us
on the smaller segment dataset) -- which is what makes the attribution
solid. Over the full chain that is ~7.5 ms of a 9.6 ms step; the
kernels together are ~0.2 ms. (How the kernel number is taken:
`benchmarks/README.md`, `ProcessKernel`.)

### The decision

A process fetches each field's ndarray ONCE, at bind time, and hands
those references to its kernel every step. `.values` is called at bind
(and rebind), never inside a step. The expected effect is a 30-50x
reduction of the compute part of a serial step. This is the memory
prime directive (`CLAUDE.md`: "reference, don't copy") applied to time
rather than bytes: the arrays handed to the framework ARE the working
buffers, so holding them is the natural thing, and re-deriving them
through xarray each step was never buying anything.

### The invariant it relies on

**One ndarray per name for the life of a run.** If a dataset ever
swapped in a new array under an existing name after a process bound
to it, a cached reference would point at a dead buffer and the kernel
would compute on stale data -- silently.

- _Serial_: holds by construction. Each grid's dataset is assembled
  once (`Model._add_process_fields`) and processes bind to it once
  (`cls(grid_ds)`).
- _MPI_: `ModelMPI.run` rebinds every distributed-grid process to each
  streaming window (`proc._obj = step`), so `self._obj` is a different
  `Dataset` object each step -- but the ndarrays behind the names a
  process reads are the same. mpixarray's `iterate`
  (`mpixarray/iterable.py`) re-points ONLY time-dimensioned variables
  present in the file (`read_from_source`: `var.data =
  src[name].data`) and leaves RAM-only buffers alone ("keeps its state
  across steps"); its per-step `assign_coords` is a shallow copy that
  keeps each Variable's data. Everything a pws_phoenix process reads is
  in the untouched set: the `(hru,)` input buffers `ModelMPI._build`
  declares shadow the raw file inputs and are refilled IN PLACE
  (`step[name].values[:] = src[name].values[0, :]`), statics are
  `.load()`ed once, state and outputs are declared RAM buffers.
  Serial (replicated) grids under MPI are plain datasets bound once.

Pinned by `tests/test_two_grid_mpi.py::test_step_buffers_persist_across_rebinds`:
a probe process records the address of every declared field it reads,
every step, through the normal `self._obj[name].values` path; each name
must map to one address. That test is the early, cheap pin of the
mechanism; once the cache exists, every parity test (serial and MPI,
all domains) is the comprehensive guard, because a swapped buffer
would fail them.

### Rules this imposes

1. **Never rebind a name to a new array after assembly.** Anything that
   must change a buffer's contents writes in place (`arr[:] = ...`):
   input refills, Map targets (`Map.apply` writes into `target_values`;
   `Map.bind`, called from `_resolve_maps`, holds the source and
   target ndarrays the same way a process holds `_refs`), restart
   restores (`_restore_grid_var`), warm-start seeding. Assigning `ds[name] = new_da` after bind is a bug.
2. **The cache is captured ONCE, at bind, and NOT at the MPI rebind**
   (`proc._obj = step` in `ModelMPI.run` only re-points the dataset).
   Recapturing would re-pay the lookups every step; the invariant
   says nothing moves, and `test_step_buffers_persist_across_rebinds`
   plus every MPI parity test guard it. (A debug-mode identity assert
   at rebind was considered and skipped.)
3. **Freezing is fine.** The post-initialize `writeable = False` flag
   on `parameter_internal` arrays changes a flag on the same ndarray,
   not the array.
4. **Variants that re-declare a field** (parameter -> time-varying
   input, as `PRMSAtmosphereTranspFrostDyn` does) change which buffer
   is assembled, not when it is captured -- no special case.

### Outcome (Oct 27, 2026)

- `ProcessLifecycle.time_calculate` on the implemented cache is within
  a few us of `time_kernel` for every process; the drb chain step fell
  from 9.80 ms to 1.92 ms (`benchmarks/README.md`, "After the `_refs`
  cache").
- One njit `_calculate` per leaf for the four multi-kernel processes
  (atmosphere, stream temperature, RunoffAg, SoilzoneAg/AgObsET) was
  measured and dropped: the Python between kernels is about 2 us in
  three of them, and moving AgObsET's obs-ET loop (about 21 us) into
  numba gained nothing (`benchmarks/README.md`, "The multi-kernel
  processes, measured").

## Numba cache: on for users, off for tests, opt-in per kernel

`pws_phoenix.NUMBA_CACHE` (read once at import from `PWS_NUMBA_CACHE`;
default on) feeds `@numba.njit(cache=NUMBA_CACHE)` on the kernels that
opt in. A cached kernel skips its compile (seconds per process) on
every later Python process: model runs, notebooks, benchmarks, each
MPI rank. The files are `*.nbi`/`*.nbc` in `__pycache__` beside the
source (gitignored).

Opt out for a whole environment (the developer setting; see why below):

    conda env config vars set PWS_NUMBA_CACHE=0 -n pwpx

`tests/conftest.py` forces `PWS_NUMBA_CACHE=0` before the first
`pws_phoenix` import, so every test run -- bare `pytest` or
`run_tests.sh`, serial or per MPI rank -- compiles fresh. The suite
never exercises the cached path; that path is numba's, not ours.

### What invalidates the cache (numba 0.64, `numba/core/caching.py`)

The index key is the call signature, the target machine, a hash of
the function's bytecode and its closure cells; the stamp that
invalidates an entry is the **defining file's** mtime and size.
Nothing else is tracked. The `.nbc` holds the linked machine code of
every jitted function the kernel calls and the frozen value of every
global it reads (numba freezes module globals as constants).

**If a caller has callees jitted in separate files, stale callees are
not discovered by the caller.** Edit `prms_stream_shade.py`: its own
helpers' entries invalidate, but `prms_stream_temp.py`'s `.nbc` still
contains the old shade code and runs it -- wrong numbers, no error.
The same for a constant imported from another module (which is why
`prms_atmosphere` keeps its `_SOLF` table in its own file rather
than in `solar_constants.py`). Hence the rule:

- opt in (`cache=NUMBA_CACHE`) only where every jitted callee and
  every captured constant lives in the kernel's own file:
  groundwater, runoff (+Ag, +NoDprst), soilzone (+NoDprst), canopy,
  snow, channel, hydraulic geometry, stream shade, atmosphere,
  `maps.py`, `processes_concrete.py`;
- leave plain `@numba.njit`, with a comment naming the callee file:
  `prms_stream_temp` (-> stream shade), `prms_soilzone_ag` (->
  soilzone), `starfit_source_sink_flow_node` and
  `starfit_daily_flow_node` (-> starfit_flow_node), and the
  FlowGraph kernel (built per
  composition; closes over the node types' Dispatcher tuples).

Helpers may be cached freely: a stale helper entry is only ever read
by a caller that is itself fresh. And under `pip install -e` a
developer who edits a file with the cache on is still safe for the
opt-in set -- numba's own file stamp sees the edit -- which is why the
exclusions above exist: they make the forgot-to-set-the-variable case
harmless too.

### MPI: no barrier

The cache is read only at a kernel's first call in a process (index,
stamp check, load). A miss compiles in memory -- LLVM never touches
the cache files -- then writes `.nbc` and `.nbi` by temp file +
`os.replace`. Four cold ranks each compile and write identical bytes;
a rank that reads another's just-renamed file gets a hit on the same
content; a partial file is never visible. No coordination needed.

### Measured

`benchmarks/_common.get_model()` (the drb stream-temperature chain:
build + one step, every kernel compiled), LaNueva, Oct 28, 2026,
import time excluded: **5.1 s cold** (cache wiped) -> **1.9 s warm**.
The warm 1.9 s still held the uncached atmosphere (since opted in,
Oct 30, 2026: its `solf` table moved in-file) and stream-temperature
compiles plus the data loading and assembly.

## Numba threading: `prange` over the HRU loop, 1 thread unless asked

The main `_calculate` of every hru-grid process -- atmosphere, canopy,
snow, runoff (+Ag, +NoDprst), soilzone (+Ag, +NoDprst), groundwater
(+NoDprst) -- is ALWAYS compiled `@numba.njit(parallel=True)`, and its
single per-element loop is `numba.prange`. The Model sets how many
threads they run on: `control["numba_n_threads"]`, default 1
(`Model._set_numba_threads`, which calls `numba.set_num_threads` at
build and again at the top of `run()`). Results are bit-identical at
any count: nothing crosses elements, so there is no reduction and no
float reordering (`tests/test_numba_threads.py` compares every
variable of the drb NHM model on 1 and on 3 threads).

Why always threaded (Nov 19, 2026): the earlier switch
(`PWS_NUMBA_PARALLEL`, read at import) compiled one of two kernels,
and numba's cache does not tell them apart -- a threaded run loaded
the unthreaded kernels an earlier run had saved (CONUS, Hovenweep:
first step 0.6 s, no kernel faster than unthreaded). With one kind of
kernel there is nothing to mix up. A threaded kernel held to 1 thread
costs nothing measurable:

| kernels (median ms per step, model without channel) | drb, 765 hrus (LaNueva) | 32 processes of 3,592 hrus (Hovenweep) |
| --- | --- | --- |
| unthreaded | 0.185 to 0.188 | 1.51 |
| threaded, held to 1 thread | 0.177 to 0.188 | 1.51 to 1.53 |
| threaded, all 14 threads | 0.79 to 0.80 | -- |

The price is the compile: about twice as long (drb first step 7.3 s
against 3.7 s with the cache off), paid once per machine with the
cache on and by every test process.

Rules that follow from `numba.set_num_threads`:

- The maximum is numba's thread pool, `numba.config.NUMBA_NUM_THREADS`,
  fixed when numba is imported: the `NUMBA_NUM_THREADS` environment
  variable when set, else the cores the process may use (a process
  bound to one core by `srun -c 1` gets 1). A
  `control["numba_n_threads"]` above it raises at build; under MPI the
  ranks agree first (the smallest maximum), then all raise.
- So `NUMBA_NUM_THREADS` is only a ceiling. Set above 1 with no
  `control["numba_n_threads"]`, the run is on 1 thread and the build
  warns (rank 0 only under MPI).
- The count belongs to the calling thread of the process and is not
  restored: other numba code run after a Model is built gets the
  Model's count.
- A kernel called WITHOUT a Model runs on whatever was last set --
  every core in a fresh process. `tests/conftest.py` calls
  `numba.set_num_threads(1)` at session start for that reason
  (`tests/kernel_units.py` builds no Model).

Deliberately NOT threaded:

- the light per-element helpers (`_areas`, `_update_areas`,
  `_pre`/`_post`, `_iter_diag`, `_compute_ccov_hru`, `_transp_tindex`,
  `_transp_frost`): a parallel region costs a thread launch (~10-30
  us), more than those loops;
- `_prep_aet` / `_adjust_irrigation` in soilzone_ag: an integer counter
  carried across iterations;
- the segment kernels (`for jseg in segment_order` is sequential) and
  the FlowGraph kernel;
- budget sums (a threaded sum changes the float order).

### Threads vs ranks

Under MPI `control["numba_n_threads"]` is PER RANK, and each rank
starts its own pool of `NUMBA_NUM_THREADS` threads (default every
logical core, 14 on LaNueva), of which it uses the count; the rest
stay idle (measured on Hovenweep, Nov 19, 2026: 128 idle threads in
each of 32 processes cost nothing). ranks x threads must fit the
cores. MPI already divides the HRUs, so threads help under MPI only
when the rank count is limited by memory, not cores. For an ensemble
(calibration), independent processes beat threads: no launch cost,
perfect scaling.

Measured on CONUS (114,958 hrus, Hovenweep, Nov 19, 2026; the six hru
kernels of one step): 72.4 ms on 1 core, 18.0 ms on 8 threads, 14.1 on
32, 11.0 on 128 -- 3.2 times at 8 threads, 5.2 times at 128. The same
hrus as 64 independent processes without the channel: 0.96 ms per
step.

### Where it pays (estimate; measurement pending)

Per kernel call: `t_serial ~ n_hru x c` with c ~ 50-300 ns per HRU
(groundwater light, snow/soilzone heavy); `t_parallel ~ L +
t_serial / n_threads` with L ~ 10-30 us per region launch. Break-even
at ~100-500 HRUs; >= 2x on 8 threads at ~300-2000. drb_2yr (765 HRUs)
sits on the line; ucb_2yr (3851) should gain. The per-step framework
cost (~0.1 ms) does not shrink with threads, so the chain-level gain
lags the kernel-level one until ~10k HRUs; M-series efficiency cores
finish their slice last and lengthen every region.

**To be measured (Nov 2026):** ucb_2yr full run at
`control["numba_n_threads"]` = 1, 4, 10, 14 (14 shows the
efficiency-core tail). Record the four numbers here.
