# Developer notes

Design decisions with their evidence, and the invariants they impose on
anyone writing a `Process`. (How we work -- tests, envs, review -- is
README "Installation" and `CLAUDE.md`; the pywatershed port record is
`docs/PORTS.md`.)

## Per-step data access: hold the ndarray, not the Dataset lookup

**Status (Oct 27, 2026): measured, decided, IMPLEMENTED** --
`Process.__init__` captures `self._refs: dict[str, np.ndarray]` (one
entry per declared field, `_resolved_fields`), and every
`advance()`/`calculate()`/`initialize()` reads `refs[name]`; no
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
   input refills, Map targets (`Map.apply` already writes into
   `target_values`), restart restores (`_restore_grid_var`), warm-start
   seeding. Assigning `ds[name] = new_da` after bind is a bug.
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

### Open

- Re-run `ProcessLifecycle.time_calculate` on the implemented cache --
  expected to fall to about `time_kernel` plus numba dispatch (a few
  us for long argument lists) plus ~40 ns per dict lookup.
- The four multi-kernel processes (atmosphere, stream temperature,
  RunoffAg, SoilzoneAg/AgObsET) get one njit `_calculate` per leaf in
  the same pass (njit-to-njit calls are free; the obs-ET iteration can
  move into numba), which also gives them `time_kernel` rows.
