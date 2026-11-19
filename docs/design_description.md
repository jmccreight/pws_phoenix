<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Design descriptions](#design-descriptions)
  - [Framework](#framework)
    - [One shared dataset per grid, the same structure in serial and MPI](#one-shared-dataset-per-grid-the-same-structure-in-serial-and-mpi)
    - [Variables classed by their relation to model time](#variables-classed-by-their-relation-to-model-time)
    - [Global state split: `Time` passed down, options consumed at build](#global-state-split-time-passed-down-options-consumed-at-build)
    - [Maps: a per-variable correspondence between grids, applied once per step](#maps-a-per-variable-correspondence-between-grids-applied-once-per-step)
    - [MPI: one distributed grid, serial grids replicated on every rank](#mpi-one-distributed-grid-serial-grids-replicated-on-every-rank)
    - [Memory: reference, do not copy; kernels write in place](#memory-reference-do-not-copy-kernels-write-in-place)
    - [The input contract: `Model.input_spec` and three classes of parameter](#the-input-contract-modelinput_spec-and-three-classes-of-parameter)
    - [Initial state: the restart variables, supplied under their own names](#initial-state-the-restart-variables-supplied-under-their-own-names)
    - [Three init hooks: `allocate`, `init_from_params`, `init_state`](#three-init-hooks-allocate-init_from_params-init_state)
    - [Reuse of a built model: `reset()`, `set_parameters()`, `ResultCollector`](#reuse-of-a-built-model-reset-set_parameters-resultcollector)
    - [Restart: flagged prognostic state, one file format for serial and MPI](#restart-flagged-prognostic-state-one-file-format-for-serial-and-mpi)
  - [Ports and variants](#ports-and-variants)
    - [Port conventions: names verbatim, element-loop kernels, pywatershed's own tolerance](#port-conventions-names-verbatim-element-loop-kernels-pywatersheds-own-tolerance)
    - [Variants by addition: a minimal base class that variants extend](#variants-by-addition-a-minimal-base-class-that-variants-extend)
    - [Ag variants: extend on a superset interface, sibling otherwise](#ag-variants-extend-on-a-superset-interface-sibling-otherwise)
  - [FlowGraph](#flowgraph)
    - [FlowGraph: node types as data, one kernel calling each type's functions](#flowgraph-node-types-as-data-one-kernel-calling-each-types-functions)
    - [STARFIT nodes: frozen parameters, unit factors, two time modes](#starfit-nodes-frozen-parameters-unit-factors-two-time-modes)
  - [IO](#io)
    - [Output backends differ on purpose: zarr in serial, NetCDF stream under MPI](#output-backends-differ-on-purpose-zarr-in-serial-netcdf-stream-under-mpi)
    - [Serial output: a pre-sized zarr store written by region](#serial-output-a-pre-sized-zarr-store-written-by-region)
  - [Checks](#checks)
    - [Budgets as data in the owning Process](#budgets-as-data-in-the-owning-process)
    - [Units mode: closed vocabulary, two boundaries, kernel tests](#units-mode-closed-vocabulary-two-boundaries-kernel-tests)
    - [Tests: no skips, pinned answer keys](#tests-no-skips-pinned-answer-keys)
  - [Performance](#performance)
    - [Benchmarks: ASV in the existing environment](#benchmarks-asv-in-the-existing-environment)
    - [Per-step data access: hold the ndarray, not the Dataset lookup](#per-step-data-access-hold-the-ndarray-not-the-dataset-lookup)
    - [Numba: cache on by default, threaded kernels held to 1 thread](#numba-cache-on-by-default-threaded-kernels-held-to-1-thread)
  - [Translation and migration](#translation-and-migration)
    - [PRMS translation: one way, pyPRMS only, outside the core](#prms-translation-one-way-pyprms-only-outside-the-core)
    - [Pre-processing: derived inputs in a stamped file, verified at use](#pre-processing-derived-inputs-in-a-stamped-file-verified-at-use)
    - [A domain translated ahead of time: files written once, options from the control copy](#a-domain-translated-ahead-of-time-files-written-once-options-from-the-control-copy)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Design descriptions

One entry per completed design in pws_phoenix: what was decided and why.

Purpose:

1. A conceptual hook for agents and humans: read the table, then the
   entry, then follow its pointers.
2. A tally of completed designs and the motivations behind them.
3. A backup for context that was moved out of agent memory, including
   the unfinished edges of a design.

Rules:

- Why, not how. Mechanics live in the code, `CLAUDE.md` and
  `docs/PORTS.md`; an entry points there.
- Completed designs only. Candidates and research stay elsewhere.
- "Left unfinished" records context. An item someone still intends to
  do is also tracked wherever work is tracked.
- Every entry has the same fields, in the same order.

Terms:

- kit: the set of pieces needed to build something, handed over
  unassembled so any piece can be inspected or changed first.
  `ModelKit` (from `prms_translate.assemble_from_control`) holds
  everything `Model()` takes: `process_dict`, `control`, `maps`,
  `discretizations`; `kit.model()` builds the Model. `FlowGraphKit`
  (from `splice_flow_graph`) holds the pieces of a graph: the
  composed class, the `nnodes` grid, its parameters, the new nodes'
  initial state.

| design | status | decided | section |
|---|---|---|---|
| Budgets as data in the owning Process | built | Oct 2026 | Checks |
| Units mode: closed vocabulary, two boundaries, kernel tests | built; kernel tests partial | Nov 2026 | Checks |
| Tests: no skips, pinned answer keys | built | Oct 2026 | Checks |
| Port conventions: names verbatim, element-loop kernels, pywatershed's own tolerance | built | Jul 2026 | Ports and variants |
| Variants by addition: a minimal base class that variants extend | built | Jul 2026 | Ports and variants |
| Ag variants: extend on a superset interface, sibling otherwise | built | Jul 2026 | Ports and variants |
| FlowGraph: node types as data, one kernel calling each type's functions | built | Jul 2026 | FlowGraph |
| STARFIT nodes: frozen parameters, unit factors, two time modes | built | Jul 2026 | FlowGraph |
| One shared dataset per grid, the same structure in serial and MPI | built | Jun 2026 | Framework |
| Variables classed by their relation to model time | built | Jun 2026 | Framework |
| Global state split: `Time` passed down, options consumed at build | built; options still a dict | Jun 2026 | Framework |
| Maps: a per-variable correspondence between grids, applied once per step | built | Jul 2026 | Framework |
| MPI: one distributed grid, serial grids replicated on every rank | built | Jul 2026 | Framework |
| Memory: reference, do not copy; kernels write in place | built | Jul 2026 | Framework |
| The input contract: `Model.input_spec` and three classes of parameter | built | Aug 2026 | Framework |
| Initial state: the restart variables, supplied under their own names | built | Nov 2026 | Framework |
| Three init hooks: `allocate`, `init_from_params`, `init_state` | built | Nov 2026 | Framework |
| Reuse of a built model: `reset()`, `set_parameters()`, `ResultCollector` | built, serial only | Nov 2026 | Framework |
| Restart: flagged prognostic state, one file format for serial and MPI | built | Aug 2026 | Framework |
| Benchmarks: ASV in the existing environment | built; not in CI | Oct 2026 | Performance |
| Per-step data access: hold the ndarray, not the Dataset lookup | built | Oct 2026 | Performance |
| Numba: cache on by default, threaded kernels held to 1 thread | built; threads measured on CONUS, not on ucb_2yr | Nov 2026 | Performance |
| Output backends differ on purpose: zarr in serial, NetCDF stream under MPI | built | Jun 2026 | IO |
| Serial output: a pre-sized zarr store written by region | built | Oct 2026 | IO |
| PRMS translation: one way, pyPRMS only, outside the core | built | Aug 2026 | Translation and migration |
| Pre-processing: derived inputs in a stamped file, verified at use | built | Aug 2026 | Translation and migration |
| A domain translated ahead of time: files written once, options from the control copy | built, serial run entry only | Nov 2026 | Translation and migration |

## Framework

### One shared dataset per grid, the same structure in serial and MPI

- **Status:** decided June 2026; built June-July 2026; on develop.
- **Why:** the first framework built one dataset per process and
  wired shared buffers by hand, with `a.values is b.values` asserts
  to prove the wiring. MPI needs a unit to decompose. Making the grid
  that unit answers both: two processes that name the same variable
  share it because it is the same variable.
- **Decision:**
  - The discretization (grid) is the unit of MPI decomposition, and
    owns one `xr.Dataset` holding every variable of its processes.
  - Buffer sharing is structural: a name is added to a grid's dataset
    once.
  - Serial and MPI have one object model. A serial run uses no MPI
    and no mpixarray: each grid's dataset is a plain `xr.Dataset` at
    full extent. Under MPI the distributed grid's dataset is an
    mpixarray dataset holding one rank's block. A process cannot tell
    which it is bound to; the two paths differ only inside
    `Discretization` (`decompose()` is the identity in serial,
    `parallelize` under MPI) and in the Model's run loop.
  - The dataset is flat, not a DataTree: a tree inherits coordinates,
    not data variables, shared variables cut across any tree, and
    mpixarray cannot stream one.
  - A Discretization may own parameters (areas, topology); assembly
    looks there first, then in the process's own parameters.
  - `finalize` releases external resources. Serial keeps its data in
    memory for inspection; MPI closes its dataset.
- **For users:**
  - Two processes on one grid cannot each have a private variable of
    the same name: same names share, silently.
  - An input's dimension must be the grid's; a mismatch raises at
    build.
- **Where:** `CLAUDE.md` "Object model";
  `pws_phoenix/discretization.py`; `Model._add_process_fields`;
  `tests/test_two_grid.py`.
- **Left unfinished:**
  - Detecting a name collision at assembly is decided, not built.
  - A DataTree may return only as a view for output groups.

### Variables classed by their relation to model time

- **Status:** decided and built June 2026; on develop.
- **Why:** a time-varying parameter had been treated as neither
  static nor streamed. What separates the cases cleanly is the
  variable's time coordinate set against model time.
- **Decision:**
  - Static parameter: no time axis. Loaded once, read-only.
  - Time-varying parameter: an axis that is not model time (month,
    day of year). Kept whole, read-only; the process looks up the
    current slice from the clock.
  - Input: its time axis is model time. One slice is served per step.
  - A parameter declared on the model `time` axis is a misdeclaration
    and raises.
  - The clock reaches `calculate()`; a kernel receives arrays and
    scalars (a month index), never the `Time` object.
- **For users:**
  - Calendar fields are 1-based; a positional lookup uses
    `field - 1`.
  - An input a process edits in place is `kind="mutable_input"`.
- **Where:** `CLAUDE.md` "Variable taxonomy"; `DataArrayMeta` in
  `pws_phoenix/process.py`; `tests/test_time.py`.
- **Left unfinished:**
  - Parameters on a coarser linear axis (per year, per year and
    month) are not implemented.

### Global state split: `Time` passed down, options consumed at build

- **Status:** `Time` built June 2026; options are still the `control`
  dict.
- **Why:** pywatershed bundles the clock and the configuration in one
  Control object that every process holds. They have different
  lifetimes: every step needs the clock; configuration is used once,
  at build.
- **Decision:**
  - `Time` is an argument of `calculate(dt, time)`; `advance()` takes
    no time.
  - Serial: the Model drives the clock. MPI: mpixarray owns the loop
    and the clock is set from each step. A process reads it the same
    way in both.
  - Options are consumed at build; a process never needs them at run
    time.
- **For users:**
  - `dt` is in seconds (86400.0 for a daily model).
- **Where:** `CLAUDE.md` "Global state"; `pws_phoenix/clock.py`;
  `pws_phoenix/config.py` (`load_model_yaml`, a first probe of
  options from a file).
- **Left unfinished:**
  - An `Options` class and `Model.from_yaml()`.

### Maps: a per-variable correspondence between grids, applied once per step

- **Status:** built July 2026; on develop. Weights held as their
  nonzero entries since Nov 2026.
- **Why:** pywatershed aggregates hru values to segments inside its
  channel and stream temperature processes, which makes them two-grid
  processes without saying so. Coupling grids needs one explicit
  concept with one MPI treatment. A full weight matrix does not fit a
  large domain: 58,489 segments x 114,958 hrus x 8 bytes is 54 GB,
  for 114,958 entries that are not zero. The first serial run of that
  domain peaked at 51.5 GB, nearly all of it this matrix.
- **Decision:**
  - A Map carries one variable from a source grid to a target grid
    through static weights; a rename or a static factor is part of
    it. It never originates a quantity: a calculation belongs to a
    process on the grid where its inputs live (cloud cover moved into
    PRMSAtmosphere for this reason).
  - Maps are declared once and never placed in the schedule. Each is
    applied exactly once per step, just before its first consumer.
  - A mapped value is then constant for the rest of the step. That is
    checked at build (every declared writer precedes the first
    consumer), not tracked at run time.
  - Process order is the order of `process_dict`. Every `advance()`
    runs before any `calculate()`, so a read of another process's
    `*_prev` is the prior step's value.
  - One pass per step; iterative coupling between grids is out of
    scope.
  - Fluxes are mapped separately and summed in the consumer's kernel.
  - Weights for a pywatershed aggregation are derived by probing its
    kernel with basis vectors.
  - A Map holds only the nonzero entries of its weights
    (`maps.SparseWeights`: row index, column index and value of each
    entry, plus the shape). A full matrix is still accepted, and is
    converted at construction and not kept. The entries are held
    sorted by (row, column), the order the full matrix gives, so the
    sums are the same bit for bit; a zero value is dropped and a
    repeated (row, column) raises.
  - One form in a file, for the pre-processing file and for a weights
    file of the model yaml: three 1-D variables, `<name>_rows`,
    `<name>_cols`, `<name>_vals`. Our own small class, not
    `scipy.sparse`: `maps.py` imports only numpy, xarray and numba.
- **For users:**
  - `Map(weights=..., grid={"hru": "segment"},
    variable={"flow": "flow"})`, with `weights` a full
    `(n_target, n_source)` array or a `SparseWeights(rows, cols,
    vals, shape)`. An unused Map or a weights shape that does not fit
    the grids raises at build.
  - `mm.weights.dense()` builds the full matrix (a new array at every
    call): for inspection and small domains.
  - A map's `weights:` file in the model yaml holds the full matrix
    as its one variable, or `weights_rows` / `weights_cols` /
    `weights_vals` (`SparseWeights.to_dataset("weights")`).
- **Where:** `CLAUDE.md` "Schedule and Maps";
  `pws_phoenix/maps.py` (docstring); `Model._resolve_maps`;
  `tests/test_maps.py`; `docs/PORTS.md` "The stream-temperature
  chain".
- **Left unfinished:**
  - `derive_aggregation_weights` (stream temperature) still fills four
    full matrices in memory before they are converted. It needs to
    collect each probe's nonzero entries before a large domain can
    run stream temperature.
  - No Map from a serial grid to the distributed grid.
  - One Map carrying several variables through the same weights (one
    batched product).
  - A default process order with subsetting.

### MPI: one distributed grid, serial grids replicated on every rank

- **Status:** built July 2026; on develop.
- **Why:** the real pattern is a large hru grid feeding a small
  channel network, and mpixarray streams one decomposed dataset.
- **Decision:**
  - An MPI run has one distributed grid (`control["mpi_grid"]`) and
    any number of serial grids, each replicated on every rank.
    Replication was chosen over one owning rank: every rank runs the
    same loop, with no rank branch that could leave a collective call
    unmatched.
  - `MapMPI` crosses the boundary: each rank's partial product, then
    `Allreduce(SUM)`.
  - Output follows the owning grid: distributed variables go through
    mpixarray's NetCDF stream, serial-grid variables to a zarr store
    written by rank 0.
  - A raise is collective: the ranks agree first, then all raise. One
    rank raising alone hangs the others.
  - FlowGraph needs no MPI code: its grid sits where the segment grid
    sits.
  - This is interim; mpixarray's planned streaming DataTree is
    expected to take over the role.
- **For users:**
  - `control["mpi_grid"]`, `control["output_parallel_netcdf"]`,
    `control["output_serial_zarr"]`.
  - Segment-grid results agree with serial to about 1e-9 (sum order);
    hru-grid results are bit for bit.
- **Where:** `CLAUDE.md` "MPI path"; `ModelMPI` in
  `pws_phoenix/model.py`; `MapMPI` in `pws_phoenix/maps.py`;
  `tests/test_two_grid_mpi.py`.
- **Left unfinished:**
  - A derived buffer with a second dimension cannot be declared
    (mpixarray decomposes every dimension).
  - The distributed grid's inputs are one combined file; structured
    input is undecided.

### Memory: reference, do not copy; kernels write in place

- **Status:** adopted July 2026; on develop.
- **Why:** two goals of the rewrite are speed and a small memory
  footprint, and both should be obvious from reading the code.
- **Decision:**
  - An array handed to the framework is the working buffer. The Model
    copies only the structure of `process_dict`.
  - Every allocation is one of: model state, allocated once at
    assembly; a declared per-step window buffer; a temporary inside a
    numba kernel. Anything else is a bug or carries a comment.
  - A kernel takes its output buffers first, writes in place, returns
    nothing and allocates nothing per step. A ported kernel is an
    explicit element loop with scalar temporaries, because numba does
    not remove named intermediate arrays.
- **For users:**
  - The arrays you pass become the model's buffers, and its read-only
    flag on parameters lands on your arrays.
  - Hidden copies to watch for: `deepcopy`, `np.asarray` / `astype`,
    fancy indexing, xarray selection on lazy data.
- **Where:** `CLAUDE.md` "Prime directive: memory";
  `tests/test_ref_behaviors.py`; `DEVELOPER.md`.
- **Left unfinished:** none.

### The input contract: `Model.input_spec` and three classes of parameter

- **Status:** built July-Aug 2026; on develop.
- **Why:** what a model needs from its user was spread over the
  declarations of its processes. A user should be able to ask before
  having any data, and the PRMS translator needed that answer as its
  job description.
- **Decision:**
  - `Model.input_spec(process_dict, maps)` is a classmethod that
    needs no data. Its top level is required / maps / optional, then
    grid, then category.
  - Required: external inputs, parameters, derivable parameters,
    initial state. Optional, only on request: inputs another process
    supplies, internal parameters, inputs a Map supplies.
  - Three classes of parameter: authored (supplied); derivable
    (required, but it can be generated: `DataArrayMeta.derivation`
    names its inputs); internal (`kind="parameter_internal"`,
    computed at init, never supplied).
  - A Map states its weights requirement and their `derivation` in
    the same contract.
  - Which file a parameter comes from is not in the contract; that is
    the supplier's job.
  - `model_contract_graph.py` draws the contract. Graphviz
    (`to_dot()`) is the renderer of record, because the flow must run
    along one axis in schedule order; mermaid is the fallback.
- **For users:**
  - Call `Model.input_spec(...)` first; `examples/00_input_contract.py`
    walks through it.
  - When changing the graph code, render with `dot -Tpng` and look at
    the image.
- **Where:** `Model.input_spec` in `pws_phoenix/model.py`; the
  `DataArrayMeta` docstring in `pws_phoenix/process.py`;
  `pws_phoenix/model_contract_graph.py`; `tests/test_input_spec.py`;
  `examples/00_input_contract.py`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - A Map does not declare the source variable it needs. A forcing
    read only by a Map therefore needs a carrier process
    (`HumidityCarrier` in `prms_translate/assemble.py`).
  - Compute-if-absent: a derivable parameter is required today; later
    the model could generate it when it is not supplied.

### Initial state: the restart variables, supplied under their own names

- **Status:** decided Oct 23, 2026; built Nov 10-11, 2026; on develop
  (PR 7).
- **Why:** `DataArrayMeta.initial=` let a state variable take its
  start from a parameter. A test then ran on all-nan state unnoticed:
  a fixture never supplied `gwstor_init`, and `assert_array_equal`
  treats nan as equal to nan. JLM: "consuming state from parameters is
  a bad design smell", and keeping `initial=` at all was "too
  generous". Restart, supplied state and a later calibration reset
  should also be one path, not three.
- **Decision:**
  - Starting state is the `restart=True` variables. One order in
    serial and MPI: allocate, `init_state(time)` sets each one's
    default, then supplied state or `control["restart_read"]`
    overrides it.
  - A process never reads state from a parameter. PRMS's
    `gwstor_init` and `segment_flow_init` are translated only in
    `prms_translate.prms_initial_state`.
  - `segment_flow_init` becomes `seg_inflow`, not `seg_outflow`:
    `seg_outflow` is rebuilt from zero every step, and the parameter
    only ever reached routing through the previous step's inflow.
  - Every process has a default (PRMSGroundwater: PRMS's declared
    2.0 inches). There is no nan check, because nan is legitimate
    state (stream temperature's never-has-flow marker, FlowGraph rows
    of other node types, STARFIT `lake_storage`).
  - State with no sensible default is declared
    `required_initial_state` (first user: STARFIT
    `lake_storage_sub`); `check_initial_state()` reports bad starting
    state and the Model raises on every rank.
  - `state_groups` name restart variables that must be supplied
    together, because `init_state(time)` computes some members from
    the others' defaults.
  - The `*_init_frac` parameters stay process parameters (Nov 12,
    2026): they are fractions of maxima that calibration changes, so
    an absolute supplied state would stop following a calibrated
    maximum.
  - Under MPI the initial state is not in the combined input file:
    that file should not be fixed to one start.
- **For users:**
  - Supply starting state as an array under the variable's own name
    in the process's `process_dict` entry, or give
    `control["restart_read"]`. Both together raise.
  - Supply every member of a state group, or none.
  - A graph with a STARFIT node type must supply `lake_storage_sub`,
    in millions of cubic meters even in a cfs graph; with no
    reservoir rows it is an all-nan array.
  - Coming from PRMS, `prms_translate` does the translation of
    `gwstor_init` and `segment_flow_init`.
- **Where:** `CLAUDE.md` "Initial state";
  `docs/PORTS.md` "IC status"; `docs/MAINTENANCE.md`;
  `pws_phoenix/model.py` (`_apply_supplied_state`,
  `_restore_grid_var`); `pws_phoenix/process.py` (`state_groups`,
  `required_initial_state`, `check_initial_state`);
  `pws_phoenix/prms_translate/assemble.py` (`prms_initial_state`);
  `tests/test_restart*.py`, `tests/test_input_spec.py`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - One first-step block still overwrites supplied state on a cold
    start: StarfitDailyFlowNode's first substep, which needs the
    first substep's inflow. The others were closed by the init hook
    split ("Three init hooks").
  - Setting part of a state group (for example soil moisture alone,
    for data assimilation) needs derived state recomputed from
    whatever state is in place. `dprst_area_clos` is frozen at its
    initial value, so that is not uniformly clean.
  - No check yet of supplied soil state above its maximum, nor a
    warning for `slow_stor > pref_flow_thrsh` (zero interflow,
    `docs/MAINTENANCE.md`).
  - JLM: the argument that took initial state out of the combined
    MPI file applies to parameters too; forcings, parameters and
    initial state should be separate files.
  - STARFIT's requirement is by class composition; the alternative
    is a check on reservoir rows only.

### Three init hooks: `allocate`, `init_from_params`, `init_state`

- **Status:** decided and built Nov 12-13, 2026 on branch
  feat_spotpy, for every process and FlowGraph node type.
  `Model.reset()` and `Model.set_parameters()`, which the split
  prepares, are the next entry.
- **Why:** calibration runs one built model hundreds of times with
  changed parameters, and rebuilding the model for each run is the
  larger cost. The single hook `initialize()` allocated private
  arrays, derived parameters and set the starting state in one body,
  so no part of it could be run again alone: a second call
  reallocated, and a start that is a rule relative to a parameter
  (`soil_moist_init_frac * soil_moist_max`) could not follow a
  changed maximum. Separately, first-step blocks inside `calculate()`
  overwrote supplied state on a cold start.
- **Decision:**
  - `allocate()` runs once per model: private arrays and tuples of
    array references. It is the only hook that may allocate.
  - `init_from_params()` computes everything that is a function of
    the parameters: the `parameter_internal` fields, private derived
    values, and the parameter checks that raise. It writes no
    variable and may be run again.
  - `init_state(time)` writes the cold-start default of every
    variable and of private state, with the clock at index 0. It runs
    after `parameter_internal` is frozen, so a write to one raises.
    It may be run again.
  - The Model's order: bind, `allocate`, `init_from_params`, freeze,
    `init_state` and the budget accumulators, supplied state or a
    restart, `check_initial_state`. Each hook runs over all processes
    before the next. All three are local to a rank.
  - The `*_init_frac` parameters stay parameters. Their rule sits in
    `init_state`, so a later reset recomputes the start from the
    maximum then in place.
  - What `init_state` needs from the parameters it recomputes
    (soilzone's `soil_rechr` seed, taken before the clamps) or
    receives as a private array filled by `init_from_params` (stream
    temperature's `_never_has_flow` mask).
  - `init_state` reads no input: under MPI the inputs exist only
    inside the time loop. A first-step block that reads an input
    cannot move there. PRMSAtmosphere's transpiration block reads
    none and moved; the Ag area blocks were removed by another route
    ("Ag variants").
  - A composed FlowGraph forwards to its node types: the required
    `init_from_params_type` and the optional `allocate_type` and
    `init_state_type`.
- **For users:**
  - Writing a process: put each line in the hook whose contract
    covers it. A state default left in `init_from_params`, private
    state with no default in `init_state`, or an array created
    outside `allocate` are each caught by the two replays in
    `tests/test_init_replay.py`.
- **Where:** the hook docstrings in `pws_phoenix/process.py`;
  `Model._run_initialize_hooks` in `pws_phoenix/model.py`;
  `pws_phoenix/flow_graph.py`; `CLAUDE.md` "Parameters: owned by a
  grid, or computed at init"; `tests/test_discretization.py`
  (`TestInitHooks`), `tests/test_init_replay.py`.
- **Left unfinished:**
  - StarfitDailyFlowNode's first substep still overwrites supplied
    state on a cold start.
  - JLM's third phase, `init_diagnostic_state` (derived state
    recomputed from whatever state is in place), is deferred.
  - The toy processes of `tests/test_restart.py` still set state
    defaults in `init_from_params`.

### Reuse of a built model: `reset()`, `set_parameters()`, `ResultCollector`

- **Status:** decided and built Nov 14, 2026 on branch feat_spotpy,
  for the serial `Model`. `ModelMPI` raises for all three.
- **Why:** calibration runs one model hundreds of times with changed
  parameters; the target is calibration by HRU, up to 350,000 HRUs.
  Measured on drb (Nov 2, 2026, warm): a run is 0.26 s of compute
  against 1.07 s of build, 0.64 s of kernel cache load and 0.65 s of
  imports. Per run that is 2.6 s in a fresh Python process, 1.3 s
  with a rebuild in one session, and 0.26 s when the built model is
  run again: 5 times faster than a rebuild per run. The three init
  hooks made each part of the build repeatable; these three pieces
  are the Model's use of them.
- **Decision:**
  - `reset()` -- what it does:
    - It puts the model back to its state right after the build and
      stores no copy of that state. A stored copy would double the
      state in memory, and a start that is a rule relative to a
      parameter has to be recomputed after `set_parameters()`
      anyway.
    - Its order: refill every variable (budget variables included)
      with its allocation fill; rewind the inputs; zero the Map
      target buffers; `init_state` and the budget accumulators over
      all processes with the clock at index 0; the supplied state;
      `check_initial_state`; the step cursor back to the start.
    - The allocation fill comes first because some state relies on
      it (a STARFIT row with no reservoir stays nan), and because a
      default that only the build put in place then shows as a fill.
    - The supplied state is read again from the caller's arrays, so
      it can be edited between runs.
    - The inputs' current values return to nan and Map target
      buffers to 0.0, their values at construction. A correct model
      rewrites both before it reads them. The refill makes every
      array of a reset model equal a fresh build's, so the test
      compares all of them with no exclusions, and a process that
      wrongly reads an input before the first `advance()` gets nan,
      not the last run's final step.
    - Parameters and `parameter_internal` values are not touched.
  - `reset()` -- what it refuses:
    - A warm-started model (`control["restart_read"]`) raises: the
      restart state is not kept after the build. A reused model that
      needs a warmer start takes supplied state. The alternatives,
      reading the restart files on every reset or keeping them in
      memory, were left until a use appears.
    - A model with a zarr `Output` raises: a second run would
      overwrite the store, and every calibration run would pay for
      writes that nobody reads. It also catches a calibration script
      that left the zarr keys in its control.
    - A finalized model raises.
  - `set_parameters(values, grid=None)` -- what it does:
    - It writes into the model's existing parameter arrays. None is
      replaced, because every process and every kernel argument
      tuple holds references to them.
    - It then refills every `parameter_internal` with its allocation
      fill and runs `init_from_params` over all processes, as the
      build does, between an unfreeze and a refreeze. The fill makes
      the rerun equal a fresh build even where a hook writes a
      derived value only for some elements.
    - `set_parameters({})` runs the hooks again with the values in
      place; the parameter replay test is that call.
    - Values are in the declared units; there is no units check.
      They are broadcast to the parameter's shape (a scalar, or one
      value per element into a `(month, space)` parameter).
  - `set_parameters()` -- the guard:
    - It raises unless the model was built with
      `control["calibration_mode"] = True`, before any other check
      (`set_parameters({})` included). Outside calibration nothing
      should run `init_from_params` a second time or write the
      caller's parameter arrays.
    - A control key was chosen over an environment variable
      (`PWS_CALIB_MODE` was the first idea): the control is the same
      on every MPI rank, so all ranks raise together, where an
      environment variable can differ between ranks; it is known at
      the build; it marks the one model meant for calibration, not
      every model in the session; and options are build-time
      configuration in `control`, as `collect_var_names` is.
    - `reset()` is not guarded: it does not run `init_from_params`,
      and running a built model twice with the same parameters is
      not specific to calibration.
  - `set_parameters()` -- what it checks, all before anything is
    written:
    - The name is a supplied parameter. A `parameter_internal` name
      raises (set what it is computed from). A name that is a
      parameter on two grids needs `grid=`.
    - A parameter owned by a `Discretization` raises: the grid's
      topological order was built from it.
    - The dtype is of the same kind: a float into an integer
      parameter raises, where numpy would truncate silently.
    - The shape broadcasts, and the array can be made writeable (a
      view of a read-only array raises).
    - The model is one `reset()` accepts. Otherwise the model would
      be left needing a reset it can never get.
  - `set_parameters()` -- what follows it:
    - The state in place belongs to the old parameters, so `step()`
      raises until `reset()` is called. `set_parameters()` does not
      reset by itself: the reset stays visible in the caller's code,
      and two calls do not reset twice.
    - When a parameter check inside `init_from_params` raises, the
      new values stay written, the `parameter_internal` arrays are
      frozen again, and `reset()` and `step()` raise until a later
      `set_parameters()` succeeds. Falling back to the old values
      would need a copy of every parameter being set and would hide
      the failure (JLM: a raise is better). A calibration driver
      scores the sample as failed and sets the next one.
  - `ResultCollector`:
    - `control["collect_var_names"]` keeps the named variables in
      memory for the whole run, as `model.results`: one
      `(n_time, space)` array per name, allocated once at the build
      and filled with nan. Row `tt` is the variable after step `tt`.
    - It is a separate key and a separate class from the zarr
      `Output` because it is not output: nothing is written, and the
      caller computes on the arrays right after the run. "Memory"
      was kept out of the names, since it also means prognostic
      state here. One run may have both a zarr `Output` and a
      collector, with different variables.
    - A control key was chosen over an object attached after the
      build: it reads like `output_var_names` and passes through
      `model_from_control(..., control={...})`.
    - The collector holds each variable's ndarray ("Per-step data
      access"), so a step costs one row copy per name.
    - The row is the model's time index, so there is no cursor to
      rewind. `reset()` blanks nothing: blanking would write the
      whole buffer on every calibration run. `n_collected` counts
      the rows the current run has written and `reset()` sets it to
      0; a run shorter than an earlier one leaves the earlier rows
      past it.
    - First cut: every time step, the full spatial extent, float
      variables only (an integer buffer has no nan to mark an
      unwritten row). The size is `n_time` times the variable's
      size per name: 2 GB per variable at 350,000 HRUs over 2 years
      of daily steps.
- **For users:**
  - The loop:

    ```python
    control = {
        "calibration_mode": True,   # allows set_parameters()
        "collect_var_names": ["seg_outflow"],
    }
    model = model_from_control(control_file, control=control)  # once
    for values in samples:          # {parameter name: array}
        model.set_parameters(values)
        model.reset()
        model.run(dt, n_steps)
        score(model.results.values["seg_outflow"])
    ```

  - `model.results.values[name]` is the buffer itself, not a copy:
    the next run overwrites it.
  - `set_parameters()` writes the arrays the model was built from,
    and those are the caller's ("Memory: reference, do not copy").
    The caller's dataset changes, and so does any other model built
    from it in the same session, without its `init_from_params`
    being run.
  - Only `init_from_params` is run again. What was derived from
    parameters OUTSIDE the model is not: Map weights (in a translated
    PRMS model, from `hru_segment` and `hru_area`), the solar tables
    (from `hru_slope`, `hru_aspect`, `hru_lat`), supplied initial
    state translated from a parameter. The Model does not detect a
    change to one of these sources.
  - The solar radiation parameters (`dday_slope`, `dday_intcp`,
    `radmax`, `radadj_intcp`, `radadj_slope`, `ppt_rad_adj`,
    `tmax_index`) are ordinary parameters and can be set. The solar
    tables are parameters too: to change the topography behind them,
    recompute them with `compute_soltabs` and pass them in the same
    call.
  - A reused model starts from the `init_state` defaults plus
    supplied state. Some defaults follow the parameters and move with
    `set_parameters()`: the soil zone's `*_init_frac` times the
    maxima, runoff's `dprst_frac_init`, and the transpiration flag
    from `transp_beg` / `transp_end` and the start date. The others
    are fixed (groundwater's 2.0 inches; zeros for snow, canopy,
    impervious storage and the channel; stream temperature's -99.9)
    and take supplied state for any other start. STARFIT's
    `lake_storage_sub` has no default.
- **Where:** the docstrings of `Model.reset` and
  `Model.set_parameters` in `pws_phoenix/model.py`; `Input.reset`
  and `ResultCollector` in `pws_phoenix/data_io.py`. Tests, all bit
  for bit, in `tests/test_init_replay.py`:
  - run / reset / run: `test_state_replay` (toy),
    `test_drb_stream_temp_kit_replays`,
    `test_ucb_starfit_graph_replays`;
  - set / reset / run against a fresh build holding the new values:
    `test_set_parameters` (toy), `test_drb_set_parameters` (five
    parameters, one of each kind of consequence);
  - the refusals and the failed parameter check: `test_reset_raises`,
    `test_set_parameters_raises`;
  - the guard: `test_set_parameters_needs_calibration_mode`;
  - the collector against the live variables, the zarr `Output` and
    a repeated run: `test_result_collector` (toy),
    `test_drb_result_collector`.
- **Left unfinished:**
  - MPI: the mpixarray stream owns the time loop, and rewinding it
    (a new `as_iterable`) is undecided. `set_parameters()` and the
    collector under MPI are not designed either. No test covers the
    three `ModelMPI` raises.
  - Collector subsets, if the objective functions need them: a
    spin-up window, a spatial subset, float32, or an objective
    accumulated step by step with no full buffer.
  - `reset()` of a warm-started model (read the restart files again,
    or keep them in memory) and of a model with a zarr `Output`
    (rewind the writer).
  - No test sets a name that is a parameter on two grids. The ucb
    FlowGraph model is covered by the replay (`set_parameters({})`)
    and run / reset / run, not by new values against a fresh build.
  - A check that refuses, or recomputes, what is derived outside the
    model. The `derivation=` strings on the declarations already name
    the sources of the solar tables.
  - One user-facing description of where each process's cold-start
    state comes from.
  - The evaluator that drives this loop for calibration by HRU, and
    where it lives (pws_phoenix or the calibration repository).
  - A timing of the loop itself: the 5 times figure is the Nov 2
    measurement of a run against a build, not of these methods.

### Restart: flagged prognostic state, one file format for serial and MPI

- **Status:** built Aug 2026 (serial, Ag, FlowGraph node types, MPI);
  on develop.
- **Why:** pywatershed keeps a hand-written list of restart variables
  per class. Here the state a run must carry is part of each
  variable's declaration, and the same set serves state updating
  (data assimilation) and supplied initial state.
- **Decision:**
  - `DataArrayMeta(restart=True)` marks prognostic state;
    `get_restart_variables()` is derived from the flags. Only current
    variables are flagged; `advance()` regenerates the `*_prev`
    copies.
  - State held in Python attributes travels through
    `get_restart_state()` / `set_restart_state()` (stream
    temperature's silos and indices are the only case).
  - Restart files are full extent and in the serial format on both
    paths, so a serial run resumes from an MPI run and the reverse.
    Under MPI each rank restores its own block.
  - An MPI resume slices the input dataset lazily in time before
    decomposition (JLM's call): the stream begins at the resume step,
    and one input file serves any restart.
  - Model time stays indexed from the original start, so first-step
    blocks do not run again.
  - The test is a perfect restart: every variable bit for bit, for
    each process and for the full chain.
- **For users:**
  - `Model.write_restart(dir)` writes; `control["restart_read"]`
    resumes at the following step.
  - A new process flags its prognostic variables. The perfect-restart
    test of a full chain is what finds a missed one (it found snow's
    `pk_temp`).
  - Attribute-held state on a distributed grid raises under MPI.
- **Where:** `docs/PORTS.md` "Restart"; `write_restart` and
  `_restore_grid_var` in `pws_phoenix/model.py`;
  `tests/test_restart*.py`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - Stream temperature's attribute-held state is outside the
    declarations, so the units check, supplied state, `input_spec`
    and a distributed grid do not cover it.

## Ports and variants

### Port conventions: names verbatim, element-loop kernels, pywatershed's own tolerance

- **Status:** set by the groundwater port, July 2026; every NHM
  process ported by July 2026; on develop.
- **Why:** a goal of the rewrite is a workable path for porting
  pywatershed's processes. Agreement with pywatershed's answer files
  is the proof of a port, and identical names are the way back to its
  domains and files.
- **Decision:**
  - Parameter, input and variable names are pywatershed's. The one
    kind of new name is a flux mapped onto another grid
    (`seg_sroff_vol`).
  - A kernel is pywatershed's numpy function rewritten as an explicit
    element loop that writes in place, keeping the order of
    operations per element.
  - Compiled strictly (no fastmath). pywatershed's numba path uses
    fastmath, which is the source of every limit on agreement.
  - Each process is held to the tolerance of pywatershed's own
    autotest for that process, against its generated answers.
  - Not ported: adapters, the `calc_method` switch (numba is the only
    path), `verbose`, declared but unused parameters.
  - Upstream quirks are kept verbatim and recorded. Where upstream
    edits a true parameter, the port raises `NotImplementedError`
    instead. A departure from upstream is deliberate and listed
    (`docs/MAINTENANCE.md`).
  - Solar geometry is a function that makes parameters
    (`compute_soltabs`), not a Process.
- **For users:**
  - pywatershed's output files use the dimension `nhm_id`, its
    parameter files `nhru`: rename to the grid's dimension.
  - Snow matches pywatershed's numpy path bit for bit. The answer
    files come from its fastmath path, which differs by about 1e-8
    and flips pack survival on about 0.02% of hru-days. Tests
    downstream of snow therefore use an outlier-fraction criterion;
    do not chase a tighter match.
  - Read a process's own pywatershed autotest before setting an
    expectation: tolerances run from 1e-13 (groundwater) to 1e-3
    (snow).
- **Where:** `CLAUDE.md` "Porting pywatershed processes";
  `docs/PORTS.md` (inventory, conventions, the snow story, domain
  facts).
- **Left unfinished:**
  - `docs/PORTS.md` "Not ported / backlog" (southern-hemisphere
    domains, nonzero `snowpack_init`, and others).

### Variants by addition: a minimal base class that variants extend

- **Status:** decided and built July 2026 (the NoDprst classes); on
  develop.
- **Why:** pywatershed derives a variant by subtraction: the parent
  is the most specific class and the child removes physics
  (`PRMSRunoffNoDprst(PRMSRunoff)`,
  `PRMSSoilzoneAg(PRMSSoilzoneAgObsET)`). What follows from that
  (analysis agreed with JLM):
  - The hierarchy points opposite to the abstraction. The shared core
    has no name: it is the flag-off path inside the largest class,
    and the simplest model is the most derived class.
  - The child redeclares nearly its whole interface anyway; only the
    kernel is really shared.
  - Init runs twice: the child reruns the parent's setup after
    `super()`, which is safe only because those steps happen to be
    repeatable.
  - Disabled physics still costs: NoDprst runoff allocates about 30
    zero arrays per step to feed the parent kernel and discards 12
    outputs.
  - The subclass does less than its parent promises, so `isinstance`
    misleads, and one choice is expressed twice (a flag and a
    subclass).
- **Decision:**
  - A variant is one of: a sibling class that shares module-level
    njit helpers (the STARFIT node types); a class composed from
    options by a factory (`make_flow_graph`); or, where a family has
    a real common core, a minimal base class that variants extend by
    adding fields.
  - The general case never depends on the specific one.
  - `DataArrayMeta` declarations inherit additively, and an inherited
    field cannot easily be removed. That is kept on purpose.
  - First use: `PRMS{Groundwater,Runoff,Soilzone}NoDprst` are the
    base classes; the full classes add the depression-storage fields
    and override init, advance and the kernel.
- **For users:**
  - Before adding a variant, compare its interface with the base: a
    strict superset extends, anything else is a sibling (next entry).
  - A variant that changes a field's kind redeclares that field; a
    variant with a budget redeclares the whole budget.
- **Where:** `docs/PORTS.md` "How variants are done here" and
  "Realized for NoDprst"; `tests/test_prms_no_dprst.py` (bit for bit
  against the full class with depression storage disabled by data);
  `tests/test_prms_no_dprst_parity.py`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - The ledger of variant cases against pywatershed's
    `DESIGN_NOTES.md` (`process_variants_design.md`, branch
    `process_variants_design`) is not on the remote. JLM wants to
    revisit variant design after the cascade ports.

### Ag variants: extend on a superset interface, sibling otherwise

- **Status:** built July 2026 (PRMSRunoffAg, PRMSSoilzoneAg,
  PRMSSoilzoneAgObsET); on develop. Restart (Aug 2026), budgets
  (Oct 2026) and kernel unit tests (Nov 2026) cover the family.
- **Why:** variants here are built by addition: a minimal base class
  that each variant extends (`docs/PORTS.md` "How variants are done
  here"). The agricultural soil zone was the first port where that
  could not be done honestly. Plain soilzone takes `infil_hru`
  (whole-hru depth); the ag soilzone takes `infil` (pervious-area
  depth) plus `infil_ag`. Inheriting would leave an `infil_hru`
  declaration that assembly demands be fed, and removing an inherited
  field is what the framework deliberately makes hard.
- **Decision:**
  - The superset rule: extend a class only when the variant's
    interface (parameters, inputs, variables) is a strict superset of
    it; otherwise write a sibling class and share code as module-level
    njit helpers.
  - `PRMSRunoffAg(PRMSRunoff)` extends: its interface only adds
    (`ag_frac` and the ag fields). The pervious geometry changes kind,
    from frozen `parameter_internal` to per-step variable, by
    redeclaring the field in the subclass (the redeclaration wins).
  - `PRMSSoilzoneAg(Process)` is a sibling of PRMSSoilzone and calls
    the same four soilzone helpers. About 50 declarations are
    duplicated: cheap, and pinned by parity tests.
  - `PRMSSoilzoneAgObsET(PRMSSoilzoneAg)` extends by addition: the
    observed-evapotranspiration iteration saves and restores state
    around the same kernel; the base class never knows iteration
    exists.
  - An option that differs between PRMS and GSFLOW is a runtime
    kernel argument, not a compile-time constant
    (`intcp_changeover_in_net_rain`: True on PRMSRunoffAg, False on
    the other runoff classes).
  - Upstream quirks are kept verbatim, including one Fortran bug
    pywatershed also keeps: the Fortran answers are the reference.
  - A PRMS "dynamic parameter" is an input and is handled as one
    (JLM, Nov 13, 2026). `ag_frac` is always `kind="input"`; whether
    it is static or dynamic is settled outside the processes, where
    a static field is repeated along time. The memory that costs is
    accepted.
  - The areas derived from `ag_frac` (`ag_area`, the pervious area
    and its fraction) are results of the current step's input: no
    restart flag, no first-step block, no state group. PRMSRunoffAg
    computes them before its kernel. This departs from pywatershed,
    which updates them after its kernel (`docs/MAINTENANCE.md`).
  - The one area state is PRMSSoilzoneAg's `ag_frac_prev`: the
    `ag_frac` its storages are distributed for. When the input
    differs from it, water moves between the ag store and
    `slow_stor`.
  - Why the areas stopped being state: both Ag classes declare
    `ag_area`, and one name on one grid is one array. In a live
    chain runoff wrote the current step's value before the soil zone
    read it as the previous step's, so the redistribution never ran
    (0 hrus against 82 standalone on the one change date of
    fgr_ag_2yr).
- **For users:**
  - `ag_frac` is a time-varying input: a static field is still
    supplied along time.
  - The five storage-change variables (`ag_soil_moist_change`,
    `ag_soil_rechr_change`, `slow_stor_change`, `soil_lower_change`,
    `soil_rechr_change`) exclude the water moved between stores when
    `ag_frac` changes. They do not match GSFLOW Fortran output on
    those dates; the budget is the check (`control["budget"] =
    "error"`).
  - The Ag classes are validated on one domain, fgr_ag_2yr (GSFLOW),
    at pywatershed's 1e-5 with per-variable exceptions.
  - Under MPI, a domain where `pref_flow_den` is nonzero on some
    ranks and all zero on others gives answers that differ from
    serial.
- **Where:** `pws_phoenix/hydrology/prms_soilzone_ag.py` (module
  docstring: the full extend-or-sibling argument and the preserved
  quirks); `PRMSRunoffAg` in `pws_phoenix/hydrology/prms_runoff.py`;
  `docs/PORTS.md` "The agricultural (Ag) ports";
  `tests/test_prms_{runoff,soilzone}_ag{,_mpi}.py`,
  `tests/test_prms_soilzone_ag_obs_et.py`,
  `tests/test_prms_runoff_soilzone_ag.py`,
  `tests/test_restart_processes_ag.py` (area timing, `ag_frac_prev`,
  the chain under dynamic `ag_frac`),
  `tests/test_process_fields.py` (the redeclaration pins).
- **Left unfinished** (moved from memory Nov 12, 2026):
  - PRMSSoilzoneNoDprstAg is not ported (pywatershed does not test
    it either).
  - `_pref_flow_flag` is `any()` over a rank's own rows; a domain
    with mixed `pref_flow_den` needs a reduction over all ranks,
    which the init hook's local-only contract does not allow.
  - PRMSRunoffAg's area timing is not checked against GSFLOW. The
    fgr_ag_2yr answers cannot tell the two orders apart: on the one
    change date (step 366, 2001-01-01) the 124 changed hrus have no
    rain, melt, infiltration or runoff. The GSFLOW Fortran source
    would settle it.
  - No test compares the live chain (PRMSRunoffAg ->
    PRMSSoilzoneAgObsET, dynamic `ag_frac`) with the GSFLOW answers;
    parity is per process, and the chain test is self-consistent.
  - `hru_frac_perv` has two writers on a shared grid, PRMSRunoffAg
    and PRMSSoilzoneAg, with a different order of subtraction. Each
    writes it before its own kernel; output shows the soil zone's.
  - The original larger Ag domains (`ucb_ag_*`, kept offline) were
    not used; they may return.

## FlowGraph

### FlowGraph: node types as data, one kernel calling each type's functions

- **Status:** designed and built July 2026 (every pywatershed node
  type); graph-building helpers Nov 2026; on develop.
- **Why:** pywatershed's FlowGraph holds one Python object per node,
  with scalar properties and method dispatch. That cannot be compiled
  by numba and breaks the memory rule. The concept is kept (different
  node types on one directed graph, such as a reservoir inserted in a
  muskingum network) and re-expressed as data.
- **Decision:**
  - A FlowGraph is a Process on its own `nnodes` grid; the grid owns
    the connectivity and the node order.
  - A node type is data: the class declares the union of all its
    types' fields as `(nnodes,)` arrays, nan where a field does not
    apply. Each node's type is an integer code in `node_type`. A code
    is a position in the factory's list of type names, so it differs
    between compositions: refer to a type by its name. The
    code-to-name table is on the class and in `node_type.attrs`
    (`node_type_names`).
  - `make_flow_graph(...)` is a class factory, composed per model.
    The Model is unchanged: a graph is one more `process_dict` entry.
  - The node-type contract: a node type supplies its `fields`, two
    numpy functions (init and advance) and three compiled functions
    that take one node (`prepare`, `substep`, `finalize`). The
    factory gathers each function from every type into a tuple. One
    njit kernel walks the nodes in order, over the substeps, and
    calls the function belonging to each node's type
    (`numba.literal_unroll` generates the selection). Adding a type
    needs no kernel edit.
  - State is passed as an argument: every type function takes
    `state`, a namedtuple of all the graph's arrays, and writes into
    those. The first idea, each type function closing over its own
    arrays, was not built: numba makes an array captured by a closure
    read-only.
  - Lateral inflows are three Map-fed inputs summed in the kernel; an
    inserted node is a zero row in the weights.
  - The node order replicates pywatershed's construction exactly: a
    different valid order changes the float accumulation downstream.
  - No MPI code: under MPI the graph's grid is replicated like the
    segment grid.
- **For users:**
  - `flow_graph_build.splice_flow_graph(...)` with `NewNode(...)`
    returns a `FlowGraphKit` (see Terms);
    `prms_translate.insert_flow_graph(kit, new_nodes)` swaps the
    channel of a `ModelKit` for a graph.
  - `n_substeps` belongs to the graph: every node shares it (24 for
    the muskingum channel).
  - An obsin node with a negative observation holds the first
    substep's inflow as its outflow for the whole day (pywatershed's
    behaviour). It is not a pass-through.
  - Say "node type", not "kind": `DataArrayMeta.kind` means something
    else.
- **Where:** `CLAUDE.md` "FlowGraph"; `pws_phoenix/flow_graph.py`;
  `pws_phoenix/flow_graph_build.py`;
  `pws_phoenix/hydrology/*_flow_node.py`; `tests/test_flow_graph*.py`;
  `tests/test_obsin_source_sink_nodes.py`.
- **Left unfinished:**
  - The dispatch uses a numba feature marked experimental (a warning
    on each compile).
  - No full comparison against a pywatershed FlowGraph run (about 8
    million pure-Python node-substep calls).
  - A kit with other segment-grid processes beside the channel
    (stream temperature) cannot take a graph yet.
  - A second dispatch was seen to be possible and not built:
    first-class function types held in a `numba.typed.List`, which
    would avoid recompiling the kernel for each composition. In the
    July 2026 spike it failed only on the read-only capture, which
    passing state as an argument removes.

### STARFIT nodes: frozen parameters, unit factors, two time modes

- **Status:** built July 2026; required initial storage Nov 2026; on
  develop.
- **Why:** the STARFIT reservoir was the first new node type written
  to the node-type contract (previous entry), and it needed no change
  to the graph kernel. pywatershed's node edits its own parameters
  and seeds its storage inside the node; here parameters freeze at
  assembly, and a graph must run in either flow unit.
- **Decision:**
  - Data preparation is outside the node: a missing mean flow or
    starting storage is the caller's to supply, and the node raises
    on its own rows.
  - Flow units (`io_in_cfs`, graph level) with no branch in compiled
    code: two per-node factors hold the conversion constants in a cfs
    graph and 1.0 in a cms graph. Multiplying by 1.0 changes no bits,
    so the validated cms path is untouched.
  - The family (plain, with source or sink, daily) shares njit
    helpers instead of pywatershed's subclassing.
  - The hourly node run with one 24-hour substep reproduces the
    STARFIT daily reference, which releases from the same day's
    inflow.
  - Daily mode (`StarfitDailyFlowNode`) is pywatershed's, verbatim:
    it releases from the previous day's mean inflow, a one-day lag.
    pywatershed has no reference values for it, so it is validated by
    parity against pywatershed's own node.
- **For users:**
  - Daily mode cannot match the STARFIT reference closely, because of
    the lag. Do not "fix" it.
  - Daily mode needs `n_substeps >= 2`.
  - `lake_storage_sub` and `source_sink_storage_min` are in millions
    of cubic meters in a cfs graph too.
  - The mixed channel and reservoir graph is tested on ucb_2yr (Big
    Sandy), not drb.
- **Where:** `CLAUDE.md` "FlowGraph"; the module docstrings of
  `pws_phoenix/hydrology/starfit_*.py` (unit factors, the family's
  shared helpers; the daily module tells the lag story in full);
  `tests/test_starfit_flow_node.py`,
  `tests/test_starfit_daily_parity.py`,
  `tests/test_mixed_channel_starfit.py`.
- **Left unfinished:**
  - No daily node with a source or sink (pywatershed has none).
  - The units of `lake_release_sub` / `lake_spill_sub` in daily mode
    are unsettled.

## IO

### Output backends differ on purpose: zarr in serial, NetCDF stream under MPI

- **Status:** decided June 2026; on develop.
- **Why:** the best output backend depends on the context, and
  forcing one writer on both paths would give up the better choice in
  one of them.
- **Decision:**
  - Serial writes one zarr store; the MPI distributed grid writes
    through mpixarray's NetCDF stream. They are not unified behind a
    shared writer.
  - The backend follows the grid's context: under MPI a serial
    (replicated) grid still writes zarr, from rank 0.
- **For users:**
  - The difference matters only when separating compute time from
    output time in a scaling study.
- **Where:** `CLAUDE.md` "MPI path";
  `Output` in `pws_phoenix/data_io.py`; `ModelMPI` in
  `pws_phoenix/model.py`.
- **Left unfinished:**
  - `Output` handles variables with one spatial dimension only; it
    fails loudly on others.
  - mpixarray's NetCDF writer re-emits a `_FillValue` inherited on a
    coordinate (warnings only; not worked around here, to be reported
    upstream).

### Serial output: a pre-sized zarr store written by region

- **Status:** built Oct 30, 2026; on develop.
- **Why:** each `to_zarr(append_dim="time")` spent about 2 s
  re-validating the store, whatever the data size: 26 s of a 32.6 s
  ucb_2yr run.
- **Decision:** `Output` creates the store once at the full time
  length (coordinates through xarray, arrays declared through zarr,
  nothing materialized), writes each full buffer as a region, and at
  `finalize` writes the tail and resizes time to the steps written.
  The same run takes 7.0 s.
- **For users:**
  - Call `finalize()`: without it the last partial chunk is never
    written.
  - The path in `control["output_serial_zarr"]` is used verbatim.
- **Where:** `Output` in `pws_phoenix/data_io.py`; `CLAUDE.md` "Serial
  path"; `benchmarks/bench_output.py`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - `Model.run()` after a restart: `Output` labels rows from
    `time_values[0]` whatever the start index.

## Checks

### Budgets as data in the owning Process

- **Status:** built Oct 20 - Nov 5, 2026; on develop (PR 2).
- **Why:** pywatershed checks conservation with a Budget object held by
  a ConservativeProcess. Here the same statement is data on the grid
  dataset, declared beside the physics it constrains (JLM: "the
  conservation statement next to the physics"), computed in one njit
  pass with no per-step allocation.
- **Decision:**
  - The Process owns the declaration (`budgets = {...}`), the generated
    fields, the kernel and the local count of imbalanced locations.
  - The Model owns when (right after each `calculate`) and the error
    mode. Warn is local to a rank; error is collective
    (`allreduce(MAX)` of the local count, so every rank raises
    together). Serial is the one-rank case of the same method.
  - One accumulated residual per budget (`residual_accum`, restarted).
  - The optional `exchanges` term (signed fluxes) is kept; the STARFIT
    sink/source is the case that needs it.
  - Budget terms are their own kind, `budget_variable`: the metadata
    itself says what the field is, so "all variables" means physics.
  - Default `budget_rtol` / `budget_atol` are 1e-10; every port closes
    at 1e-12 except the STARFIT family, whose storage term is a
    difference of large reservoir storages.
  - `gwres_sink` is an output of the groundwater budget although
    pywatershed's term list omits it (the kernel subtracts it from
    storage).
  - PRMSChannel is not budgeted: muskingum has no storage state and
    `seg_stor_change` is defined as the imbalance, so a per-segment
    budget is an identity.
- **For users:**
  - Budgets are computed and checked only by the Model. A process
    driven by hand (the pywatershed habit) gets no budget; run it as a
    one-process Model.
  - `control["budget"]`: `None` computes and writes the terms without
    judging them; `"warn"` reports per rank; `"error"` raises on every
    rank.
- **Where:** `pws_phoenix/budget.py`; `Process.budgets` in
  `process.py`; `Model._check_budgets`, `_raise_if_imbalanced`;
  `tests/test_budget.py`, `tests/test_budget_mpi.py`; `CLAUDE.md`
  "Budgets (IMPLEMENTED Oct 2026)"; the channel algebra is in the
  `prms_channel.py` docstring.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - Reverting the snow departure to prove detection needs a seam that
    survives compilation: `PRMSSnow._calculate` binds its helper at
    first compile, so a monkeypatch depends on test order.
  - The one-process-Model rule has no user-facing doc to live in yet.

### Units mode: closed vocabulary, two boundaries, kernel tests

- **Status:** vocabulary, declarations and both boundary checks built
  Nov 5-12, 2026 (PRs 5 and 6 on develop; the state check is commit
  684e2ae on feat_units). Kernel tests exist for the groundwater,
  canopy, channel, soilzone, runoff and Ag families only.
- **Why:** a wrong unit is silent in three places: in what a user
  supplies, in what PRMS's own metadata says about a parameter, and
  inside a kernel (a bare `* 60.0 * 60.0`). JLM wanted all three
  checked, at no per-step cost and without touching the arrays.
- **Decision:**
  - Every float `DataArrayMeta` declares `units=` from a closed
    vocabulary (`units.py` `VOCABULARY`: key -> pint expression, `None`
    where pint cannot express it). The declarations are the metadata;
    pywatershed's metadata yaml is not copied in.
  - pint sits behind the table and touches strings only. Arrays stay
    plain numpy: a Quantity inside a DataArray would break the
    one-ndarray-per-name rule and numba. pint is never imported in the
    default mode.
  - Conventions: `fraction` is a proportion (specific gravities
    included); `none` is a dimensionless flag stored as float;
    temperature differences are `delta_degF` / `delta_degC`; rate
    coefficients keep their physical `1/day`. FlowGraph fields in the
    graph's io units declare placeholders resolved at composition.
  - PRMS boundary (`prms_translate/prms_units.py`): PRMS's metadata
    words are translated to vocabulary keys, with a by-name table where
    PRMS is wrong or vague. A declared-vs-translated disagreement
    raises: it is a table or declaration bug, never data.
  - Model boundary (`control["units"]`, judged at build): `None`,
    `"check"` or `"convert"`. A missing units attribute is an error;
    `control["units_patch"]` is the escape hatch.
  - Where "convert" converts follows from who owns the array. A
    parameter is the model's working buffer, so it is converted in
    place once. Whatever the model copies is converted in the model's
    copy and the supplied data are never edited: supplied initial
    state, the declared variables of a restart file (a netCDF file
    anyone can write), and every input.
  - An input is converted in its one-step buffer after each refill, in
    memory, file-backed and streamed alike (changed Nov 2026). Before,
    an in-memory input was converted in place and a streamed one
    raised ("conversion is never a per-step edit"). The model already
    copies one row per step, so converting the copy costs one multiply
    over the row; it lets forcing that never went through PRMS (mm,
    degC) be used as it is; and it makes a float32 file give the same
    answers file-backed and loaded, because the row is widened to
    float64 before it is converted.
  - Kernels: a test runs the kernel's undecorated body on pint Quantity
    arrays built from the declarations and compares magnitudes bit for
    bit with the compiled run. pint alone misses a compatible unit at
    the wrong scale; the magnitude comparison catches it. Every line
    must execute, or carry `# units-test: unreachable (<reason>)`.
  - Kernel convention that follows: a zero or threshold of a working
    unit is a named module float (`_ZERO_INCHES`, `_NEARZERO_ACRES`).
  - Budget unit agreement stays string equality, which is exact with a
    closed vocabulary.
- **For users:**
  - Nothing is checked unless `control["units"]` is set.
  - Under `"check"` every supplied array needs `attrs["units"]` naming
    a vocabulary key. A bare ndarray of initial state raises; name its
    units in `control["units_patch"]`.
  - `"convert"` never edits a supplied input, state array or forcing
    file; it edits a supplied parameter array in place.
  - A PRMS file with metric precipitation raises: the kernels are
    imperial.
  - MPI restart files written before Nov 12, 2026 carry no units and
    fail the check.
  - A new float field needs `units=`; a new unit is a new key in
    `VOCABULARY`.
- **Where:** `pws_phoenix/units.py`;
  `pws_phoenix/prms_translate/prms_units.py`; `tests/test_units.py`,
  `tests/test_units_check.py`, `tests/test_units_check_mpi.py`,
  `tests/test_prms_units.py`, `tests/test_kernel_units.py` and its
  harness `tests/kernel_units.py`; `CLAUDE.md` "Units mode".
- **Left unfinished** (moved from memory Nov 12, 2026):
  - No kernel test yet for: snow and atmosphere (waiting on how offset
    temperatures are handled inside a kernel), the stream-temperature
    group, `flow_graph.py` and the seven node types, the `budget.py`
    and `maps.py` kernels, `compute_soltabs`, and every init derivation
    except the Muskingum one.
  - Not checked at the Model boundary: stream temperature's undeclared
    restart arrays, `NewNode` values (bare floats), Map weights.
  - Not looked at: whether the zarr and NetCDF writers stamp units on
    output variables.

### Tests: no skips, pinned answer keys

- **Status:** adopted Oct 2026; on develop.
- **Why:** a skipped test shrinks the suite without anyone noticing:
  a machine without the answer data looked green. And answer data
  that is regenerated can shift results silently.
- **Decision:**
  - `tests/run_tests.sh` passes `--error-for-skips` to every pytest
    call: missing data fails the sweep.
  - A parametrized case that does not apply asserts something true of
    that case instead of skipping.
  - The answer keys are pinned in `tests/prms_oracle.toml` and
    checked at session start. Regenerated answers mean a reviewed
    change of key.
  - Tests load a netCDF file and close it (`xr.load_dataset`). With
    more than 128 files left open, xarray's file cache thrashes and
    the suite appears to hang.
- **For users:**
  - Run `./tests/run_tests.sh` from the repo root (`--serial-only`,
    `--parallel-only` for one half).
  - `PWS_TEST_DATA` points the tests at another data root.
- **Where:** `CLAUDE.md` "Environment (tests and tooling)";
  `tests/run_tests.sh`; `tests/conftest.py`;
  `tests/prms_oracle.toml`.
- **Left unfinished:**
  - Keys are pinned for macos-arm64 only.
  - CI cannot fetch the answer data yet.
  - `examples/*.py` are in no automated test.

## Performance

### Benchmarks: ASV in the existing environment

- **Status:** built Oct 26-27, 2026; on develop. Run locally only.
- **Why:** the performance goal needs a number per process and per
  commit, taken in the environment the tests run in, so that two
  commits differ only in code.
- **Decision:**
  - ASV's existing-environment mode, not an environment ASV builds
    (JLM; xarray's `asv_bench` does the same). A comparison is base
    against head in one environment.
  - That mode saves results only with `--set-commit-hash` and refuses
    commit ranges, so `benchmarks/asv_run.sh` checks out each hash in
    the editable install and records `conda list --export` beside the
    results.
  - ASV re-runs `setup` before every timed call, so a model is built
    once per ASV subprocess (`_common.get_model()`), never in `setup`.
  - Per-step allocation is measured with tracemalloc after five warm
    steps; peak memory is too noisy (about 50 MB) to show it.
  - `tests/test_benchmark_coverage.py` fails when a process class or
    node type has no benchmark.
- **For users:**
  - Run benchmarks through `benchmarks/asv_run.sh`, from a clean tree.
  - To try uncommitted benchmarks add `--dry-run`; without it the
    results are filed under a commit that lacks those benchmarks.
  - Results are per clone and per machine (`.asv/` is git-ignored);
    keep one benchmarking environment per machine.
- **Where:** `benchmarks/README.md`; `asv.conf.json`; `benchmarks/`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - No benchmark job in CI: it waits on test data being fetchable
    there and on the public release of mpixarray and mpix.
  - Undecided: commit `.asv/results/` and publish the html.
  - A FlowGraph class composed of all seven node types costs about
    12 ns per node-substep more than a channel-only class, even when
    every node is a channel. Not pursued (0.13 ms per step on drb).

### Per-step data access: hold the ndarray, not the Dataset lookup

- **Status:** built Oct 27, 2026; on develop.
- **Why:** the benchmarks showed the physics was 0.5-18% of a process
  step. The rest was `self._obj[name].values`: `xr.Dataset.__getitem__`
  builds a DataArray on every call (8-20 us each).
- **Decision:**
  - `Process._refs`: a process captures its fields' ndarrays once at
    bind and reads `refs[name]` in every step method.
  - The same rule in the framework: a Map binds its arrays and applies
    sparse weights in numba, an Input holds ndarrays, and `Time`
    derives its calendar fields once per `set_index`.
  - This relies on one ndarray per name for the whole run. Serial has
    it by construction; under MPI ModelMPI refills its step buffers in
    place. A test pins it.
  - Stream shade geometry is tabulated per (day of year, segment) at
    init, and one trig chain is reduced algebraically: the one place a
    kernel departs from verbatim for speed (agrees to rounding).
  - Result on the drb chain: 9.80 ms to 0.40 ms per step.
- **For users:**
  - In a step method read `refs[name]`, never `self._obj[name].values`.
  - Write into a field's array in place; never replace the array.
  - Tried, no gain (do not retry without a new reason):
    - One njit kernel per process for the multi-kernel processes
      (atmosphere, stream temperature, RunoffAg, SoilzoneAg): the
      Python between kernels is about 2 us; only AgObsET's loop
      (about 21 us) is real.
    - Moving the obs-ET iteration loop into numba (reverted Oct 27,
      2026).
    - A namedtuple of arrays as one kernel argument: numba types a
      tuple element by element, so the per-argument cost stays.
- **Where:** `DEVELOPER.md` "Per-step data access"; `CLAUDE.md` "Prime
  directive: memory" item 4; `benchmarks/README.md` section 5 (the
  numbers);
  `tests/test_two_grid_mpi.py::test_step_buffers_persist_across_rebinds`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - A numba call costs by argument count (10-15 us for snow's about
    80 arguments): one struct of references per process would save an
    estimated 0.1-0.15 ms of the 0.36 ms drb step.
  - `Output` still makes one DataArray call per output variable per
    step.

### Numba: cache on by default, threaded kernels held to 1 thread

- **Status:** cache built Oct 28, 2026; threading Oct 30 - Nov 2,
  2026; on develop. The thread count moved to the control Nov 19,
  2026.
- **Why:** the first step of a run was about 4 s of compilation. And
  the hru loop of every hru-grid kernel is independent per element.
- **Decision:**
  - `PWS_NUMBA_CACHE`, default on; the tests force it off. A kernel
    opts in only when every jitted function and constant it uses is
    in its own file, because numba's cache does not see a change in
    another file.
  - Threading: `prange` on the one per-element loop of each hru-grid
    kernel, ALWAYS compiled threaded (`parallel=True`); the Model
    holds numba at `control["numba_n_threads"]` threads, default 1.
    Results are bit for bit the same at any count, because nothing
    crosses elements.
  - Changed Nov 19, 2026 from a switch read at import
    (`PWS_NUMBA_PARALLEL`, default off, which chose between two
    kernels): numba's cache does not tell a threaded kernel from an
    unthreaded one, so a threaded run loaded the unthreaded kernels
    an earlier run had saved. One kind of kernel removes the case,
    and a threaded kernel held to 1 thread measured no slower. The
    cost is a compile about twice as long.
  - The count is a control key because every run has a control, and
    `numba.set_num_threads` must be called after numba is imported:
    the `NUMBA_NUM_THREADS` environment variable is read at numba's
    import, before any control exists, and is only the maximum.
  - No warning on the default of 1 (every build would warn). The
    build warns only when `NUMBA_NUM_THREADS` is set above 1 and the
    control key is absent: the one case where the user asked for
    threads and does not get them.
  - Not threaded: light helper loops (a thread launch costs more),
    loops that carry a counter, the segment and FlowGraph kernels
    (sequential order), budget sums (float order).
- **For users:**
  - `PWS_NUMBA_CACHE` is an environment variable read at import;
    the thread count is `control["numba_n_threads"]`.
  - Under `mpirun` the count is per rank: ranks x threads must fit
    the machine.
  - A count above numba's maximum for the process raises at build
    (collectively under MPI).
- **Where:** `DEVELOPER.md` "Numba cache" and "Numba threading";
  `README.md` "Run-time switches".
- **Left unfinished** (moved from memory Nov 12, 2026):
  - The threading payoff is an estimate (break-even at 100-500 hrus);
    the ucb_2yr sweep has not been run.

## Translation and migration

### PRMS translation: one way, pyPRMS only, outside the core

- **Status:** built July-Aug 2026; on develop.
- **Why:** tests and examples depended on files pywatershed had made
  (per-process parameter files, converted forcings) and on test-side
  workarounds. A user holding a PRMS control file should get a model
  in one call.
- **Decision:**
  - One way: PRMS's own files (control, parameter, forcing, dynamic
    parameter) to pws_phoenix inputs, never back.
  - A package inside the repo (`prms_translate/`) that the core never
    imports, so it can be split out later. Hence the entry point is
    `model_from_control(control_file)`, not a `Model` method.
  - Files are decoded with pyPRMS only, never pywatershed, which is
    the test oracle. The generic parts are meant to move into pyPRMS;
    each workaround for pyPRMS sits in `readers.py` and is tied to an
    upstream ask.
  - The control file is the only input. Its module names and flags
    select the classes; an unported choice raises.
  - Parameters are packaged from the class declarations, with no
    hand-kept lists.
  - Names translate, paths never: PRMS's output variable names are
    carried over, its output paths are not (the drb control's path
    points into the answers directory).
  - `PET_cbh_file` is not consumed: potential evapotranspiration is
    computed, and that file is used on the PRMS side only.
  - Two layers: `assemble_from_control` returns a `ModelKit` to
    inspect or change; `model_from_control` is the one call.
    `mpi_model_from_control` is the same under `mpirun`.
- **For users:**
  - `model_from_control(control_file)`; use `assemble_from_control`
    when the kit must be changed before `.model()`.
  - PRMS's requested output names are filtered to the model's
    variables; the rest are in `kit.dropped_output_var_names`.
  - A PRMS binary restart (`init_vars_from_file=1`) raises.
  - Under MPI, hru-grid results match serial bit for bit;
    segment-grid results match to about 1e-9 (the Map's sum order).
- **Where:** `pws_phoenix/prms_translate/` (package docstring);
  `docs/UPSTREAM_ASKS.md` (the asks and which workaround each one
  retires); `docs/PORTS.md`; `examples/01_prms_legacy_translation.py`;
  `tests/test_prms_translate*.py`.
- **Left unfinished** (moved from memory Nov 12, 2026):
  - The pyPRMS asks are drafted, with branches, but not posted.
  - The constant-shade and segment-humidity stream temperature
    classes are selected from the control, but assembly raises.
  - Not started: streamflow data files (`sf_data`), writing a kit to
    yaml.

### Pre-processing: derived inputs in a stamped file, verified at use

- **Status:** built Aug 2026 (commit 7e52b73); on develop.
- **Why:** some required inputs are derived from others (solar
  tables, segment order, `hru_in_to_cf`, Map weights). Deriving them
  at every build is simplest; a large domain wants to do it once. A
  file derived once can silently stop matching its inputs.
- **Decision:**
  - Deriving at build stays the default; the file is opt-in.
  - `write_preprocessed(control, path)` writes every derivation of a
    configuration to one netCDF file. Each variable carries a digest
    of each input it was derived from, and of itself.
  - `verify_preprocessed` checks the digests at use. A stale or
    edited variable raises, naming the variable, the input and the
    remedy.
  - A weight matrix is stored as its nonzero entries (Nov 2026; see
    "Maps"): three variables, each with the stamp. A file from before
    that, with a full matrix, raises on read and is written again.
  - `derive_preprocessed(params, stream_temp)` does the deriving, from
    parameters already in memory; `write_preprocessed` and
    `translate_domain` both call it.
- **For users:**
  - `model_from_control(control_file, preprocessed=path)`.
  - Rewrite the file after changing any parameter a derivation reads.
  - A domain translated ahead of time gets this file from
    `translate_domain` (the next entry).
- **Where:** `pws_phoenix/prms_translate/preprocess.py`;
  `examples/02_preprocessing.py`; `tests/test_preprocess.py`.
- **Left unfinished** (moved from memory Nov 12, 2026): none.

### A domain translated ahead of time: files written once, options from the control copy

- **Status:** built Nov 2026; the serial run entry only.
- **Why:** `model_from_control` parses the CBH text and the parameter
  file at every start, and under MPI every rank does. At NHM scale
  (114,958 hrus, 47 years) one CBH file is 9.87 GB of text and takes
  6 to 7 minutes and about 100 GB of memory to parse. The files do
  not change between runs.
- **Decision:**
  - `translate_domain(control_file, out_dir)` writes the files a
    control names ONCE, into a flat domain directory: one combined
    forcing netCDF (`cbh.nc`: the whole record, float32 where that is
    exact, no compression, no chunking), `<stem>_parameters.nc`,
    `<stem>_preprocessed.nc`, a byte copy of the control file, and
    `<stem>_translation.yaml`.
  - `<stem>_preprocessed.nc` (the previous entry's file) is derived
    from `<stem>_parameters.nc`, not from the parameter text: a run
    verifies its stamps against that file's arrays. It is written
    again whenever the parameter file is written. A run whose yaml
    names none derives at build, as `model_from_control` does.
  - The yaml records FILES only: the products relative to its own
    directory, the files that were read as absolute paths, and a
    sha256 of the control file. OPTIONS (classes, the simulation
    window, output names) are read from the control copy at every
    run, so an option supported later needs no new translation. The
    copy's own file entries are never resolved: they are relative to
    the original's directory.
  - An existing product is reused when it was written from the files
    the control names, and raises when it was written from others.
    Controls that name the same CBH files share one forcing file.
  - A run takes one argument, the yaml:
    `model_from_translation(yaml)`, over the kit from
    `assemble_from_translation(yaml)`. Both share
    `assemble_from_control`'s assembly; only where the parameters and
    the forcings are read from differs. PRMS logic and pyPRMS still
    run at every start (the control file is parsed).
  - The forcing file belongs to the CBH files, not to one control, so
    it holds the whole record. The core gained two things for that,
    both also needed by a native model yaml:
    `control["start_time"]` / `control["end_time"]`, and an input
    given as the path of a file that holds several inputs.
  - A file-backed input stays on disk, one time row read per step,
    and its per-step buffer is float64 whatever the file holds.
    Before this, a read-only file-backed input was read whole at
    build, by the line that set its read-only flag.
- **For users:**
  - `python -m pws_phoenix.prms_translate.translate_domain
    CONTROL_FILE --out DOMAIN_DIR`, then
    `model_from_translation(DOMAIN_DIR / "<stem>_translation.yaml",
    control={...})`.
  - A `start_time` / `end_time` in the caller's control takes
    precedence over the control file's.
  - `control["load_all"] = True` reads the forcing into memory at
    build. With a window it still reads the whole file.
  - `control["units"] = "convert"` converts a forcing file in other
    units as it is read, file-backed or loaded.
  - A control with dynamic-parameter files raises. An edited control
    copy raises: translate the domain again.
  - `--only parameters --force` rewrites the parameter file and
    leaves the preprocessed file as it was. The next run then raises
    "STALE": run `--only preprocessed --force`.
- **Where:** `pws_phoenix/prms_translate/translate_domain.py` (module
  docstring), `cbh.write_cbh_netcdf`,
  `preprocess.derive_preprocessed`,
  `assemble.assemble_from_translation`; `Model._time_window`;
  `tests/test_translate_domain.py`, `tests/test_cbh_netcdf.py`,
  `tests/test_model_from_translation.py`, `tests/test_time_window.py`,
  `tests/test_input.py`.
- **Left unfinished:**
  - `translate_domain` does not verify an existing preprocessed file
    against the parameter file (the run does).
  - For a control with stream temperature the preprocessed step
    builds four full weight matrices in memory (see "Maps", left
    unfinished): not possible at NHM scale.
  - The MPI run entry. `ModelMPI` reads one combined file that also
    holds the hru parameters; whether translation writes such a file
    or `ModelMPI` takes forcing and parameters separately is
    undecided. `ModelMPI` raises on the two window keys.
  - Reads of several days at a time, and the chunk shape of the
    forcing file.
  - Translation that writes a native model yaml, so that a run needs
    no PRMS logic.
