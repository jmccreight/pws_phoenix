# April 6, 2026: Design Summary: `incarnations/xr` Prototype Modeling Framework

## Overview

This document summarizes the design principles of the prototype process-based
modeling framework in `base.py`, as illustrated by the toy `Upper`/`Lower`
model in `processes.py` and its regression test in
`tests/test_up_low_regression.py`. It also compares and contrasts this design
with two established frameworks in the geoscience Python ecosystem:
[xarray-simlab](https://xarray-simlab.readthedocs.io) and
[Landlab](https://landlab.readthedocs.io).

---

## Core Design Principles

### 1. Explicit Declarative Interfaces via Static Methods

Each `Process` subclass declares its data needs through a small set of static
methods:

| Method                     | Role                                                        |
| -------------------------- | ----------------------------------------------------------- |
| `get_parameters()`         | Names of static config variables from a parameter `Dataset` |
| `get_inputs()`             | Names of read-only, time-varying external inputs            |
| `get_mutable_inputs()`     | Names of read-write, time-varying external inputs           |
| `get_variables()`          | Public state variables (shareable downstream) with metadata |
| `_get_private_variables()` | Internal scratch variables, not exposed outside the process |

This is a lightweight, introspectable contract system. The framework calls
these methods at initialization time to wire everything together automatically.
The distinction between `get_inputs()` (immutable) and `get_mutable_inputs()`
(mutable) is a good instinct toward enforcing data integrity — it mirrors the
`intent(in)` vs `intent(inout)` distinction in Fortran-style scientific code.

One opinion worth stating: encoding the schema in static methods returning
plain dicts and tuples is pragmatic, but as the framework matures it may be
worth moving toward a more structured metadata object (a dataclass, or even
attrs/pydantic) to give better IDE support and validation. The current
`Dict[str, Any]` for variable metadata is already flagged in a TODO.

### 2. Everything Is xarray; Runtime Is NumPy

xarray `Dataset` and `DataArray` objects serve as the typed, labeled,
metadata-rich containers throughout initialization. Each `Process` holds
`self.data`, an `xr.Dataset`, as its internal store. This gives every variable
a home with attached dimension names, coordinates, and attributes essentially
for free.

But during the simulation loop — `advance()` then `calculate()` — the code
does in-place NumPy operations directly on `.values`. No xarray overhead, no
label alignment, no copy. This is the right approach: use xarray for what it
is good at (metadata, I/O, alignment), and use NumPy for what it is good at
(fast, in-place numerical computation). The `Output` class follows the same
philosophy, writing raw NumPy buffers to NetCDF via `netCDF4` rather than
routing through xarray at write time.

### 3. NumPy Reference Semantics as Pointers — The Central Design Innovation

This is the most distinctive and intellectually interesting aspect of the
framework. It is worth unpacking all three patterns in which it appears.

#### Pattern A: Forcing propagation through `Input._current_values`

`Input.__init__` allocates `_current_values` once:

```python
self._current_values = np.nan * self.data[0, :]
```

`Input.advance()` updates it strictly in-place:

```python
self._current_values[:] = self.data[self._current_index, :]
```

The `[:]` slice assignment overwrites the underlying NumPy buffer without
creating a new array object. When the framework later does
`self[ii] = kwargs[ii].current_values` inside `Process.__init__`, and xarray
preserves the buffer identity through the Dataset assignment (verified by the
`assert id(...)` check), every downstream `Process` that holds a reference to
that DataArray sees the updated forcing values each timestep — with zero
additional copies and zero re-assignment in the run loop. This is pointer
semantics achieved in pure Python/NumPy.

The test asserts this explicitly:

```python
assert model.model_dict["upper"]["forcing_common"].values is (
    model.model_dict["lower"]["forcing_common"].values
)
```

#### Pattern B: Shared parameters via `.values =` reassignment

`Process.__init__` starts with:

```python
self.data = parameters[list(self.get_parameters())]
```

This creates a new Dataset — and xarray copies the data backing in the
process. A naive `self[pp] = parameters[pp]` (whole-DataArray assignment into
the Dataset) would replace the DataArray but may not preserve the original
buffer. Instead the code does:

```python
self[pp].values = parameters[pp].values
```

This swaps in the original NumPy array into the DataArray that already lives
in `self.data`. Under the hood, `DataArray.values =` calls
`Variable._data = as_compatible_data(value)`, which stores the array object
directly — no copy — so the `is` identity is preserved. The commented-out
assert (`# assert self[pp].values is parameters[pp].values`) tells the story.

This is what the code correctly calls "the most mystifying" usage. It works,
but it is also genuinely fragile: it depends on xarray's internal
implementation of `Variable._data` not inserting a copy. This is the one place
in the design where I would flag a real risk: a future xarray version could
silently break this invariant, and the only protection is that assertion (which
is commented out). My strong recommendation is to un-comment or re-enable those
assertions — at minimum in a debug/test mode — so that any breakage is caught
immediately rather than producing wrong answers silently.

#### Pattern C: Inter-process variable sharing

When `Lower` declares `"flow"` in its `get_inputs()`, and `Upper` has `"flow"`
in its `get_variables()`, the framework does this in
`_initialize_inputs_and_proceses`:

```python
init_dict[ii] = self.model_dict[pp][ii]  # gets Upper's 'flow' DataArray
```

This `DataArray` is then stored inside `Lower`'s Dataset. If the buffer
identity is preserved, `Upper.calculate()` writing into `self["flow"]` in-place
is immediately visible to `Lower.calculate()` reading `self["flow"]` — without
any explicit communication step. The test asserts this:

```python
assert model.model_dict["upper"]["flow"].values is (
    model.model_dict["lower"]["flow"].values
)
```

This is an elegant zero-copy inter-process communication mechanism. The
ordering guarantee (all `advance()` calls, then all `calculate()` calls, in
dict order) is what makes this safe.

### 4. Ordered Dict as an Implicit Topological Sort

`get_preceeding_processes` simply returns all process names that appear before
the target in the dict:

```python
for pp in self._process_dict:
    if proc_name != pp:
        preceeding_procs.append(pp)
    else:
        return preceeding_procs
```

This means the user specifies the execution order by the order they insert
processes into `process_dict`. Python 3.7+ guarantees dict insertion order, so
this is reliable. It is also admirably simple.

The honest tradeoff: this approach works perfectly for linear chains (A → B →
C) but becomes ambiguous or incorrect for diamond dependencies (A → B, A → C,
B+C → D) or any graph where two processes are peers that both feed a third.
For the hydrology use case, where process chains are often genuinely linear
(forcing → upper zone → lower zone → routing), this is probably fine for a
long time. But the design should be documented clearly as "linear chain only"
so users do not accidentally construct a dict whose order implies a wrong
dependency graph.

### 5. File Deduplication via Path Counting

`_load_shared_data_files` uses a `Counter` over all `pl.Path` values in
`process_dict` to find paths that appear more than once, opens them once, and
returns a lookup dict. `_load_paths_to_data` then replaces all occurrences of
that path with the single opened object. This ensures that a parameter file
shared between `Upper` and `Lower` is opened exactly once and both processes
receive references to the same in-memory object — which then participates in
the reference-sharing mechanism above.

This is a clean solution. The only subtlety is that it operates on path
identity (`pl.Path` equality), so two `Path` objects pointing to the same file
via different relative paths would be opened twice. Normalizing to absolute
paths (`.resolve()`) before counting would close that gap.

### 6. Strict Separation of I/O from Computation

All file opening happens at `__init__` time. The `run()` loop is pure
in-memory computation. `Output` buffers writes into NumPy arrays and flushes
to NetCDF in chunks of `time_chunk_size` steps. This separation is
architecturally sound: it makes performance profiling clean (I/O cost shows up
in initialization, not inside the time loop), and it makes the model easy to
test with entirely in-memory fixtures — which is exactly what the regression
test exploits with its `memory` vs `file` parameterization.

The `Output` class writes via `netCDF4` directly rather than xarray. This is
a pragmatic and correct choice for append-mode chunked writing; xarray's
`to_netcdf` is not well-suited to incremental appending.

### 7. Context Manager Protocol

`Model` implements `__enter__` / `__exit__` for RAII-style cleanup. File
handles opened at initialization are closed in `finalize()`, which `__exit__`
calls unconditionally. This is the right pattern for a class that owns
external resources. The `_finalized` guard prevents double-close and
double-run bugs. Good defensive engineering.

---

## How Variables Come from Disk

The pipeline is:

1. User supplies `process_dict` with values that are either in-memory
   (`xr.DataArray`, `xr.Dataset`) or on-disk (`pl.Path`).
2. `_load_shared_data_files` counts path occurrences and opens shared files
   once via `open_xr()`, which auto-selects `Dataset` vs `DataArray` by
   variable count.
3. `_load_paths_to_data` opens any remaining single-use paths.
4. All opened objects are tracked in `_opened_files` for cleanup.
5. Time-varying inputs (forcings, ICs) become `Input` objects, which hold the
   xarray object and expose a `current_values` DataArray that is updated
   in-place each timestep.
6. Parameters become slices of the process's internal `self.data` Dataset,
   with their numpy buffers replaced to preserve reference identity.
7. By default, data is accessed lazily from disk (only the current time slice
   is pulled during `advance()`). The `load_all` flag trades memory for speed
   by loading everything upfront.

The `Input` class is the key mediator: it decouples "where does the data live"
(file or memory) from "how does the process see it" (always a DataArray with
a stable numpy buffer that gets updated in-place).

---

## How Variables Are Shared Among Processes

Sharing works through three channels, all unified by the same numpy reference
principle:

| What is shared   | Mechanism                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------------- |
| Parameters       | Same `pl.Path` opened once → same `Dataset` → `.values =` buffer swap into each process                         |
| External forcing | Single `Input` object in `inputs_dict` → same `_current_values` DataArray → in-place `[:]` update each step     |
| Process outputs  | Downstream process receives the upstream's `DataArray` directly → same numpy buffer → in-place writes propagate |

No message passing. No copy. No explicit "push" step. If a buffer is modified
in-place by its owner, every holder of a reference to that buffer sees the
change. The framework's ordering guarantee (advance all inputs, advance all
processes, calculate all processes — in dict order) is what makes this
deterministic.

---

## Comparison with xarray-simlab

[xarray-simlab](https://xarray-simlab.readthedocs.io) is a framework designed
explicitly around xarray for process-based modeling. Key similarities and
differences:

### Variable Declaration

xarray-simlab uses class-level variable declarations with decorators and intent
annotations (inspired by attrs):

```python
class SomeProcess:
    slope = xs.variable(dims='x', intent='inout')
    upstream_var = xs.foreign(OtherProcess, 'some_var', intent='in')
```

This code uses static methods returning dicts:

```python
@staticmethod
def get_variables():
    return {"flow": {"dims": ("space",), "dtype": np.float64, ...}}
```

xarray-simlab's approach gives better IDE support and is more declarative.
This framework's approach is more runtime-flexible (the dict can be constructed
dynamically) but harder to inspect statically. xarray-simlab's `xs.foreign`
is the direct equivalent of this framework's inter-process input wiring — but
it is explicit and type-checked at model-build time, rather than resolved by
name-matching at runtime.

### Dependency Resolution

xarray-simlab builds an explicit directed acyclic graph (DAG) from `intent`
annotations and `xs.foreign` references. Execution order is topologically
sorted. This framework relies on dict insertion order. For linear chains,
both approaches give the same result. For anything more complex, xarray-simlab
is safer.

### Data Sharing

xarray-simlab does not use numpy reference semantics for inter-process sharing.
Processes access foreign variables by routing through the framework's
state store at each step, which typically involves more copying. This
framework's pointer-style sharing is more aggressive and more performant —
at the cost of being harder to reason about and more dependent on internal
xarray/numpy behavior.

### I/O Interface

xarray-simlab's primary user interface is an `xr.Dataset` that is passed to a
`run` method, with simulation results returned as a new Dataset. This is very
xarray-native and composable with the broader ecosystem. This framework uses
plain dicts for configuration and writes output separately to NetCDF. The
xarray-simlab model is more ergonomic for interactive/notebook use. This
framework's model is more explicit about memory layout and I/O costs, which
matters more for long operational runs.

### Overall

xarray-simlab is more polished, more compositional, and more integrated with
the xarray ecosystem. This framework makes more deliberate choices around
performance (numpy refs, chunked I/O, lazy loading) and is arguably more
transparent about what is happening in memory at any given moment. They are
optimizing for different things.

---

## Comparison with Landlab

[Landlab](https://landlab.readthedocs.io) is a toolkit for 2D earth-surface
process modeling centered on a computational grid. Key similarities and
differences:

### The Shared Namespace Model

Landlab's central idea is that all components operate on a shared `ModelGrid`
object. Fields (e.g., `'topographic__elevation'`) are stored on the grid at
node/link/face locations:

```python
grid.at_node['topographic__elevation'] = ...
component_a.run_one_step(dt)
component_b.run_one_step(dt)
```

Components read and write to the grid's field dictionary. This is a
**shared global namespace** pattern — effectively a message bus backed by a
dict of NumPy arrays. Sharing is implicit: if two components use the same
field name string, they share data.

This framework shares data via numpy buffer references, which is more explicit
(the wiring is resolved at init time and asserted) but also harder to see
in a high-level model configuration. Landlab's approach is arguably more
transparent at the "script" level; this framework's approach is more
transparent at the "framework internals" level.

### Grid-Centricity vs. Dimension-Agnosticism

Landlab is deeply grid-centric. Its components are designed around 2D spatial
topology. This framework has no notion of a grid — space is just a dimension
in an xarray object. This makes it more flexible for problems where the
"spatial" dimension is actually HRUs, catchments, or some other abstract index.
For gridded 2D PDEs, Landlab has far more infrastructure. For irregular or
abstract spatial structures, this framework is more natural.

### Component Interface

Landlab components declare `_input_var_names` and `_output_var_names` as
class-level attributes (lists of strings). This is similar in spirit to
`get_inputs()` and `get_variables()` here. Landlab components expose a
`run_one_step(dt)` method. This framework separates `advance()` (state
bookkeeping) from `calculate(dt)` (physics), which is a cleaner distinction
when advance semantics are non-trivial (e.g., a process that saves its
previous state before computing the new one — as both `Upper` and `Lower` do).

### I/O

Landlab has no xarray-native I/O layer. Output typically involves manual
NumPy or netCDF4 calls. This framework's `Output` class, with its chunked
buffering and automatic NetCDF initialization, is ahead of Landlab in this
respect.

### Overall

Landlab is production-grade, well-documented, and has a large component
library. It is the right choice for gridded 2D earth-surface problems. This
framework is more suited to problems with flexible spatial dimensions (HRUs,
sub-basins, 1D reaches) and has a cleaner story around xarray metadata and
lazy file I/O.

### Architecture decision: Discretization object and BMI coupling

A more detailed treatment of Landlab grid types, the HRU-mapping problem, and
the recommended interoperability architecture is documented in:

`external_repos/landlab/landlab_overview.md`

The key conclusions (Section 9 of that document):

- A formal `Discretization` class (wrapping an `xr.Dataset` of HRU areas,
  connectivity, slopes, etc.) is a worthwhile investment independent of Landlab
  coupling, and can support MPI domain decomposition is introduced --
  the discretization is the natural unit of partitioning across ranks.
- The recommended coupling architecture keeps pywatershed and Landlab on
  **separate grids**: pywatershed operates on its HRU-scale `Discretization`;
  Landlab operates on a higher-resolution grid appropriate to the physics
  (raster for overland flow/groundwater, `NetworkModelGrid` for channel
  sediment). Conservative mapping operators between the two grids are computed
  **offline** and stored as sparse weight matrices. **BMI** is the runtime
  exchange layer -- no shared grid object is required.

---

## Strengths Worth Preserving

- The three-channel numpy reference sharing mechanism is genuinely clever and
  performant. It is worth documenting extensively and protecting with active
  assertions.
- The `Input` abstraction cleanly separates "where data comes from" from "how
  processes see data." It is simple and should be kept.
- The `memory vs. file` test parameterization is excellent. It proves that the
  framework is transparent to the data source, which is exactly the right
  invariant to test.
- The chunked `Output` buffering is the right pattern for long model runs.
- Context manager support and `_finalized` guard are good defensive engineering.

---

## Areas Worth Revisiting

- **Buffer-identity assertions: partially covered, unit test now added.**
  The regression test (`test_up_low_regression.py`) already asserts `is`
  identity for all three sharing channels (parameters, forcing, inter-process
  variables) at the integration level — if the `.values =` swap in
  `Process.__init__` ever broke silently, that test would catch it.
  `test_base.py` now also directly asserts buffer identity for both parameters
  and `Input._current_values` inside `TestProcess.test_init`, providing a
  faster-failing unit-level signal. The commented-out `assert` lines in
  `base.py` itself can remain as documentation of intent.
- **Normalize paths to `.resolve()` before deduplication** in
  `_load_shared_data_files` to handle equivalent paths specified differently.
- **The for-loop over space in `calculate()`** (`for loc in self["space"]:`)
  is a Python loop that will be very slow for large spatial domains. The toy
  processes should be vectorized as a design signal to process authors.
- **`get_mutable_inputs` is not fully implemented.** The `raise ValueError`
  in `_initialize_inputs_and_proceses` for mutable inputs from files should
  either be completed or removed and documented as a known gap.
- **Linear-chain assumption should be documented explicitly.** The dict-order
  DAG works for linear chains. Users should be warned that diamond or peer
  dependencies require manual ordering care.
- **The `open_xr` / `xr.open_dataarray` inconsistency.** `Input.__init__`
  calls `xr.open_dataarray` directly while the rest of the Model pipeline uses
  `open_xr`. These should be unified.
- **Time validation.** `_set_time` trusts the first input's time dimension and
  does no cross-input consistency check. For multi-forcing models this is a
  potential source of silent misalignment.
- **The "copy above" comment in `Process.__init__` is likely stale.** The
  comment reads `# self[pp] = parameters[pp]  # no no no - this ruins the refs`
  and `# Apparently a copy above happens.` Testing in `mre_buffer_sharing.py`
  (see below) shows that `Dataset.__setitem__` preserves buffer identity
  unconditionally for numpy-backed DataArrays in current xarray. The `.values =`
  trick is almost certainly redundant today. The comment should be revisited
  against xarray's version history to establish when the behaviour changed,
  and updated to reflect current reality either way.

---

## MRE: Buffer-Sharing Behaviour in xarray (`mre_buffer_sharing.py`)

`mre_buffer_sharing.py` is a self-contained pytest file documenting the numpy
buffer-sharing behaviour that the framework depends on. It has two purposes:

1. **Developer reference** -- each test is a minimal, annotated example of one
   specific sharing behaviour so contributors can understand the mechanism
   without reading `base.py` in full.
2. **Candidate xarray contribution** -- the tests pin behaviour that is not
   explicitly covered in xarray's own test suite. If submitted upstream, any
   future xarray release that silently changes these semantics would be caught.

### Key finding ... however,

For numpy-backed `DataArray`s, `Dataset.__setitem__` **always preserves buffer
identity** in current xarray. This holds regardless of:

- whether the key is new or already exists in the `Dataset`
- whether coordinates are present on the `DataArray`
- whether the incoming dtype differs from the existing variable's dtype
  (xarray replaces the variable wholesale rather than casting into the existing
  slot, so no copy occurs and the `Dataset` variable dtype changes to match)

**However**, see ~L215 in incarnations/xr/base.py

```
# self[pp] = parameters[pp]  # no no no - this ruins the refs
```

Where the refs used to fail (to confirm this is still the case and why, as it is counter the MREs.)

### Test inventory

| Test                                                                         | What it establishes                                                                          |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `test_setitem_into_empty_dataset_preserves_buffer`                           | `ds["x"] = da` on a fresh key preserves the buffer                                           |
| `test_setitem_overwrite_existing_dataset_key_preserves_buffer`               | `ds["x"] = da` on an existing key also preserves the buffer                                  |
| `test_values_assignment_preserves_buffer_identity`                           | `da.values = arr` stores by reference (`da.values is arr`)                                   |
| `test_values_assignment_on_dataset_member_preserves_identity`                | `.values =` trick works on a `DataArray` already inside a `Dataset`                          |
| `test_values_assignment_shared_across_two_datasets`                          | Two `Dataset`s given the same buffer share it; in-place update propagates                    |
| `test_inplace_slice_update_visible_through_original_dataarray`               | `[:]` update is visible through any other reference to the same object                       |
| `test_inplace_slice_update_NOT_visible_through_dataset_copy`                 | Confirms `[:]` update on a `DataArray` does NOT reach an independent copy                    |
| `test_inplace_slice_update_visible_after_values_trick`                       | After `.values =` establishes sharing, `[:]` updates propagate through the `Dataset`         |
| `test_input_advance_pattern_end_to_end`                                      | Full simulation of `Input.advance()` propagating to two shared process `Dataset`s            |
| `test_dataset_variable_selection_buffer_identity` _(Probe A)_                | `ds[["v1","v2"]]` (the exact operation in `Process.__init__`) preserves buffers              |
| `test_setitem_with_coordinates_preserves_buffer_identity` _(Probe B)_        | Coordinates on the `DataArray` do not trigger a copy                                         |
| `test_setitem_dtype_mismatch_preserves_buffer_and_changes_dtype` _(Probe C)_ | Dtype "mismatch" does not trigger a copy; xarray replaces the variable and the dtype changes |

### Implication for `base.py`

The `.values =` trick in `Process.__init__` is redundant for numpy-backed
arrays in current xarray -- plain `Dataset.__setitem__` already preserves the
buffer. It should be retained as defensive programming until the xarray version
history is checked and the stale comment is resolved, at which point the code
and comment can be simplified together.
