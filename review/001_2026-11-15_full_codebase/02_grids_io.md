<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 2: grids and IO](#slice-2-grids-and-io)
  - [Findings](#findings)
    - [2.1 `Output` ignores the time index it is given, so a warm-started run writes its output under the wrong dates](#21-output-ignores-the-time-index-it-is-given-so-a-warm-started-run-writes-its-output-under-the-wrong-dates)
    - [2.2 A run that stops on a chunk boundary without `finalize` leaves rows of 0.0 and gets no warning](#22-a-run-that-stops-on-a-chunk-boundary-without-finalize-leaves-rows-of-00-and-gets-no-warning)
    - [2.3 The buffer a process receives for a `kind="input"` field is writable; `read_only` freezes the source array instead](#23-the-buffer-a-process-receives-for-a-kindinput-field-is-writable-read_only-freezes-the-source-array-instead)
    - [2.4 Setting the read-only flag on a file-backed input with `load=False` reads the whole variable](#24-setting-the-read-only-flag-on-a-file-backed-input-with-loadfalse-reads-the-whole-variable)
    - [2.5 The `.zarr` suffix check raises on rank 0 alone under `ModelMPI`](#25-the-zarr-suffix-check-raises-on-rank-0-alone-under-modelmpi)
    - [2.6 `open_xr` returns a DataArray for a one-variable file: `close()` on it does nothing, and a one-variable parameter file fails with a wrong message](#26-open_xr-returns-a-dataarray-for-a-one-variable-file-close-on-it-does-nothing-and-a-one-variable-parameter-file-fails-with-a-wrong-message)
    - [2.7 `topological_order` checks nothing about the to-index values](#27-topological_order-checks-nothing-about-the-to-index-values)
    - [2.8 The dense weights stay in memory on every rank after `bind`, and `MapMPI.bind` compresses the global matrix only to discard the result](#28-the-dense-weights-stay-in-memory-on-every-rank-after-bind-and-mapmpibind-compresses-the-global-matrix-only-to-discard-the-result)
    - [2.9 The per-step input buffer carries the first time step's `time` as a scalar coordinate into the grid dataset, loses the input's attrs, and is always float64](#29-the-per-step-input-buffer-carries-the-first-time-steps-time-as-a-scalar-coordinate-into-the-grid-dataset-loses-the-inputs-attrs-and-is-always-float64)
    - [2.10 A `Discretization` built from a `Path` opens a file that nothing closes, and `topo_order` writes into the caller's dataset](#210-a-discretization-built-from-a-path-opens-a-file-that-nothing-closes-and-topo_order-writes-into-the-callers-dataset)
    - [2.11 A `MapMPI` whose source is not the distributed grid fails in `bind` with an `AttributeError`](#211-a-mapmpi-whose-source-is-not-the-distributed-grid-fails-in-bind-with-an-attributeerror)
    - [2.12 A `MapMPI` result differs from the serial `Map` in the last bits, and changes with the number of ranks; neither class says so](#212-a-mapmpi-result-differs-from-the-serial-map-in-the-last-bits-and-changes-with-the-number-of-ranks-neither-class-says-so)
    - [2.13 `Output` writes no attrs: the zarr variables have no units and no description](#213-output-writes-no-attrs-the-zarr-variables-have-no-units-and-no-description)
    - [2.14 Arguments that are not validated fail late, with errors that do not name them](#214-arguments-that-are-not-validated-fail-late-with-errors-that-do-not-name-them)
    - [2.15 Docstrings that no longer match the code, and old typing aliases](#215-docstrings-that-no-longer-match-the-code-and-old-typing-aliases)
  - [Checklist A, item by item](#checklist-a-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 2: grids and IO

- pws_phoenix commit: 76af76c (branch `review_001`; the package is
  unchanged from e33e217).
- Paths: `pws_phoenix/{maps,discretization,data_io}.py`
- Checklist: A. Reviewed 2026-11-15.
- Method: `/code-review high` on the three files, then a read of each
  file against checklist A, with the call sites in
  `pws_phoenix/model.py`. The session executed nothing. A finding
  marked "not run" rests on reading how xarray, zarr or numpy behaves
  and needs a short run to confirm before it is fixed.

Findings are ordered by consequence, worst first.

## Findings

### 2.1 `Output` ignores the time index it is given, so a warm-started run writes its output under the wrong dates

- Where: `pws_phoenix/data_io.py:230-242`, `:262-266`
- Code:

      def collect_current_timestep(self, time_index: int) -> None:
          ...
          buffer_index = self.current_time_step % self.time_chunk_size
          ...
          self.current_time_step += 1

      if n_written < len(self.time_values):
          ...
          group["time"].resize((n_written,))

- Consequence: `time_index` is never read. The store's `time`
  coordinate is the whole input time axis
  (`time_values=self.times.values`, `pws_phoenix/model.py:174`), and
  rows are written from row 0. A serial `Model` with
  `control["restart_read"]` that resumes at index 100 and runs 50
  steps calls `collect_current_timestep(100)` ... `(149)`
  (`pws_phoenix/model.py:981`); the values of days 100-149 land in
  rows 0-49, and `finalize` cuts `time` to its first 50 values. The
  store then labels days 100-149 with the dates of days 0-49. No
  error. The rank-0 `Output` of `ModelMPI` is built and called the
  same way (`pws_phoenix/model.py:1640`, `:1730`). The store is also
  created with `mode="w"`, so a resumed run that reuses the first
  run's path replaces the first run's output. No test file uses both
  `restart_read` and `output_serial_zarr`.
- Source: /code-review
- Outcome: open

### 2.2 A run that stops on a chunk boundary without `finalize` leaves rows of 0.0 and gets no warning

- Where: `pws_phoenix/data_io.py:275-276`, `:213-219`
- Code:

      unwritten = self.current_time_step % self.time_chunk_size
      if unwritten > 0:
          warnings.warn(

      self._arrays[var_name] = group.create_array(
          var_name,
          shape=(n_time,) + var_ref.shape,
          chunks=(self.time_chunk_size,) + var_ref.shape,
          dtype=var_ref.dtype,
          dimension_names=("time", str(var_ref.dims[0])),
      )

- Consequence: the warning in `__del__` fires only when a partly
  filled buffer is pending. With `time_chunk_size=1`, or a run that
  stops (an exception, a `step()` loop that ends early) at a multiple
  of `time_chunk_size` short of the full time axis, `unwritten` is 0
  and nothing is reported. The store keeps its full, pre-sized time
  axis. `create_array` is given no `fill_value`, so the rows never
  written read back as zarr's default for a float, 0.0, which cannot
  be told from data. This is the remaining case of the zero tail
  found by pws-prms-bench on 2026-11-03. Not run.
- Source: /code-review
- Outcome: open

### 2.3 The buffer a process receives for a `kind="input"` field is writable; `read_only` freezes the source array instead

- Where: `pws_phoenix/data_io.py:62-65`, `:75`
- Code:

      if read_only:
          self.data.values.flags.writeable = False
      self._current_index = np.int64(-1)
      self._current_values = np.nan * self.data[0, :]
      ...
      self._current: np.ndarray = self._current_values.values

- Consequence: a process is bound to `_current_values`
  (`grid_ds[ii] = current`, `pws_phoenix/model.py:766`), a new
  writable array. The flag goes on `self.data`, which only
  `Input.advance` reads. A process or kernel that writes into a field
  it declares as `kind="input"` is not stopped, in a serial or an MPI
  run; numba types the buffer as writable. The build-time guarantee
  for Maps (`Model._resolve_maps`: every declared writer of the
  source variable precedes the first consumer) counts only
  `mutable_input` declarations, so an undeclared write to an input
  after a Map was applied goes unseen. The only test of the flag,
  `tests/test_up_low_regression.py:133`, checks a parameter.
- Source: checklist A (a check that cannot fail)
- Outcome: open

### 2.4 Setting the read-only flag on a file-backed input with `load=False` reads the whole variable

- Where: `pws_phoenix/data_io.py:63`, `:71`
- Code:

      self.data.values.flags.writeable = False
      ...
      self._in_memory: bool = self.data.variable._in_memory

- Consequence: `.values` on a lazy `(time, space)` variable reads all
  of it from the file. What follows depends on xarray, and the two
  readings disagree. Not run.
  - If xarray keeps the array it read (its default `cache=True`): the
    whole forcing stays in memory, `_in_memory` becomes True, and
    `convertible=inp.data.variable._in_memory`
    (`pws_phoenix/model.py:751`) lets `control["units"] = "convert"`
    convert a file input that the design says must raise.
  - If each `.values` returns a new array, as
    `tests/test_ref_behaviors.py:126-138` asserts for a lazy
    dataset: the full read is thrown away at build, and the flag is
    set on an array nobody holds.
  Either way `load=False` does not hold for a `kind="input"` field.
  `coverage_serial.txt` supports the first reading: the lazy branch
  of `advance` (`pws_phoenix/data_io.py:83`) is never executed by the
  serial tests.
- Run by JLM, 2026-11-15 (env pwpx2), on a `(time, space)` file
  opened with `xr.open_dataarray`: after `a = da.values`,
  `a is da.values` is `False` and `da.variable._in_memory` is
  `True`. CONFIRMED in part: xarray keeps the array it read, so the
  whole variable stays in memory, the lazy branch of `advance` is
  not taken, and `convertible` is True. Each `.values` nevertheless
  returns a different object.
- Second run by JLM, same day: after
  `da.values.flags.writeable = False`, a later `b = da.values` gives
  `np.shares_memory(b, da.values)` `True`, `b.flags.writeable`
  `False`, `b.base is None` `False`. So there is ONE copy of the
  variable in memory, later `.values` are views of it, and the
  array `self._data_values` (line 73) holds is read-only. Not
  determined, and it does not change this finding: whether that is
  because line 63 was the first access and set the flag on the kept
  array itself, or because xarray's views of a kept array are
  read-only anyway.
- What stands: `load=False` does not hold for a `kind="input"`
  field (the whole variable is read at build and kept), and a file
  input passes as `convertible`. The second reading above (a full
  read thrown away) is ruled out.
- Source: /code-review
- Outcome: open

### 2.5 The `.zarr` suffix check raises on rank 0 alone under `ModelMPI`

- Where: `pws_phoenix/data_io.py:152-156`, called from
  `pws_phoenix/model.py:1635-1641`
- Code:

      if self.output_store.suffix != ".zarr":
          raise ValueError(

      if comm.rank == 0:
          self.output = Output(

- Consequence: before the rank-0 branch `ModelMPI._build` checks only
  that `control["output_serial_zarr"]` is present. With
  `output_serial_zarr = "out_seg.nc"` and an output variable on a
  serial grid, rank 0 raises inside `Output.__init__` and the other
  ranks go on to the next collective call, where they wait. A failure
  of `to_zarr` in `_create_store` (a path that cannot be written)
  raises on rank 0 alone in the same way. The comment at
  `pws_phoenix/model.py:1625` ("Output holds NO collectives") covers
  a hang caused by a collective, not one caused by a raise.
- Source: checklist A (a raise on one rank alone)
- Outcome: open

### 2.6 `open_xr` returns a DataArray for a one-variable file: `close()` on it does nothing, and a one-variable parameter file fails with a wrong message

- Where: `pws_phoenix/data_io.py:30-35`
- Code:

      ds = xr.open_dataset(path)
      da_ds: Union[xr.DataArray, xr.Dataset]
      if len(ds.data_vars) == 1:
          da_ds = ds[list(ds.data_vars)[0]]

- Consequence: `ds[name]` does not carry the dataset's close function
  (`xr.open_dataarray` sets it; indexing a dataset does not). The
  Model keeps the DataArray in `_opened_files` and calls `f.close()`
  in `finalize` (`pws_phoenix/model.py:1146`), which does nothing.
  `Input.close` does nothing either, because the Model passes `Input`
  the opened DataArray and `_input_file` stays `None`
  (`coverage_serial.txt`: `pws_phoenix/data_io.py:56-57` and `:100`
  never run). So a one-variable forcing file given as a `Path` stays
  open until garbage collection, against "`finalize` RELEASES
  EXTERNAL RESOURCES"; on Windows an open file cannot be deleted or
  rewritten. Second case: every `Path` in a process entry goes
  through `open_xr` (`pws_phoenix/model.py:184-190`), `parameters`
  included, so a parameter file with exactly one variable arrives as
  a DataArray. `pp in parameters` (`pws_phoenix/model.py:712`) then
  tests the array's values, and the Model raises "declares parameter
  ... but it is in neither ..." for a parameter that is in the file.
  `if isinstance(parameters, pl.Path)` at `pws_phoenix/model.py:704`
  cannot be true for the same reason. Not run.
- Source: /code-review
- Outcome: open

### 2.7 `topological_order` checks nothing about the to-index values

- Where: `pws_phoenix/discretization.py:143-154`
- Code:

      if one_based:
          to_seg = (self.parameters[to_index].values - 1).astype("int64")
      else:
          to_seg = self.parameters[to_index].values.astype("int64")
      ...
          if to_seg[iseg] < 0:
              outflow_mask[iseg] = True
              continue
          connectivity.append((iseg, to_seg[iseg]))

- Consequence, by input:
  - A value larger than the grid (a `tosegment` that points outside
    a subset domain): the edge adds a node that does not exist, and
    the order has more entries than the grid. Through `topo_order=`
    this fails at `pws_phoenix/discretization.py:89` with an xarray
    size error that does not name the cause. A direct call returns
    the bad index.
  - An unsigned `tosegment` with outlet value 0: the subtraction
    happens before the cast, so 0 - 1 wraps to the largest unsigned
    value, the outlet test `< 0` fails, and the same nonexistent
    node appears.
  - A float `tosegment` with nan: the cast gives a negative integer
    on common platforms, and the element becomes an outlet without
    an error.
  - A cycle raises networkx's `NetworkXUnfeasible`, which names
    neither the variable nor the elements.
  Not run. Whether pywatershed checks any of these is a question for
  slice 7.
- Source: /code-review
- Outcome: open

### 2.8 The dense weights stay in memory on every rank after `bind`, and `MapMPI.bind` compresses the global matrix only to discard the result

- Where: `pws_phoenix/maps.py:122`, `:134`, `:169-176`
- Code:

      self.weights = np.asarray(weights)
      ...
      def bind(self, source_ds: Any) -> None:
          super().bind(source_ds)
          ...
          start, stop = self._local
          self._indptr, self._cols, self._vals = _compress_rows(
              self.weights[:, start:stop]
          )

- Consequence: `apply` uses only `_indptr`, `_cols`, `_vals`. After
  `bind` the dense `(n_target, n_source)` matrix is read once more,
  for its shape (`pws_phoenix/model.py:849`), yet each Map keeps it,
  and under MPI each rank holds the full global matrix. About
  110,000 hrus by 57,000 segments is about 50 GB per Map per rank; a
  FlowGraph is fed by three Maps. On each rank `super().bind` also
  runs `np.nonzero` and a fancy-index copy over the whole global
  matrix, and the three results are replaced four lines later. The
  module docstring states "dense weights at construction" as the
  scope; keeping them after `bind`, on every rank, is what prime
  directive 2 does not cover.
- Source: /code-review
- Outcome: open

### 2.9 The per-step input buffer carries the first time step's `time` as a scalar coordinate into the grid dataset, loses the input's attrs, and is always float64

- Where: `pws_phoenix/data_io.py:65`
- Code:

      self._current_values = np.nan * self.data[0, :]

- Consequence: `self.data[0, :]` keeps `time = times[0]` as a scalar
  coordinate. `grid_ds[ii] = current` (`pws_phoenix/model.py:766`)
  adds it to the grid's dataset, so every variable read from that
  dataset shows `time = <first day>` for the whole run, and
  `ds_out[name] = disc.dataset[name]` (`pws_phoenix/model.py:1016`,
  `:1834`) writes it into a restart file beside
  `attrs["state_time"]` of a later day. The multiplication also
  drops the attrs, so the buffer in the grid dataset has no `units`,
  and it turns an integer input into a float64 buffer whatever dtype
  the field declares. Not run.
- Source: /code-review
- Outcome: open

### 2.10 A `Discretization` built from a `Path` opens a file that nothing closes, and `topo_order` writes into the caller's dataset

- Where: `pws_phoenix/discretization.py:69-71`, `:89-92`
- Code:

      if isinstance(parameters, pl.Path):
          parameters = xr.open_dataset(parameters)
      self.parameters = parameters
      ...
      self.parameters[new_var] = (
          self.parameters[to_index].dims,
          self.topological_order(to_index, one_based=topo_one_based),
      )

- Consequence: the class has no `close`, and `Model.finalize` closes
  only `inputs_dict` and `_opened_files`. The file stays open for the
  life of the object, on every rank under `ModelMPI`. When
  `parameters` is the caller's Dataset, the second statement adds
  `new_var` to it, and replaces a variable of that name if there is
  one, without a message.
- Source: /code-review
- Outcome: open

### 2.11 A `MapMPI` whose source is not the distributed grid fails in `bind` with an `AttributeError`

- Where: `pws_phoenix/maps.py:173`
- Code:

      start, stop = self._local

- Consequence: `_local` is set only by `set_decomposition`, which
  `ModelMPI` calls only when `mm.source_grid == mpi_grid`
  (`pws_phoenix/model.py:1614`). The same `maps` dict given to the
  serial `Model` (a serial run to compare with an MPI run), or a
  `MapMPI` between two serial grids under `ModelMPI`, raises
  `AttributeError: 'MapMPI' object has no attribute '_local'`, which
  names neither the Map nor the fix (use `Map`).
- Source: /code-review
- Outcome: open

### 2.12 A `MapMPI` result differs from the serial `Map` in the last bits, and changes with the number of ranks; neither class says so

- Where: `pws_phoenix/maps.py:181-184`
- Code:

      _apply_sparse(
          self._partial, self._indptr, self._cols, self._vals, self._source
      )
      self._comm.Allreduce(self._partial, self._target)

- Consequence: the serial `Map` sums a row's products in column
  order. `MapMPI` sums each rank's part of the row, then adds the
  parts across ranks. For a row with three or more entries on two or
  more ranks the two orders can differ in the last bit, and the
  result depends on how many ranks there are and where the blocks
  split. The docstring of `_apply_sparse` records the same effect
  against a dense product; the `MapMPI` docstring says the target is
  filled "identically on every rank", which is true, and says nothing
  about serial against MPI. A reader comparing a serial and an MPI
  run of a FlowGraph model at 1e-13 needs this.
  `tests/test_two_grid_mpi.py` compares at `rtol=1e-12`.
- Source: checklist A (a serial / MPI difference beyond the table)
- Outcome: open

### 2.13 `Output` writes no attrs: the zarr variables have no units and no description

- Where: `pws_phoenix/data_io.py:213-219` (quoted in 2.2)
- Consequence: `var_ref.attrs` holds the declared `units` and
  `description` (`pws_phoenix/model.py:790-792`); `create_array` is
  not given them. A reader of the output store cannot tell inches
  from cubic feet per second. The comment at
  `pws_phoenix/model.py:1488` says the MPI NetCDF path writes attrs;
  whether it does is a question for slice 3.
- Source: checklist A (a serial / MPI difference beyond the table)
- Outcome: open

### 2.14 Arguments that are not validated fail late, with errors that do not name them

- Where: `pws_phoenix/data_io.py:149`, `pws_phoenix/maps.py:122`,
  `:132`
- Code:

      self.time_chunk_size = time_chunk_size
      ...
      self.weights = np.asarray(weights)
      ...
      self._source: np.ndarray = source_ds[self.source_var].values

- Consequence: `time_chunk_size=0` raises `ZeroDivisionError` at the
  first step, after the store was created. Weights that are not 2-D
  fail in `_compress_rows` with "not enough values to unpack",
  because `mm.bind` runs before the shape check
  (`pws_phoenix/model.py:837`, `:849`). A source variable that is not
  1-D on the source grid (a `(month, space)` field) passes
  `_resolve_maps` and fails at the first `apply` with a numba typing
  error. Not run.
- Source: checklist A (an error that passes silently)
- Outcome: open

### 2.15 Docstrings that no longer match the code, and old typing aliases

- Where: `pws_phoenix/data_io.py:12-14`, `:119-120`, `:19`, `:235`;
  `pws_phoenix/maps.py:1-3`
- Code:

      Output is the serial writer (one zarr store). The MPI path (ModelMPI in
      model.py) does NOT use Output -- it streams output through mpixarray

      from typing import Any, Dict, List, Union

      """
      map.py
      ======

- Consequence: `ModelMPI` builds an `Output` on rank 0 for
  serial-grid variables (`pws_phoenix/model.py:1636`). The file is
  `maps.py`. `Args: time_index: Global time index.` documents an
  argument that is not read (2.1). `Dict`, `List` and `Union` are
  against convention 1 ("python 3.14 syntax, particularly for
  typehinting"); `maps.py` and `discretization.py` use `dict` and
  `X | None`. Documentation and style only.
- Source: /code-review
- Outcome: open

## Checklist A, item by item

- Hidden copy: 2.4 (`self.data.values` on a lazy input). At build
  only, and small: `cols.astype(np.int64)` and `weights[rows, cols]`
  in `_compress_rows`; `(values - 1).astype("int64")` in
  `topological_order`. `np.asarray(weights)` does not copy an
  ndarray. `var_ref.values` and `self._data_values[index]` do not
  copy.
- Allocation outside the three allowed classes: 2.8 (the dense
  weights after `bind`). The lazy branch of `Input.advance`
  (`self.data[self._current_index, :].values`) allocates one
  `(space,)` array per step and says why in a comment.
  `Map.target_values`, `MapMPI._partial`, `Input._current_values`
  and `Output.data_buffers` are declared per-step buffers, allocated
  once.
- `self._obj[name].values` inside a step method: none. `Map` and
  `Input` hold ndarrays. `Output.collect_current_timestep` reads
  `var_ref.values` once per output variable per step: a property of
  a DataArray held since build, not a lookup in a Dataset. Holding
  the ndarray, as `Input` does, would remove it. Not measured.
- A raise on one rank alone: 2.5. A collective inside a rank-0
  branch: none; `Output` has no collective, and every rank calls
  `MapMPI.apply` and `Discretization.decompose`.
- Serial / MPI difference beyond the table: 2.12, 2.13.
- An error that passes silently, or a check that cannot fail: 2.1,
  2.2, 2.3, 2.6, 2.7, 2.14.

Checked and not a finding: a Map-fed input is covered by units mode
(`pws_phoenix/model.py:894-913` compares the units the source's
owner and each consumer declare).

## Notes for later slices

- Slice 3 (`pws_phoenix/model.py`): `inputs_dict` is keyed by the
  input's name alone (`pws_phoenix/model.py:738`), so two processes
  on different grids that name the same input share one `Input`.
- Slice 3: nothing in `Input` checks that an input's time axis is
  the model's; `Model._set_time` takes the axis from one input.
  `Input.advance` past the end of a shorter input raises a bare
  `IndexError`.
- Slice 3: `ModelMPI` accepts `discretizations` for serial grids
  only (comment at `pws_phoenix/model.py:1241`). Whether
  `parameters=` on the distributed grid's `Discretization` raises or
  is ignored was not checked here.
- Slice 3: does the MPI NetCDF stream carry `units` and
  `description` (2.13)?
- Slice 7: does pywatershed validate `tosegment` before its
  networkx sort (2.7)?
- Slice 16: why no test caught 2.1 (no test with `restart_read` and
  `output_serial_zarr` together), 2.2 (the `__del__` test's chunk
  size), 2.3 (no test writes to an input buffer). The serial tests
  never run a lazy input (`pws_phoenix/data_io.py:56-57`, `:61`,
  `:83`), `open_xr(load=True)` (`:37`), the serial identity of
  `decompose`, or any of `MapMPI` (expected: MPI coverage is
  deferred).
- Slice 16: `tests/test_ref_behaviors.py:126-138` explains its
  result as "Each .values access reads from disk". The run recorded
  in 2.4 shows the data are kept in memory after the first access
  while the object identity still differs, so that comment needs
  checking.
