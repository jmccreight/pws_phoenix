<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 3: the Model](#slice-3-the-model)
  - [Findings](#findings)
    - [3.1 A supplied parameter's dims are never compared with its declaration](#31-a-supplied-parameters-dims-are-never-compared-with-its-declaration)
    - [3.2 The clock comes from the first input; no other input's time axis is compared with it](#32-the-clock-comes-from-the-first-input-no-other-inputs-time-axis-is-compared-with-it)
    - [3.3 Assigning an array into the grid dataset lets xarray align it by label: differing labels give a new array filled with nan](#33-assigning-an-array-into-the-grid-dataset-lets-xarray-align-it-by-label-differing-labels-give-a-new-array-filled-with-nan)
    - [3.4 `write_restart` before the first step stamps the file with the step that has not run](#34-write_restart-before-the-first-step-stamps-the-file-with-the-step-that-has-not-run)
    - [3.5 `_read_restart` skips a process's python-attribute state without a message when the file has none under the process's current name](#35-_read_restart-skips-a-processs-python-attribute-state-without-a-message-when-the-file-has-none-under-the-processs-current-name)
    - [3.6 Under MPI the units loop reconciles a parameter once per declaring process, so a patched parameter is converted once per declarer](#36-under-mpi-the-units-loop-reconciles-a-parameter-once-per-declaring-process-so-a-patched-parameter-is-converted-once-per-declarer)
    - [3.7 Under MPI a raise that depends on the data, in `initialize()` or `calculate()`, happens on one rank alone](#37-under-mpi-a-raise-that-depends-on-the-data-in-initialize-or-calculate-happens-on-one-rank-alone)
    - [3.8 `_restore_grid_var` never checks the extent of supplied initial state or of a restart variable](#38-_restore_grid_var-never-checks-the-extent-of-supplied-initial-state-or-of-a-restart-variable)
    - [3.9 `ModelMPI` declares every state variable as float64 on `(mpi_grid,)` with a nan fill and no attrs](#39-modelmpi-declares-every-state-variable-as-float64-on-mpi_grid-with-a-nan-fill-and-no-attrs)
    - [3.10 The distributed grid's parameters are never set read-only](#310-the-distributed-grids-parameters-are-never-set-read-only)
    - [3.11 A key in a process entry that is not used gives no message, and a supplied input can take the place of a producer's variable, depending on the order](#311-a-key-in-a-process-entry-that-is-not-used-gives-no-message-and-a-supplied-input-can-take-the-place-of-a-producers-variable-depending-on-the-order)
    - [3.12 A key of `control` that the Model does not know is ignored](#312-a-key-of-control-that-the-model-does-not-know-is-ignored)
    - [3.13 `ModelMPI.run` ignores `n_steps`, and nothing stops a second call](#313-modelmpirun-ignores-n_steps-and-nothing-stops-a-second-call)
    - [3.14 Under MPI every rank holds full-extent arrays: supplied state for the whole run, the restart file at read, every restart variable at write](#314-under-mpi-every-rank-holds-full-extent-arrays-supplied-state-for-the-whole-run-the-restart-file-at-read-every-restart-variable-at-write)
    - [3.15 A Map whose source variable is another Map's target is not checked for order](#315-a-map-whose-source-variable-is-another-maps-target-is-not-checked-for-order)
    - [3.16 Units mode never compares two declarations of one name on one grid, nor a Map whose source has no owning process](#316-units-mode-never-compares-two-declarations-of-one-name-on-one-grid-nor-a-map-whose-source-has-no-owning-process)
    - [3.17 Build checks that `Model` and `ModelMPI` do not share](#317-build-checks-that-model-and-modelmpi-do-not-share)
    - [3.18 `inputs_dict` is keyed by the input's name alone, so the error for a name supplied on two grids blames the wrong array](#318-inputs_dict-is-keyed-by-the-inputs-name-alone-so-the-error-for-a-name-supplied-on-two-grids-blames-the-wrong-array)
    - [3.19 The MPI run loop looks up each input and output variable in the dataset on every step](#319-the-mpi-run-loop-looks-up-each-input-and-output-variable-in-the-dataset-on-every-step)
    - [3.20 The MPI budget error reports the largest count on one rank as the number of locations](#320-the-mpi-budget-error-reports-the-largest-count-on-one-rank-as-the-number-of-locations)
    - [3.21 A raise during the build leaves files and the writer open, and `finalize` writes the output tail last](#321-a-raise-during-the-build-leaves-files-and-the-writer-open-and-finalize-writes-the-output-tail-last)
    - [3.22 Arguments that are not validated fail late, with errors that do not name them](#322-arguments-that-are-not-validated-fail-late-with-errors-that-do-not-name-them)
    - [3.23 `ModelMPI` repeats parts of `Model` instead of sharing them, and the copies have drifted](#323-modelmpi-repeats-parts-of-model-instead-of-sharing-them-and-the-copies-have-drifted)
    - [3.24 Docstrings and comments that no longer match the code, old typing aliases, single-letter names](#324-docstrings-and-comments-that-no-longer-match-the-code-old-typing-aliases-single-letter-names)
  - [Checklist A, item by item](#checklist-a-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 3: the Model

- pws_phoenix commit: c1860bc (branch `review_001`; the commits since
  e33e217 are review commits).
- Paths: `pws_phoenix/model.py`
- Checklist: A. Reviewed 2026-11-15.
- Method: `/code-review high` on the file, then a read of the whole
  file against checklist A, with the code it calls in
  `pws_phoenix/{process,units,data_io,discretization,maps}.py` and
  the `initialize` methods of the HRU-grid processes. The session
  executed no project code; it used text scans (grep, awk) only.
  mpixarray was not read (it is outside the repository), so a
  statement about what mpixarray does with a declaration says so. A
  finding marked "not run" rests on reading how xarray, numpy or
  numba behaves and needs a short run to confirm before it is fixed.

Findings are ordered by consequence: first those that give wrong
numbers without an error (3.1-3.6), then those that hang or differ
between serial and MPI (3.7-3.13), then cost, messages and upkeep.

## Findings

### 3.1 A supplied parameter's dims are never compared with its declaration

- Where: `pws_phoenix/model.py:720-729` (serial), `:1375-1377` and
  `:1411-1415` (MPI)
- Code:

      source[pp].load()
      ...
      grid_ds[pp] = source[pp]
      grid_ds[pp].values.flags.writeable = False

      bad_dims = {
          name: ds_input[name].dims
          for name in file_input_names
          if ds_input[name].dims != ("time", mpi_grid)
      }

- Consequence: inputs get a dims check (`:755`, `:1411`); parameters
  get none. Serial: a parameter supplied as `(nhru, nmonth)` for a
  field declared `("nmonth", "space")` is wired in as it is. The
  kernel indexes `[month, hru]` on a `(nhru, 12)` array; numba does
  not check bounds, so it reads another element or memory past the
  array, with no error. A parameter on `nhm_id` where the grid dim is
  `nhru` rides along as a second dim of the grid dataset, which is
  the case the comment at `:756-759` describes for inputs only. MPI:
  a static parameter in `input_file` on a dim other than `mpi_grid`
  is not split by `decompose`, is loaded whole at `:1375-1377`, and
  each rank binds the full-extent array. A kernel that loops over the
  rank's own HRU count reads the first `local_n` values on every
  rank, so ranks above 0 use rank 0's parameter values. Not run.
- Source: /code-review
- Outcome: open

### 3.2 The clock comes from the first input; no other input's time axis is compared with it

- Where: `pws_phoenix/model.py:938-943`, `:1117-1118`, `:1706-1707`
- Code:

      kk0 = list(self.inputs_dict.keys())[0]
      data = self.inputs_dict[kk0].data
      self.ntime = data.sizes["time"]
      self.times = data["time"]

      for inp in self.inputs_dict.values():
          inp.set_index(self._start_index)

- Consequence: `Input.advance` serves each input by position. With
  `prcp` starting 1979-01-01 and `tmax` starting 1980-01-01, both of
  the same length, every step pairs 1979 precipitation with 1980
  temperature, with no error. A shorter second input raises a bare
  `IndexError` in the middle of the run. A warm start applies the one
  positional index to every input (`:1117-1118`). Under `ModelMPI`
  the serial-grid inputs advance by position beside the stream
  (`:1706-1707`) and are never compared with `ds_input["time"]`. The
  Model docstring promises that inputs "need only COVER the restart
  time" (`:54-55`); that holds only when every input has the same
  axis. This is the slice 2 note on `Input`, confirmed.
- Source: /code-review
- Outcome: open

### 3.3 Assigning an array into the grid dataset lets xarray align it by label: differing labels give a new array filled with nan

- Where: `pws_phoenix/model.py:728-729`, `:766`
- Code:

      grid_ds[pp] = source[pp]
      grid_ds[pp].values.flags.writeable = False

      grid_ds[ii] = current

- Consequence: `Dataset.__setitem__` aligns a DataArray to the index
  the dataset already has on that dim. If the first array added
  carries an `nhru` index of nhm ids (a pywatershed output after
  `.rename({"nhm_id": "nhru"})`) and a later one carries
  `nhru = 0..n-1`, the later one is reindexed onto the first's
  labels: a new array, nan wherever a label is absent. For an input,
  the process then holds the new array while `Input.advance` keeps
  writing the original buffer, so the input reads nan on every step.
  For a parameter, the read-only flag lands on the new array and the
  process reads nan or reordered values. No error in either case, and
  the zero-copy rule is broken without a message. Nothing checks that
  the arrays agree on the grid dim's labels or that
  `grid_ds[name].values is source.values` after the assignment. When
  only one of the two carries an index, or the labels are equal,
  there is no reindexing. Not run.
- Source: /code-review
- Outcome: open

### 3.4 `write_restart` before the first step stamps the file with the step that has not run

- Where: `pws_phoenix/model.py:1005-1006` with `:143`; `:1807-1808`
  with `:1334`
- Code:

      self.current_time_index = np.array([self._start_index], dtype=np.int32)

      tt = int(self.current_time_index[0])
      state_time = self.times[tt].values

      self.current_time_index = np.zeros(1, dtype=np.int64)

- Consequence: the docstring says the file is written "for the LAST
  COMPLETED step". Before any step, `current_time_index` holds the
  next step to run (serial) or 0 (MPI). Serial cold start,
  `model.write_restart(dir)` before `step()` (saving the initial
  state): the file is stamped `times[0]`, and a model warm-started
  from it resumes at index 1 (`:1111`), so day 0 is never run. Serial
  warm start at index 100, write before stepping: state as of day 99
  is stamped day 100, and the next run skips day 100. `ModelMPI`
  warm-started at index 100, write before `run()`: day-99 state is
  stamped `times[0]`. No error in any of the three.
- Source: /code-review
- Outcome: open

### 3.5 `_read_restart` skips a process's python-attribute state without a message when the file has none under the process's current name

- Where: `pws_phoenix/model.py:1059-1061`, `:1084-1092`, `:1049-1050`
- Code:

      file_vars = {nn for nn in all_names if "__" not in nn}
      if file_vars != expected:

      prefix = f"{proc_name}__"
      state = { ... for nn in all_names if nn.startswith(prefix) }
      if state:
          self.model_dict[proc_name].set_restart_state(state)

- Consequence: the declared restart variables are compared as a set;
  the `{process}__{name}` variables are not compared with anything. A
  restart written with the process_dict key `"PRMSStreamTemp"` and
  read by a model that names the same class `"stream_temp"` passes
  the set comparison, finds no name starting with `stream_temp__`,
  and never calls `set_restart_state`. The silos and their indices
  keep their cold values, and the file's `PRMSStreamTemp__*`
  variables are ignored. The run resumes with different stream
  temperatures and no message. The same holds for a file that lacks
  the hook state entirely, and for a grid whose processes have hook
  state but no declared restart variable and no file
  (`if not expected and not found: continue`).
- Source: /code-review
- Outcome: open

### 3.6 Under MPI the units loop reconciles a parameter once per declaring process, so a patched parameter is converted once per declarer

- Where: `pws_phoenix/model.py:1430-1440`; contrast `:708-709`
- Code:

      for proc_name, cls in mpi_classes.items():
          fields = _resolved_fields(cls)
          for pp in cls.get_parameters():
              if pp in ds_mpi_stream:
                  self._reconcile_units(
                      pp, ds_mpi_stream[pp], fields[pp].units,
                      convertible=True, ...

- Consequence: with `control = {"units": "convert", "units_patch":
  {"hru_area": "m^2"}}` and two distributed-grid processes that both
  declare `hru_area` (acres), the first pass converts the array in
  place and sets its attrs. The second pass reads the patch before
  the attrs (`pws_phoenix/units.py:250`), sees `m^2` again and
  multiplies again. The serial path reaches each parameter once
  (`if pp in grid_ds: continue`). This is finding 1.1 inside a single
  build. Without a patch the second pass finds the corrected attrs
  and does nothing, provided `ds_mpi_stream[pp].attrs` is the stored
  variable's attrs (true for a plain `xr.Dataset`; not checked for
  mpixarray's).
- Source: /code-review
- Outcome: open

### 3.7 Under MPI a raise that depends on the data, in `initialize()` or `calculate()`, happens on one rank alone

- Where: `pws_phoenix/model.py:547-549`, `:1720`; the raises are in
  `pws_phoenix/hydrology/prms_soilzone.py:1041-1042`, `:1063-1064`,
  `:192-193`, `pws_phoenix/hydrology/prms_soilzone_ag.py:822`,
  `:855`, `:860`, `pws_phoenix/hydrology/prms_runoff.py:2005-2007`,
  `pws_phoenix/hydrology/prms_snow.py:1984-1985`,
  `pws_phoenix/atmosphere/prms_atmosphere.py:448-449`
- Code:

      for kk in self.model_dict:
          self.model_dict[kk].initialize()

      if (refs["soil_moist_max"] < 1.0e-5).any():
          raise NotImplementedError(

      if not (refs["hru_lat"] > 0.0).any():
          raise NotImplementedError(

- Consequence: under `ModelMPI`, `refs[...]` is one rank's block.
  The rank whose block holds the offending value raises; the others
  go on to the `allreduce` in `_raise_if_bad_initial_state`
  (`:1749`) and wait. `ModelMPI` agrees across ranks before raising
  in two places only (`:1739`, `:1749`). Nothing covers
  `initialize()`, `calculate()` (the `RuntimeError` in
  `compute_interflow`), `_apply_supplied_state` (3.8) or the rank-0
  `Output` (2.5). Neither the package nor `tests/run_tests.sh` has
  an `Abort`, an excepthook or `python -m mpi4py`, so an uncaught
  exception on one rank does not stop the job. The atmosphere guard
  also changes meaning: in serial it asks whether the whole domain
  has an HRU north of the equator, under MPI whether one rank's
  block has. A domain with HRUs in both hemispheres passes in serial
  and can raise on one rank.
- Source: checklist A (a raise on one rank alone)
- Outcome: open

### 3.8 `_restore_grid_var` never checks the extent of supplied initial state or of a restart variable

- Where: `pws_phoenix/model.py:1134-1135`, `:1867-1869`
- Code:

      buffer = grid_ds[name].values
      buffer[:] = values

      aa, bb = self._decomp_slice
      buffer = self._ds_mpi_stream[name].values
      buffer[:] = values[aa:bb]

- Consequence: MPI, 765 HRUs on 4 ranks, supplied `gwres_stor` of
  length 700: ranks 0-2 pass; rank 3 gets 127 values for 192 slots
  and raises alone, and the others wait in the `allreduce` of
  `_raise_if_bad_initial_state`. An array of length 800, or a restart
  file from a larger domain, is accepted and cut to the first 765
  values. Serial: a wrong length raises numpy's broadcast error, but
  a length-1 array or a scalar fills the whole variable. `n_global`
  is known at `:1553`, so the same check could run on every rank.
  `buffer[:] = values` also casts without a check: a float value
  written into a bool or int64 buffer (see 3.9) becomes True or an
  arbitrary integer when it is nan.
- Source: /code-review
- Outcome: open

### 3.9 `ModelMPI` declares every state variable as float64 on `(mpi_grid,)` with a nan fill and no attrs

- Where: `pws_phoenix/model.py:1489-1504`; contrast `:788-797`
- Code:

      def _declare(name, dims, fill, to_file):
          ds_mpi_stream.mpi[name] = {
              "dims": dims,
              "dtype": f64,
              "fill_value": f64(fill),
              ...
      for name in cls.get_var_names(include_budget=True):
          if name not in declared:
              _declare(name, (mpi_grid,), np.nan, False)

- Consequence: the serial path uses `meta.dtype`, `meta.dims`,
  `_FILL_VALUE[meta.dtype]`, `meta.units` and `meta.description`;
  this loop uses none of them. PRMSSnow's `iasw`, `lst`, `newsnow`
  (bool) and `int_alb`, `iso`, `lso`, `mso` (int64) are float64 on
  the distributed grid. The kernels are then compiled for a signature
  other than the one the serial parity tests exercise. A bool starts
  nan (true in an `if`) instead of False;
  `pws_phoenix/process.py:248` relies on "initialize() must fully set
  them". MPI restart files hold these variables as float64, so the
  files are not "format-identical" to serial ones (`:57`, `:1797`),
  and a serial read casts them back through `buffer[:] = values`
  (3.8). `write_restart` has to re-attach units by hand
  (`:1823-1831`) and drops the description. The `{name}_out`
  declarations pass no attrs either (`:1537`): whether mpixarray adds
  any to the NetCDF file was not checked. This answers the slice 2
  note on 2.13 as far as this file goes. A state variable with a
  second dim would be declared 1-D without a raise.
- Source: /code-review
- Outcome: open

### 3.10 The distributed grid's parameters are never set read-only

- Where: `pws_phoenix/model.py:1375-1377`; contrast `:729`
- Code:

      for name in list(ds_mpi_stream.data_vars):
          if "time" not in ds_mpi_stream[name].dims:
              ds_mpi_stream[name] = ds_mpi_stream[name].load()

- Consequence: `flags.writeable = False` is set at `:729` (the
  parameters `_add_process_fields` adds: serial, and the serial grids
  under MPI) and at `:553` (internal parameters, both paths). The
  parameters that come from `input_file` are never frozen. A kernel
  that writes to a parameter raises in serial and runs under MPI,
  where it changes the parameter for every later step and for every
  process that shares it. The rule "frozen parameters are never
  edited" is therefore enforced by the serial tests only.
- Source: /code-review (a candidate it cut), checklist A
- Outcome: open

### 3.11 A key in a process entry that is not used gives no message, and a supplied input can take the place of a producer's variable, depending on the order

- Where: `pws_phoenix/model.py:687-691`, `:735-737`, `:786-787`,
  `:802`, `:1586-1590`
- Code:

      if ii in grid_ds:
          continue  # produced/added on this grid -> structural share
      if ii in init_dict:

      for name, meta in allocated.items():
          if name in grid_ds:
              continue
          ...
          if meta.restart and name in init_dict:

- Consequence:
  - `{"class": PRMSGroundwater, "parameters": params, "gwres_store":
    arr}` (a misspelling of `gwres_stor`) builds and runs from
    `initialize()`'s default storage. Only the names in
    `required_initial_state` are caught.
  - A value supplied for a variable without `restart=True` is
    ignored.
  - Under MPI every data key of a distributed-grid entry except the
    restart variables is ignored (`:1586`): a serial process_dict
    reused for `ModelMPI` has its `parameters` and forcing arrays set
    aside for whatever `input_file` holds.
  - Order. Process A consumes `flow`, process B produces it, and A's
    entry supplies a `flow` array. With A listed first, A's array
    becomes an `Input`; B's variable is then not allocated (`:786`)
    and B writes its result into the `Input`'s buffer (writable, see
    2.3), which `Input.advance` overwrites at the next step. A reads
    the supplied values on every step and never B's. With B listed
    first, A's array is ignored. `input_spec` reports `flow` under
    "internal_inputs" in both orders (`:388`). The same applies to a
    supplied input whose name a later process declares as a
    parameter (`:708`): that process's own parameter data are set
    aside for the `Input`'s buffer.
  - Supplied state for a variable name that two processes on one
    grid declare: the second process's array is skipped in serial
    (`:786` comes before `:802`) and replaces the first's under MPI
    (`:1586`, `:577`).
- Source: /code-review, checklist A (an error that passes silently)
- Outcome: open

### 3.12 A key of `control` that the Model does not know is ignored

- Where: `pws_phoenix/model.py:107`, `:111`, `:135`, `:152`, `:161`,
  `:469`, `:476-477`, `:494`, `:502`, `:1272-1277`, `:1346`, `:1629`,
  `:1637`
- Code:

      mode = control.get("budget")
      ...
      restart_read = control.get("restart_read")

- Consequence: the keys read are `load_all`, `budget`, `budget_rtol`,
  `budget_atol`, `units`, `units_patch`, `restart_read`,
  `output_var_names`, `output_serial_zarr`, `time_chunk_size`, and
  under MPI `input_file`, `output_parallel_netcdf`, `mpi_grid`,
  `progress`. Only the values of `budget` and `units` are validated.
  `"restart_reed"` gives a cold start; `"budgets": "error"` gives a
  run in which no budget is judged; `"unit": "check"` checks nothing.
  A name in `units_patch` that matches no supplied array is ignored
  the same way. `CLAUDE.md` lists an `Options` class in the backlog.
- Source: checklist A (an error that passes silently)
- Outcome: open

### 3.13 `ModelMPI.run` ignores `n_steps`, and nothing stops a second call

- Where: `pws_phoenix/model.py:1674-1697`
- Code:

      def run(self, dt: np.float64, n_steps: np.int32 | None = None) -> None:
          """Stream the time loop via iterate(); n_steps is ignored ...
          if self._finalized:
              raise RuntimeError("Cannot run a finalized Model.")
          ...
          for tt, step in enumerate(
              ds_mpi_stream.mpi.iterate(comm=self._comm, progress=self._progress)

- Consequence: `model.run(dt, n_steps=10)` runs every remaining step
  of `input_file`; the serial Model runs 10. The docstring says so,
  but the call neither raises nor warns. `_finalized` is the only
  guard on a second `run()`: `enumerate` starts again at 0, so
  `gtt` returns to `self._start_index` while the state is that of
  the end of the first run. Whether mpixarray allows a second
  `iterate`, and what its writer does with rows already written, was
  not checked.
- Source: /code-review (a candidate it cut), checklist A
- Outcome: open

### 3.14 Under MPI every rank holds full-extent arrays: supplied state for the whole run, the restart file at read, every restart variable at write

- Where: `pws_phoenix/model.py:1263` with `:188`; `:577`; `:1057`;
  `:1820-1822`, `:1827-1830`; `:1375-1377`
- Code:

      self._load_all = True  # serial-grid data are small; keep in memory

      self._supplied_state[(grid, name)] = values

      ds_in = xr.load_dataset(found[0])

      parts = comm.allgather(
          self._ds_mpi_stream[name].values.copy()
      )

- Consequence: none of these is model state, a declared per-step
  buffer or a kernel temporary.
  - Supplied initial state for the distributed grid is opened by
    `_load_paths_to_data` with `load=True` on every rank (`:1568`;
    the comment at `:1263` speaks of serial-grid data, but the call
    covers every entry). It stays referenced by
    `self._supplied_state` and by `self._process_dict` for the life
    of the model: `Model.__init__` deletes `_process_dict` (`:126`),
    `ModelMPI` does not. In serial, a file-backed supplied state is
    read by `np.asarray(values)` (`:563`) and, by xarray's cache (see
    the run recorded in 2.4), also stays in memory.
  - `_read_restart` loads the whole file on every rank. It is
    released when the method returns.
  - `write_restart`: `allgather` gives every rank the global array
    of every restart variable, and `ds_out` holds them all until the
    grid's file is written, though only rank 0 writes.
    `gather(root=0)` would leave them on rank 0 alone. `.copy()`
    adds one local copy before the pickle.
  - `:1375-1377` loads every variable of `input_file` without a
    `time` dim, whether or not a process declares it.
  For N HRUs and S restart variables the first three cost 8 x N x S
  bytes on each rank. As an illustration, 350,000 HRUs and 60
  variables are 170 MB per rank, times the number of ranks on a
  node. S for the NHM configuration was not counted.
- Source: checklist A (an allocation outside the three allowed
  classes); the `allgather` from /code-review
- Outcome: open

### 3.15 A Map whose source variable is another Map's target is not checked for order

- Where: `pws_phoenix/model.py:868-885`, `:916-929`
- Code:

      owners = [ kk ... if self._proc_grid[kk] == mm.source_grid
                 and mm.source_var in proc.get_var_names() ]
      writers = owners + [ ... mm.source_var in proc.get_mutable_inputs() ]
      ...
      late = [ww for ww in writers
              if position[ww] >= position[first_consumer]]

- Consequence: the build guarantee is "every declared writer on the
  source grid precedes the first consumer". A source variable that
  is itself Map-fed (grids A -> B -> C, the second Map reading on B
  the buffer the first Map fills) has no owner and no writer in
  these lists; its writer is the first Map's `apply`, which runs
  before the first consumer on B. If the consumer on C is listed
  before the consumer on B, the second Map is applied first and
  carries the previous step's value. No error. `get_var_names()` is
  called without `include_budget=True`, so a mapped budget variable
  also has no owner here and its order is not checked. Whether a
  shipped configuration chains Maps was not searched: the check is
  absent, not known to be violated.
- Source: checklist A (a check that cannot fail)
- Outcome: open

### 3.16 Units mode never compares two declarations of one name on one grid, nor a Map whose source has no owning process

- Where: `pws_phoenix/model.py:708-709`, `:735-736`, `:894`
- Code:

      if pp in grid_ds:
          continue

      if ii in grid_ds:
          continue  # produced/added on this grid -> structural share

      if self._units_mode is not None and owners:

- Consequence: with `control["units"] = "check"`, a parameter is
  reconciled against the first process that declares it; a second
  declarer's units are not read. A consumer's declared units for an
  input that another process on the grid produces are not compared
  with the producer's. A Map whose source variable is a supplied
  input or a parameter (no owner) skips the comparison with its
  consumers. The scan of the `DataArrayMeta` declarations that /code-review
  made found no name declared with two different units, so no
  shipped pair is known to be wrong;
  the names of the tests in `tests/test_units.py` show none that
  compares declarations across classes (the bodies were not read:
  slice 16).
- Source: /code-review (a candidate it cut)
- Outcome: open

### 3.17 Build checks that `Model` and `ModelMPI` do not share

- Where: `pws_phoenix/model.py:1303-1312`, `:1388-1393`,
  `:1398-1422`, `:1596-1598`, `:1272-1274`, `:1637`
- Code:

      # Taxonomy guard (all grids): a parameter on the model `time` axis
      # is really an input.

      if ii not in mpi_var_names and ii not in file_input_names:
          file_input_names.append(ii)

      output_parallel_netcdf = str(control["output_parallel_netcdf"])
      out_var_names = list(control["output_var_names"])

- Consequence:
  - The guard against a `parameter` on `time` exists only in
    `ModelMPI._build`. The serial `Model` accepts the declaration and
    keeps the whole `(time, space)` array resident. `CLAUDE.md`
    states where the guard is; the declaration is wrong on both
    paths.
  - An input of a distributed-grid process that a same-named
    parameter or internal parameter satisfies builds in serial, and
    `input_spec` lists it under "internal_inputs" (`:393-414`). Under
    MPI it lands in `file_input_names` and raises "not produced by
    any process ... and were not found in input_file" (internal
    parameter) or "streamed input(s) must have dims ('time', ...)"
    (parameter). A text scan found no such pair in the shipped
    HRU-grid classes.
  - A distributed-grid parameter that is absent from `input_file`
    surfaces as a bare `KeyError` from `Process.__init__`
    (`pws_phoenix/process.py:333-336`); serial names the process and
    the parameter (`:715-719`). `_validate_inputs_resolved` covers
    inputs only.
  - `control["input_file"]`, `["output_parallel_netcdf"]` and
    `["output_var_names"]` are required under MPI, each a bare
    `KeyError` when absent, even when no distributed-grid variable is
    written. The serial `Model` runs with no output keys.
    `time_chunk_size` missing warns in serial (`:165`) and is silent
    under MPI.
- Source: /code-review (the `KeyError`), checklist A (a serial / MPI
  difference beyond the table)
- Outcome: open

### 3.18 `inputs_dict` is keyed by the input's name alone, so the error for a name supplied on two grids blames the wrong array

- Where: `pws_phoenix/model.py:738-744`, `:755-765`
- Code:

      if ii not in self.inputs_dict:
          self.inputs_dict[ii] = Input(init_dict[ii], ...)
      inp = self.inputs_dict[ii]

- Consequence: `_supplied_state` is keyed by `(grid, name)`;
  `inputs_dict` is not. Processes on `nhru` and on `nsegment` each
  declare and are supplied an input `prcp`. The second finds `"prcp"`
  in `inputs_dict`, never opens its own array, runs the units
  reconcile again on the first grid's data, then raises "input 'prcp'
  arrives on dims ('nhru',) but grid 'nsegment' expects
  ('nsegment',) -- rename the input's spatial dim", although the
  segment array it was given is correct. Two grids cannot each be
  supplied an input of one name. This is the slice 2 note,
  confirmed.
- Source: /code-review
- Outcome: open

### 3.19 The MPI run loop looks up each input and output variable in the dataset on every step

- Where: `pws_phoenix/model.py:1703-1704`, `:1725-1727`
- Code:

      for name in self._file_input_names:
          step[name].values[:] = src[name].values[0, :]
      ...
      for src_name, buf in self._output_map.items():
          step[buf].values[0, :] = step[src_name].values[:]
          step[buf].mpi.write()

- Consequence: two dataset lookups per file input and three per
  streamed output variable, per step, on every rank. `DEVELOPER.md`
  ("Per-step data access") measured about 20 us per lookup on the
  serial HRU dataset and removed them from the processes; it also
  states that the input buffers and the state are the same ndarrays
  for the whole run, so the references on the left of `:1704` and on
  the right of `:1726` could be held once. `src` and the object that
  `.mpi.write()` is called on belong to the step. With 6 inputs and
  10 output variables that is 42 lookups a step; the cost on an
  mpixarray dataset was not measured. `src[name].values[0, :]` also
  allocates one array per input per step (the read from the file),
  with no comment that accepts it, unlike the lazy branch of
  `Input.advance`.
- Source: checklist A (`self._obj[name].values` inside a step)
- Outcome: open

### 3.20 The MPI budget error reports the largest count on one rank as the number of locations

- Where: `pws_phoenix/model.py:1739-1740`, `:536-540`
- Code:

      n_bad = self._comm.allreduce(int(n_bad), op=MPI.MAX)
      super()._raise_if_imbalanced(n_bad, proc)

      f"{self.time.current}: {n_bad} location(s) outside "

- Consequence: with 3 imbalanced HRUs on rank 0 and 5 on rank 2 the
  message on every rank reads "5 location(s)". `MPI.SUM` would give
  8. The decision to raise is not affected. This is the slice 1
  note, confirmed (see also 1.7).
- Source: /code-review (a candidate it cut)
- Outcome: open

### 3.21 A raise during the build leaves files and the writer open, and `finalize` writes the output tail last

- Where: `pws_phoenix/model.py:113-125`, `:1139-1149`, `:1483`,
  `:1879-1890`
- Code:

      for inp in self.inputs_dict.values():
          inp.close()
      for f in self._opened_files:
          f.close()
      if self.output is not None:
          self.output.finalize()
      self._finalized = True

- Consequence: `__exit__` runs only for a model that was
  constructed. A raise in `__init__` after `_load_paths_to_data` (a
  missing parameter at `:715`, an unresolved input) leaves every
  file in `_opened_files` open; under MPI a raise after
  `open_writer` (`:1483`) also leaves the parallel NetCDF writer and
  `ds_input` open. In `finalize`, `Output.finalize()` writes the
  partial last chunk (2.2) after the closes: if a close raises, the
  tail is not written and `_finalized` stays False.
- Source: /code-review (a candidate it cut)
- Outcome: open

### 3.22 Arguments that are not validated fail late, with errors that do not name them

- Where: `pws_phoenix/model.py:184`, `:216`, `:732` with `:843-848`,
  `:751`, `:789`, `:962`, `:1280-1298`
- Code:

      if isinstance(val, pl.Path):

      grid = entry.get("discretization") or entry["class"].discretization

      real_dim = self.discretizations[grid].dims[0]
      ...
      mm.target_grid, target_ds.sizes.get(mm.target_grid)

      shape = tuple(grid_ds.sizes[dd] for dd in dims)

- Consequence:
  - A file given as a `str` is not opened (only `pl.Path` is); it
    reaches `Input` or the parameter lookup as a string and fails
    there.
  - A `"class"` given as its name fails with `'str' object has no
    attribute 'discretization'`. `Process._registry` is used only by
    `pws_phoenix/config.py`.
  - The grid's dim is `dims[0]` at `:732` and the grid's key at
    `:843-848`. A provided `Discretization(["nsegment"])` under the
    key `"segment"` (`pws_phoenix/config.py:114` takes `dims` from
    its own entry) fails the weights check with
    `(n_target, n_source) = (None, ...)`.
  - The first process on a grid with no parameter and no supplied or
    Map-fed input fails at `:789` with a `KeyError` that names only
    the dim.
  - `dt` is never compared with the time axis. `Time` is daily and
    the ported processes expect seconds; `model.run(dt=1.0, ...)`
    runs.
  - `convertible=inp.data.variable._in_memory` (`:751`) reads a
    private xarray attribute; the guard against converting a lazy
    input rests on it alone (the slice 1 note).
  - Under MPI, a process_dict in which no process is on `mpi_grid`
    (the processes say `"nhru"`, `control` leaves the default
    `"space"`) makes every process a replicated serial one. Any error
    comes from mpixarray's `parallelize` over a dim the file lacks;
    not checked.
  - `write_restart` names its files by date, so a second write to
    one directory adds files; `_read_restart` then raises "needs
    exactly one" and the MPI peek raises "inconsistent state times".
    Both are loud; neither says to use one directory per write.
- Source: /code-review (the grid key), checklist A
- Outcome: open

### 3.23 `ModelMPI` repeats parts of `Model` instead of sharing them, and the copies have drifted

- Where: `pws_phoenix/model.py:1240-1264` and `:79-124`;
  `:1879-1887` and `:1141-1148`; `:1766-1788` and `:1093-1116`;
  `:1560-1573` and `:438-446`
- Code:

      self.current_time_index = np.array([self._start_index], dtype=np.int32)
      self.current_time = np.array(

      self.current_time_index = np.zeros(1, dtype=np.int64)

- Consequence: `ModelMPI.__init__` sets the same attributes as
  `Model.__init__` by hand. `current_time_index` is int32 in serial
  and int64 under MPI; `current_time` exists in serial only;
  `ModelMPI` sets both `self._ntime` and `self.ntime`;
  `self.time_index` (`:942`) is set and never read.
  `_peek_restart_resume_index` repeats the last 24 lines of
  `_read_restart`, messages included. `finalize` and the serial-grid
  assembly are repeated the same way. `_set_budget_options` ends by
  calling `_set_units_options` (`:478`), so the units options are
  set only as a side effect of setting the budget options. A fix
  made in one copy (3.2, 3.4, 3.21) has to be made in the other.
- Source: /code-review (a candidate it cut)
- Outcome: open

### 3.24 Docstrings and comments that no longer match the code, old typing aliases, single-letter names

- Where: `pws_phoenix/model.py:1190`, `:1367`, `:1371`, `:1425`;
  `:1219-1224`; `:1169`, `:1177`, `:1556`, `:1612`; `:774-775`;
  `:703-705`; `:23`, `:69`; `:199-201`, `:1145`; `:433`
- Code:

      (time, mpi_grid); static params + ICs as
      (mpi_grid,))

      from typing import Any, Dict, List, Literal, Union

- Consequence:
  - Four places still put initial conditions ("ICs") in
    `input_file`; they are supplied in the process entry or come
    from a restart (`CLAUDE.md` "Initial state").
  - The `ModelMPI` docstring says the fix for a second streamed
    output variable was "fixed in mpixarray by Oct 28, 2026";
    `CLAUDE.md` says it is a patch in the local clone, not yet
    released (Nov 2026). One of the two is out of date.
  - "Phase 1 pipeline" and "Step B" name past stages of the build,
    not anything in the code.
  - `:774-775` speaks of "fill + optional initial"; `DataArrayMeta`
    no longer has `initial`.
  - `:703-705` cannot run: `_load_paths_to_data` has already
    replaced every `Path` (noted in 2.6).
  - `Dict`, `List`, `Union[bool, None]` against the rule to use
    Python 3.14 syntax; `v` and `f` against the rule on single-letter
    names; `_initialize_inputs_and_proceses` is misspelled and marked
    `(sic)`.
- Source: /code-review (candidates it cut)
- Outcome: open

## Checklist A, item by item

- Hidden copy: 3.3 (xarray alignment on assignment into the grid
  dataset). At build or restart only: `np.asarray(values)` (`:563`)
  reads a lazy supplied state; `.values.copy()` before `allgather`
  (`:1821`, see 3.14). The process_dict copy at `:79` and `:1240` is
  two levels of dict, no data.
- Allocation outside the three allowed classes: 3.14. Per step:
  `src[name].values[0, :]` (3.19). `current_time_index` and
  `current_time` are one-element arrays allocated once.
- `self._obj[name].values` inside a step method: none in a process.
  The MPI run loop does the same kind of lookup on `step` (3.19).
  The serial `step()` does none.
- A raise on one rank alone: 3.7, 3.8, and 2.5 (the rank-0 `Output`,
  `:1635-1651`; a failing `to_netcdf` in the rank-0 branch of
  `write_restart`, `:1852-1853`, leaves the others in `Barrier`).
  Checked and collective or the same on every rank: the raises of
  `_build` that depend only on `process_dict`, `control`, the
  declarations and file metadata (`:1287`, `:1306`, `:1402`, `:1417`,
  `:1472`, `:1609`, `:1616`, `:1630`), `_resolve_maps` (it is given
  the global size), the units checks (metadata only; `apply_affine`
  raises on dtype, not on values), `_raise_if_imbalanced`,
  `_raise_if_bad_initial_state`.
- A collective inside a rank-0 branch: none. The `allgather` calls of
  `write_restart` and its `Barrier` are outside the rank-0 branches.
- Serial / MPI difference beyond the table: 3.6, 3.9, 3.10, 3.13,
  3.17, 3.20, 3.23.
- An error that passes silently, or a check that cannot fail: 3.1,
  3.2, 3.3, 3.4, 3.5, 3.11, 3.12, 3.15, 3.16.

Checked and not a finding: `first_consumer = min(consumers, ...)`
(`:893`) cannot see an empty list, because the "unused" check at
`:855-864` raises first. Budgets are zeroed before a restart restores
them (`:549`, then `_read_restart`). A warm start skips the
required-state check (`:625`) and the restart's set comparison then
requires every restart variable (`:1061`). `parameters=` on the
distributed grid's `Discretization` raises (`:1283-1292`), which
answers the slice 2 note.

## Notes for later slices

- Slice 4 (`pws_phoenix/config.py`): `dims` and the grid key can
  differ (`:114`, see 3.22); does `load_model_yaml` reject unknown
  `control` keys (3.12)?
- Slices 7-12: every data-dependent raise in an `initialize()` or a
  kernel of an HRU-grid process is a raise on one rank under MPI
  (3.7); the list in 3.7 came from a scan for `raise` and is the
  starting point. Slice 12: the `hru_lat` guard is evaluated per
  rank.
- Slice 11 (`prms_snow.py`): the bool and int64 state is float64
  with a nan fill under MPI (3.9). Is every use correct for 0.0 /
  1.0, and is `newsnow` set before it is read on the first step?
- Slice 13 (`prms_stream_temp.py`): the hook state that 3.5 can skip.
- Slice 16: why no test caught 3.1 (a parameter with its dims in
  another order, or on another dim under MPI), 3.2 (two inputs with
  different time axes), 3.3 (arrays with different labels on the
  grid dim), 3.6 (`units_patch` with two declarers under MPI), 3.11
  (a supplied input ahead of its producer), 3.15. Does any test
  compare `input_spec` with what assembly resolves? Does any test
  check that a distributed-grid parameter is read-only (3.10)?
- Slice 17: why no test caught 3.4 (a restart written before the
  first step), 3.5 (a renamed process key), 3.8 (state of the wrong
  length, serial and MPI), 3.9 (the dtype of a restart variable
  written under MPI and read in serial).
- Slice 15 or the fix phase: an MPI job has no abort on an uncaught
  exception (3.7); `tests/run_tests.sh` runs `mpirun -n 4` without
  `python -m mpi4py`, so a lone raise in a test hangs rather than
  fails.
