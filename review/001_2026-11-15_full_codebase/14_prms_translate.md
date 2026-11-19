<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 14: the PRMS translation layer and `utils`](#slice-14-the-prms-translation-layer-and-utils)
  - [Findings](#findings)
    - [14.1 The units of the forcings never depend on `precip_units` or `temp_units`; metric precipitation raises only if some parameter in the file happens to carry the `precip_units` placeholder](#141-the-units-of-the-forcings-never-depend-on-precip_units-or-temp_units-metric-precipitation-raises-only-if-some-parameter-in-the-file-happens-to-carry-the-precip_units-placeholder)
    - [14.2 No forcing is checked against the control's time window; the Model then reads every input by position](#142-no-forcing-is-checked-against-the-controls-time-window-the-model-then-reads-every-input-by-position)
    - [14.3 Control entries that pyPRMS's metadata does not know are removed before the `dyn_*_flag` guard runs, and a known entry with trailing whitespace on its name line is removed with them](#143-control-entries-that-pyprmss-metadata-does-not-know-are-removed-before-the-dyn__flag-guard-runs-and-a-known-entry-with-trailing-whitespace-on-its-name-line-is-removed-with-them)
    - [14.4 `from_control` reads nine named flags and the `dyn_*_flag` family; every other control entry that selects physics is ignored, against the module's first promise](#144-from_control-reads-nine-named-flags-and-the-dyn__flag-family-every-other-control-entry-that-selects-physics-is-ignored-against-the-modules-first-promise)
    - [14.5 A parameter in the file that no selected class declares is dropped with no report, and only two of PRMS's initial-condition parameters are translated](#145-a-parameter-in-the-file-that-no-selected-class-declares-is-dropped-with-no-report-and-only-two-of-prmss-initial-condition-parameters-are-translated)
    - [14.6 `forward_fill` serves the first file row on model days before the first file date, and trusts the file dates to be sorted](#146-forward_fill-serves-the-first-file-row-on-model-days-before-the-first-file-date-and-trusts-the-file-dates-to-be-sorted)
    - [14.7 The header rule of the dynamic-parameter reader takes a line of ids that starts with a 4-digit number for the first data row](#147-the-header-rule-of-the-dynamic-parameter-reader-takes-a-line-of-ids-that-starts-with-a-4-digit-number-for-the-first-data-row)
    - [14.8 `verify_preprocessed` accepts an artifact that carries no stamps](#148-verify_preprocessed-accepts-an-artifact-that-carries-no-stamps)
    - [14.9 `mpi_model_from_control` rebuilds the whole serial kit on every rank, cannot be told where the forcing files are, and never compares `input_file` with the legacy files](#149-mpi_model_from_control-rebuilds-the-whole-serial-kit-on-every-rank-cannot-be-told-where-the-forcing-files-are-and-never-compares-input_file-with-the-legacy-files)
    - [14.10 Six arrays are built at full size in memory, with figures that hold on the test domains only](#1410-six-arrays-are-built-at-full-size-in-memory-with-figures-that-hold-on-the-test-domains-only)
    - [14.11 Entries of the units table that hide a flag, name no known parameter, or are missing](#1411-entries-of-the-units-table-that-hide-a-flag-name-no-known-parameter-or-are-missing)
    - [14.12 The names of PRMS's output request that the model cannot serve are recorded in one field in serial and thrown away under MPI](#1412-the-names-of-prmss-output-request-that-the-model-cannot-serve-are-recorded-in-one-field-in-serial-and-thrown-away-under-mpi)
    - [14.13 `download_addtl_domains` judges its state by whether the zip and the directory exist](#1413-download_addtl_domains-judges-its-state-by-whether-the-zip-and-the-directory-exist)
    - [14.14 Docstrings and comments that do not match the code; two constants and one field that nothing uses](#1414-docstrings-and-comments-that-do-not-match-the-code-two-constants-and-one-field-that-nothing-uses)
  - [Checklist D, item by item](#checklist-d-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 14: the PRMS translation layer and `utils`

- pws_phoenix commit: eaec75b (branch `review_001`; read from
  `.git/HEAD` and `.git/refs/heads/review_001` as text). The earlier
  slices record that the commits since e33e217 are review commits.
  The eleven files of this slice total 2,435 lines against the
  plan's 2,429; the difference was not explained (no git command was
  run).
- Paths: `pws_phoenix/prms_translate/` (`__init__.py`, `assemble.py`,
  `assemble_mpi.py`, `cbh.py`, `control.py`, `dyn_param.py`,
  `parameters.py`, `preprocess.py`, `prms_units.py`, `readers.py`),
  `pws_phoenix/utils/` (`__init__.py`, empty;
  `addtl_domain_files.py`)
- Checklist: D. Reviewed 2026-11-18.
- Method: `/code-review high` on the two directories (10 candidates).
  Kept: all 10, as 14.2 (the time window), 14.3 (two candidates on
  `_drop_unknown_control_entries`, merged), 14.4 (flags not read),
  14.6 and 14.7 (the dynamic-parameter reader), 14.8
  (`verify_preprocessed`), 14.9 (the MPI one-liner), 14.10
  (`np.tile`, widened to the other full-size arrays), 14.13 (the
  domain download). Added by the read: 14.1, 14.5, 14.11, 14.12,
  14.14. Then every file of the slice, whole, against checklist D.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/model.py:433-457`, `:567-587`, `:704-753`,
    `:931-943`; `pws_phoenix/data_io.py:41-101`;
    `pws_phoenix/maps.py:131-185`;
    `pws_phoenix/flow_graph_build.py:97-128`;
    `pws_phoenix/hydrology/prms_stream_temp.py:967-1048`,
    `:1523-1553`; findings 2.8, 3.1, 3.2, 3.3, 8.1, 12.2, 12.3, 13.1,
    13.8; the notes for slice 14 in the files of slices 5, 7, 8, 9,
    10, 11, 12, 13; `coverage_serial.txt:104-118`.
- Not read: pyPRMS (outside the project directory), so every
  statement about pyPRMS's metadata strings, its `Cbh` units, its
  reshape order and its control metadata comes from the comments in
  this slice. The PRMS Fortran. pywatershed (checklist D does not
  need it; its `PrmsDynamicParameter` is the reference the
  dynamic-parameter tests use and was not opened). Where a finding
  says what PRMS does, it says so from memory, and the fix phase must
  read the source.
- Nothing was run. Every consequence below is from reading.

## Findings

### 14.1 The units of the forcings never depend on `precip_units` or `temp_units`; metric precipitation raises only if some parameter in the file happens to carry the `precip_units` placeholder

- Where: `pws_phoenix/prms_translate/cbh.py:65-68`;
  `pws_phoenix/prms_translate/readers.py:468-478`;
  `pws_phoenix/prms_translate/prms_units.py:36-39`, `:127-131`,
  `:140-151`
- Code:

      # pyPRMS's CBH units ("in", "degree_fahrenheit") -> vocabulary key
      key = prms_units.resolve(str(da.name), da.attrs.get("units"))

      _PRECIP_UNITS = {0: "inches"}  # 1 = mm: the kernels are imperial

      text = prms_units.strip()
      if text in ("temp_units", "precip_units", "elev_units"):
          return _resolve_flag(text, flags)

- Consequence: `_resolve_flag` is the only place the three flag
  values are looked at, and it runs only for a parameter whose
  metadata units string IS the placeholder. `load_cbh` passes no
  flags, and the unit string it translates is the fixed one pyPRMS
  gives the variable. PRMS reads the CBH precipitation in
  `precip_units` and the CBH temperatures in `temp_units` (from
  memory of `climate_hru`; to check).
  - `precip_units = 1` (mm). The CBH precipitation is stamped
    `inches`. The run stops only if the parameter file contains at
    least one parameter with metadata units `precip_units`. If it
    contains none, the model is built and every precipitation value
    is read as inches, 25.4 times too large, with no message and
    with `control["units"] = "check"` passing. Which PRMS 5.2.1
    parameters carry that string, and whether an NHM parameter file
    has one: not determined. `CLAUDE.md` ("Units mode") says "Metric
    precipitation raises"; the code cannot promise that.
  - `temp_units = 1` (degC). The CBH temperatures are stamped
    `degF`. The run does stop, but through a parameter:
    `tmax_allsnow` and the others resolve to `degC`, their
    declarations say `degF`, and `package_parameters` raises
    (`parameters.py:106-113`). So the `temp_units` branch of
    `PRMSAtmosphere.initialize` (12.3) cannot be reached from a
    translated model.
  - A parameter file that omits a flag parameter (PRMS's default is
    0 for all three, from memory) raises `ValueError` at the first
    placeholder, for a parameter no process may declare
    (`translate_units` runs over every variable in the file).
  A direct check of the three flag values in `load_parameters`, and
  passing them to `load_cbh`, would make all three cases explicit.
- Source: checklist D
- Outcome: open

### 14.2 No forcing is checked against the control's time window; the Model then reads every input by position

- Where: `pws_phoenix/prms_translate/cbh.py:62-63`;
  `pws_phoenix/prms_translate/assemble.py:342-346`, `:369-372`;
  `pws_phoenix/model.py:938-943`; `pws_phoenix/data_io.py:79-81`;
  `pws_phoenix/prms_translate/assemble_mpi.py:76-77`
- Code:

      if start_time is not None or end_time is not None:
          da = da.sel(time=slice(start_time, end_time))

      process_dict[consumer][name] = load_cbh(
          cfg.cbh_paths[name], cfg.start_time, cfg.end_time
      )

  and in the Model:

      kk0 = list(self.inputs_dict.keys())[0]
      ...
      self._current[:] = self._data_values[self._current_index]

- Consequence: control window 1979-01-01 to 1980-12-31, and one CBH
  file (for example the OpenET file of `AET_cbh_file`) that starts
  1980-01-01. `sel` returns 366 rows with no error. The dynamic
  inputs and the tiled static inputs are built on the full 731-day
  `times`. In the Model, row 0 of the short file (1980-01-01) is
  served on model day 1979-01-01, and from day 367 on `Input.advance`
  raises a bare `IndexError`. If the short file is the first input,
  `model.ntime` is 366 and the run ends a year early with no
  message. A file with a gap inside the window shifts every later
  day the same way. This is finding 3.2 seen from the layer that has
  the window in hand: `assemble_from_control` builds `times` and
  could compare each input's `time` with it.
  Under MPI `write_mpi_input_file` does `merged[key] = val`, where
  xarray aligns by label (3.3): the short forcing would be padded
  with nan on the missing days instead of shifted. Not run.
- Source: /code-review, extended by the read
- Outcome: open

### 14.3 Control entries that pyPRMS's metadata does not know are removed before the `dyn_*_flag` guard runs, and a known entry with trailing whitespace on its name line is removed with them

- Where: `pws_phoenix/prms_translate/readers.py:317-321`,
  `:346-354`; `pws_phoenix/prms_translate/control.py:271-280`,
  `:163-174`
- Code:

      def flush() -> None:
          if header or not block or block[0] in known:
              kept.extend(block)
          else:
              dropped.append(block[0])

      print(
          f"{path.name}: dropping control entries unknown to the PRMS "
          f"{PRMS_VERSION} metadata: {dropped}"
      )

      flag_names = set(_FLAG_NAMES) | {
          str(nn)
          for nn in ctl.control_variables
          if str(nn).startswith("dyn_") and str(nn).endswith("_flag")
      }

- Consequence:
  - `control.py:105-106` says "Any OTHER dyn_*_flag set in a control
    raises in resolve_classes -- silence would drop a requested time
    variation". The guard sees only the flags that survive
    `load_control`. A `dyn_*_flag` outside the pyPRMS 5.2.1.1
    metadata (as `dyn_ag_frac_flag` was before it was added to
    `_AG_CONTROL_METADATA`) is removed, printed in a list with the
    harmless entries, and the model runs with the static parameter.
    The same holds for any physics flag pyPRMS does not know. The
    docstring's "nothing here reads them" is true only because the
    entry is gone.
  - The delimiter is tested with `line.strip() == "####"`, the name
    with `block[0] in known`, not stripped. A hand-edited control
    with `dprst_flag ` (a trailing blank or tab) loses the entry;
    `flags.get("dprst_flag", 0)` is then 0 and the NoDprst classes
    are selected. A blank line after `####` makes `block[0] == ""`,
    with the same result. Whether PRMS and pyPRMS accept such a
    line: not checked.
  - The notice is a `print`: under MPI every rank prints it, and a
    caller cannot test for it.
- Source: /code-review (two candidates, merged)
- Outcome: open

### 14.4 `from_control` reads nine named flags and the `dyn_*_flag` family; every other control entry that selects physics is ignored, against the module's first promise

- Where: `pws_phoenix/prms_translate/control.py:10-12`, `:91-101`,
  `:73`, `:243-244`, `:223-228`
- Code:

      Anything the ported classes cannot honor raises NotImplementedError
      NAMING the offending module/flag -- silence would mean silently
      running different physics than the control asked for.

      _FLAG_NAMES = (
          "dprst_flag",
          "stream_temp_flag",
          "stream_temp_shade_flag",
          "strmtemp_humidity_flag",
          "init_vars_from_file",
          "iter_aet_flag",
          "dyn_ag_frac_flag",
          "dyn_fallfrost_flag",
          "dyn_springfrost_flag",
      )

      shade = bool(flags.get("stream_temp_shade_flag", 0))
      humid = bool(flags.get("strmtemp_humidity_flag", 0))

- Consequence: the module slots are checked (`_require`), the flags
  are not. Names from memory of PRMS 5.2.1, to check against its
  control metadata:
  - `cascade_flag`, `cascadegw_flag` (cascades are not ported),
    `frozen_flag`, `glacier_flag`, `snarea_curve_flag`,
    `gwr_swale_flag`, `orad_flag`, `snow_cbh_flag`, the water-use
    `*_transferON_OFF` flags: a control that sets one builds the
    same model as a control that does not.
  - `model_mode` is not read. A GSFLOW control is translated as a
    PRMS-only model. Finding 8.1's `_intcp_changeover_in_net_rain`
    depends on this distinction, and `resolve_classes` returns
    classes only, so it has no way to set that flag (the answer to
    the note from slice 8).
  - `AET_module` and `PET_ag_module` are added to the metadata
    (`readers.py:199-208`) and never compared with the one value the
    port implements.
  - `bool()` turns a flag with more than two values into on / off.
    If `strmtemp_humidity_flag` has a third value (station data,
    from memory), it selects PRMSStreamTempSegHumidity.
  - `nhruOut_freq` and the output format entries are not read; the
    request is always served daily (14.12).
  - `_IGNORED_SLOTS = ("basin_module", "obs_module")` is defined and
    referenced nowhere in the slice; the slots are ignored because
    nothing reads them.
  A list of the control entries the layer knows to be harmless, and
  a raise or a returned list for the rest, would make the promise
  hold.
- Source: /code-review, extended by the read
- Outcome: open

### 14.5 A parameter in the file that no selected class declares is dropped with no report, and only two of PRMS's initial-condition parameters are translated

- Where: `pws_phoenix/prms_translate/parameters.py:88-100`,
  `:143-145`; `pws_phoenix/prms_translate/assemble.py:152-170`
- Code:

      for name, meta in _dict_of_kind(cls, "parameter").items():
          if name in DIS_OWNED_PARAMETERS:
              continue
          if name in extra:
              src = extra[name]
          elif name in params:
              src = params[name]

      if "gwstor_init" in params:
          state["gwres_stor"] = params["gwstor_init"]
      if "segment_flow_init" in params:
          state["seg_inflow"] = seg_inflow_from_segment_flow_init(

- Consequence: the split is driven by the declarations, which is the
  design. What is missing is the other half of the comparison:
  nothing lists the parameters of the file that reached no process.
  A user cannot tell a parameter that PRMS would not have used
  either from one whose effect is not ported. Examples from memory
  of PRMS 5.2.1, each to check against the ports: `snowpack_init`,
  `stream_tave_init`, `obsin_segment`, `gwsink_coef`. The layer also
  never looks at a parameter's VALUES for a case the ports do not
  cover: `hru_type` 0 and 2 (7.2), `seg_close` (13.1), `hru_deplcrv`
  against the number of curves in `snarea_curve` (11.5).
  `prms_initial_state` is "the one place that convention lives" and
  has two entries; `CLAUDE.md` lists the `*_init_frac` parameters and
  `dprst_frac_init` as fed from parameters by decision. Whether PRMS
  has others that a port neither declares nor receives here was not
  determined.
  The reverse case is loud: a declared parameter absent from the
  file raises `KeyError` (`parameters.py:96-100`), where PRMS would
  use its default. `params["seg_close"]` (`assemble.py:320`) raises
  a bare `KeyError`.
- Source: checklist D
- Outcome: open

### 14.6 `forward_fill` serves the first file row on model days before the first file date, and trusts the file dates to be sorted

- Where: `pws_phoenix/prms_translate/dyn_param.py:70-77`;
  `pws_phoenix/prms_translate/assemble.py:347-354`
- Code:

      """... Times before the first file date get the first row (the
      PRMS 'active from the start' convention for spin-up windows)."""
      ...
      idx = np.searchsorted(file_dates, times, side="right") - 1
      idx = np.clip(idx, 0, file_dates.size - 1)

- Consequence:
  - `dyn_ag_frac.param` with its first record at 1985-01-01 and a
    run that starts 1980-01-01: the 1985 row is served for 1980 to
    1984. PRMS, as recalled from `dynamic_param_read.f90`, replaces
    a parameter when the date of a record is reached, so it would
    use the parameter file's `ag_frac` for those five years.
    `assemble_from_control` has `params[name]` in hand (it copies
    the units from it at `:353-354`) and does not use the values.
    The docstring names a PRMS convention; its source is not given.
    The behaviour is pinned only against pywatershed's reader
    (module docstring, `:7-8`).
  - `np.searchsorted` needs sorted dates. A file with one record out
    of order, or the same date twice, gives a wrong row for some
    days with no error; nothing checks `file_dates`.
  - With `transp_module = transp_tindex` and a frost flag set,
    `resolve_classes` does not raise (`control.py:192` tests the
    module first), the file is read and filled (`assemble.py:347`),
    and nothing consumes it.
- Source: /code-review, extended by the read
- Outcome: open

### 14.7 The header rule of the dynamic-parameter reader takes a line of ids that starts with a 4-digit number for the first data row

- Where: `pws_phoenix/prms_translate/dyn_param.py:37-45`
- Code:

      if in_header:
          if parts[0] == "####":
              in_header = False
              continue
          # a row starting with a 4-digit year ends the header
          if not (parts[0].isdigit() and len(parts[0]) == 4):
              continue
          in_header = False
      yy, mm, dd = (int(pp) for pp in parts[:3])

- Consequence: the docstring says the header lines "typically" name
  the space ids. A header line `5307 5308 5309 ...` ahead of the
  `####` marker ends the header; `np.datetime64("5307-5308-5309")`
  then raises a `ValueError` that does not name the cause. With ids
  such as `1980 1 2 ...` the line is accepted as a data row, and the
  `####` line that follows fails at `int("####")`. The 4-digit rule
  is needed only for a file with no marker. `coverage_serial.txt:109`
  lists `:44` as never run: every test file ends its header with
  `####`, so the rule itself has no test.
- Source: /code-review
- Outcome: open

### 14.8 `verify_preprocessed` accepts an artifact that carries no stamps

- Where: `pws_phoenix/prms_translate/preprocess.py:173-203`;
  `pws_phoenix/prms_translate/assemble.py:206-207`, `:233-241`,
  `:285`, `:308-311`, `:335`
- Code:

      own = da.attrs.get("digest")
      if own is not None and digest_array(da.values) != own:
          raise ValueError(
      ...
      for key, stamped in da.attrs.items():
          if not key.startswith("derived_from_"):
              continue

- Consequence: the answer to the note from slice 7. A variable with
  no `digest` attribute skips the self-check, and a variable with no
  `derived_from_*` attributes runs the loop zero times. A file
  rewritten by a tool that drops variable attributes, or built by
  hand with the right variable names, verifies with no comparison.
  `assemble_from_control` then takes `segment_order`, the solar
  tables, `hru_in_to_cf` and every weights matrix from it. A wrong
  `segment_order` reaches the channel and stream-temperature kernels
  unchecked (7.5).
  Also not checked: that the expected variables are present (a
  missing `weights_flow` is a bare `KeyError` at `assemble.py:309`);
  that each `derivation` string is the one the current code would
  write; the version of the code that derived the artifact. A
  matrix computed by an earlier `derive_aggregation_weights` from
  the same inputs verifies.
- Source: /code-review; slice 7 note
- Outcome: open

### 14.9 `mpi_model_from_control` rebuilds the whole serial kit on every rank, cannot be told where the forcing files are, and never compares `input_file` with the legacy files

- Where: `pws_phoenix/prms_translate/assemble_mpi.py:116`,
  `:120-134`, `:140`; `:62-90`
- Code:

      kit = assemble_from_control(control_file)

      process_dict[slot] = {
          "class": entry["class"],
          "discretization": mpi_grid,
      } | {
          name: entry[name]
          for name in entry["class"].get_restart_variables()
          if name in entry
      }

- Consequence:
  - No `cbh_paths` and no `preprocessed` argument. A control whose
    `precip_day` names a file the domain does not ship (the case
    `cbh_paths=` exists for) works in the prep half and raises
    file-not-found on every rank here.
  - When the files exist, every rank parses every ASCII CBH file for
    the whole window, computes the solar tables for the full domain
    and builds the dense weights, then discards the distributed
    grid's data in the lines quoted. The cost and the memory of the
    serial load are paid once per rank, for data `input_file`
    already holds.
  - `input_file` carries no stamp. Classes, Maps, serial-grid
    parameters and initial state come from the legacy files as they
    are now; the distributed grid's parameters and forcings come
    from a file written earlier, perhaps from other parameter files
    or with `n_days`. Nothing compares the two, while
    `preprocess.py` has the digest design for exactly this.
  - `write_mpi_input_file` keeps the first of two same-named
    parameters (`compat="override"`) and the first of two same-named
    forcings (`key not in merged`) without comparing them; the
    comment gives the reason they are equal for a kit from
    `assemble_from_control`, and the kit is documented as open for
    modification.
  `coverage_serial.txt:106`: 12% (the serial run cannot reach it).
- Source: /code-review, extended by the read
- Outcome: open

### 14.10 Six arrays are built at full size in memory, with figures that hold on the test domains only

- Where: `pws_phoenix/prms_translate/parameters.py:148-157`;
  `pws_phoenix/prms_translate/assemble.py:378-383`;
  `pws_phoenix/prms_translate/dyn_param.py:80`;
  `pws_phoenix/prms_translate/cbh.py:64`;
  `pws_phoenix/prms_translate/assemble_mpi.py:62-89`;
  `pws_phoenix/prms_translate/preprocess.py:125-130`
- Code:

      weights = np.zeros((n_seg, hru_segment.shape[0]))
      for ii in range(hru_segment.shape[0]):
          if hru_segment[ii] > 0:
              weights[hru_segment[ii] - 1, ii] = 1.0

      np.tile(params[name].values, (times.size, 1)),

      da = da.astype(np.float64)

- Consequence: on drb (765 HRUs, 456 segments, 731 days) all of
  these are a few MB. On the domain now being translated (114,958
  HRUs, 47 years = about 17,167 days; about 57,000 segments assumed,
  as in 13.8):
  - `volume_map_weights` is 114,958 x 57,000 x 8 bytes = 52 GB for a
    matrix with one nonzero per column, and `insert_flow_graph` pads
    a copy (`flow_graph_build.py:124-127`). `write_preprocessed`
    writes it dense. See 2.8 and 13.8.
  - Each CBH variable is 15.8 GB after `astype(np.float64)`, with
    the float32 original alive beside it during the call; three or
    four variables are loaded.
  - A static parameter served as an input (`ag_frac` without its
    dynamic flag) is 15.8 GB of identical rows from `np.tile`, and
    `forward_fill`'s `dyn.values[idx, :]` is the same size. A
    read-only `np.broadcast_to` view would serve the tiled case with
    no allocation.
  - `write_mpi_input_file` merges every parameter and forcing into
    one in-memory dataset before `to_netcdf`.
  The module docstrings call these "load-time prep allocations"
  outside the model, which answers the memory directive on drb. The
  tile at `assemble.py:379` has no such comment. The serial arrays
  ARE the model's input buffers for the whole run (the Model shares
  them by reference).
- Source: /code-review (the tile), extended by the read
- Outcome: open

### 14.11 Entries of the units table that hide a flag, name no known parameter, or are missing

- Where: `pws_phoenix/prms_translate/prms_units.py:45-72`,
  `:122-123`; `pws_phoenix/prms_translate/readers.py:74-94`
- Code:

      "radadj_slope": "dday/delta_degF",  # PRMS: "dday/temp_units"
      ...
      "tsi": "hours",
      "lso": "days",

      if name in PRMS_UNITS_BY_NAME:
          return PRMS_UNITS_BY_NAME[name]

- Consequence:
  1. `radadj_slope` is fixed at `dday/delta_degF` by name, though
     its PRMS string is the placeholder that `resolve` handles at
     `:129-130`. With `temp_units = 1` it is stamped per degF while
     `dday_slope` in the same file resolves per degC. The run stops
     anyway (14.1), so the entry is wrong without a consequence
     today.
  2. `tsi` and `lso` have no `# PRMS:` comment, and CodeGraph finds
     no `kind="parameter"` declaration of either name. If they are
     internal fields, no parameter file can contain them and the
     entries never apply. Not confirmed by a text scan.
  3. The answer to the note from slice 12: `tmax_cbh_adj`,
     `tmin_cbh_adj` and `tmax_allrain_offset` have no entry by name,
     so they take whatever `temp_units` resolves to, an absolute
     `degF`. That matches their declarations (12.2) and is why no
     test fails. Fixing 12.2 needs three by-name entries and a
     `delta_degF` key in the vocabulary. Their PRMS strings were not
     read.
  4. `_AG_PARAMETER_UNITS` is said to "feed translate_units through
     PRMS_UNITS_BY_STRING unchanged". The value `"fraction"` is not
     a key of that table; `resolve` returns None and the caller
     keeps the raw string, which is already the vocabulary key. The
     result is right by a different path than the comment gives.
     `soilzone_aet_converge` is in both tables, with the by-name
     comment `# PRMS: "unknown"` beside an injected `"inches"`.
  5. A disagreement between the table and a declaration raises in
     `package_parameters` only for a class that is packaged.
     `assemble_from_control` never packages
     PRMSStreamTempConstantShade, PRMSStreamTempSegHumidity or a
     hydraulic-geometry class with depth, so `depth_alpha`,
     `depth_m`, `seg_humidity`, `segshade_sum` and `segshade_win`
     are judged by no run of the layer. `seg_humidity` is declared
     `fraction` (`prms_stream_temp.py:1538-1543`); its PRMS string
     was not read.
  6. `agree` accepts `fraction` against `none` in both directions,
     by design (`:154-161`); a declared `fraction` therefore cannot
     catch a PRMS parameter that is a percent labelled "none".
- Source: checklist D; slice 12 and 13 notes
- Outcome: open

### 14.12 The names of PRMS's output request that the model cannot serve are recorded in one field in serial and thrown away under MPI

- Where: `pws_phoenix/prms_translate/assemble.py:398-407`, `:415`;
  `pws_phoenix/prms_translate/assemble_mpi.py:139-145`;
  `pws_phoenix/prms_translate/control.py:316-325`
- Code:

      kept, _dropped = requested_output_names(kit.config, kit.process_dict)
      control["output_var_names"] = [
          nn
          for nn in kept
          if nn in mpi_vars or "output_serial_zarr" in control
      ]

- Consequence: text of the module docstring: "Dropped names are not
  reported here". In serial the list is in
  `ModelKit.dropped_output_var_names`, which `model_from_control`
  never returns. Under MPI a serial-grid name is also left out when
  no zarr store was given, in the same silence. Without
  `output_serial_zarr` in the caller's control the whole request is
  ignored (documented). Only `nhruOutVar_names` and
  `nsegmentOutVar_names` are read; `nhruOut_freq` (monthly or yearly
  output, from memory) and the other output families are not, so a
  request for monthly means is served as daily values. A user who
  compares the PRMS output list with the pws_phoenix store has to
  find the difference by hand.
- Source: checklist D
- Outcome: open

### 14.13 `download_addtl_domains` judges its state by whether the zip and the directory exist

- Where: `pws_phoenix/utils/addtl_domain_files.py:21`, `:61-65`,
  `:85-88`
- Code:

      if not ADDTL_DOMAINS_ZIP.exists():
          _fetch_zip()
      if not ADDTL_DOMAINS_DIR.exists():
          _extract_zip()

- Consequence:
  - The zip deleted after extraction (it is the larger copy): every
    later call downloads it again, or raises `URLError` offline
    though the domains are complete.
  - `extractall` interrupted: the next call finds a good zip and an
    existing directory, skips the extraction, and returns a
    directory with domains missing. Only `--force` recovers.
  - `DATA_DIR = pl.Path(__file__).parents[1] / "data"` is inside the
    package. For an install that is not editable this is
    `site-packages`, which may be read-only, and the data are
    removed with the package.
  - `assert ADDTL_DOMAINS_DIR.exists()` is the only check of the
    archive's layout and disappears under `python -O`.
  `coverage_serial.txt:118`: 24%; no test runs any function of the
  module (it needs the network).
- Source: /code-review, extended by the read
- Outcome: open

### 14.14 Docstrings and comments that do not match the code; two constants and one field that nothing uses

- Where: `pws_phoenix/prms_translate/__init__.py:18-24`;
  `pws_phoenix/prms_translate/readers.py:3-49`, `:74-79`,
  `:300-303`, `:250-256`;
  `pws_phoenix/prms_translate/parameters.py:33`;
  `pws_phoenix/prms_translate/control.py:73`, `:132`;
  `pws_phoenix/prms_translate/assemble.py:5`, `:10`
- Code:

      # GSFLOW agricultural parameters absent from pyPRMS's master metadata
      # (ask #3 above): name -> (datatype, dimension, description);
      # dims/dtypes verified against the fgr_ag_2yr parameter file blocks
      # units of the injected ag parameters, in pws_phoenix vocabulary keys
      ...
      _AG_PARAMETER_UNITS = {

      # ask #6: force_default on the *_dynamic path entries shadows the
      # control file's actual values with the metadata defaults
      for entry in md["control"].values():
          entry.pop("force_default", None)

- Consequence, text and dead code only:
  1. `__init__.py` lists two upstream asks, numbered 1 and 2.
     `readers.py` lists five, numbered #1, #3, #4, #5, #6, and
     `dyn_param.py` calls its reader ask #2.
  2. At `readers.py:74-79` the comment for `_AG_PARAMETER_METADATA`
     runs into the comment for `_AG_PARAMETER_UNITS` and sits above
     the wrong dict.
  3. The comment says `force_default` is stripped from the
     `*_dynamic` entries; the loop strips it from every control
     entry. What other entries carry it: not read.
  4. The docstring of `prms_metadata` does not mention the control
     entries it adds or the `force_default` change.
  5. `COMPUTED_PARAMETERS` (`parameters.py:33`) and `_IGNORED_SLOTS`
     (`control.py:73`) are referenced nowhere in the slice.
     `PrmsRunConfig.dynamic_parameters` is `sorted(dyn_param_paths)`
     and nothing in the slice reads it.
  6. The module docstring of `assemble.py` gives both entry points
     with two arguments; each has four.
  7. `prms_metadata()` parses pyPRMS's metadata anew on every call:
     once per control file, once per parameter file (twice for a
     partial one) and once per CBH file. Cost only.
  8. `package_parameters` tests uniformity with `vals ==
     vals.flat[0]`; an all-nan field is reported as "spatially
     VARYING".
- Source: checklist D; /code-review (item 7 was below its cap)
- Outcome: open

## Checklist D, item by item

- A PRMS value that is changed without a raise or a message:
  - `seg_inflow_from_segment_flow_init` assigns at a confluence (the
    last upstream segment wins, not the sum). pywatershed's
    behaviour, kept and commented (`assemble.py:133-137`); no
    message at run time.
  - `forward_fill` before the first file date (14.6).
  - A declared scalar that PRMS dimensions by HRU is collapsed to
    its first value only if the field is uniform; otherwise it
    raises (`parameters.py:117-133`). This is all that stands
    between a per-HRU `den_max` and element 0 (the note from slice
    11).
  - Exact changes of representation, each commented: float32 to
    float64 for CBH, int32 to int64, `(nhru, nmonths)` to
    `(nmonths, nhru)`.
  - Passed through unchanged: `seg_close` (13.1), -1.0 in the OpenET
    file (the note from slice 10: nothing compares it with
    `ag_frac`, so 10.7 is reachable whenever the file has -1.0 on an
    agricultural HRU), `pref_flow_infil_frac` (the note from slice
    9: the layer fills no parameter from another; `hru_in_to_cf` is
    the only derived one).
- A PRMS value that is dropped without a raise or a message: 14.3,
  14.4, 14.5, 14.12. Dropped with a documented reason:
  `PET_cbh_file`, the output paths.
- An entry of the units table that disagrees with the declarations:
  none for the classes `assemble_from_control` packages. A
  disagreement raises (`parameters.py:106-113`), and
  `coverage_serial.txt:110` lists `:107` as never run, so no test
  domain has one. Outside that guarantee: 14.1, 14.11.
- Checked and not a finding:
  - The dims of a parameter are renamed by position
    (`parameters.py:140-141`), after a comparison of rank only. A
    source on the wrong dim gets the declared name; the size
    conflict then raises when the slot's dataset is built, naming
    the variable but not the cause. The Model compares no dims
    (3.1).
  - A PRMS parameter given with fewer dimensions than its
    declaration raises (rank, or the size conflict); it is not
    broadcast.
  - Whether a monthly row of a transposed 2-D parameter is
    contiguous in memory depends on pyPRMS's reshape order
    (`readers.py:409-411`); not read. If the rows are strided, each
    kernel gets a second compiled signature and reads every twelfth
    element.
  - The frost flags (the note from slice 12): one flag alone raises
    under `transp_frost` (`control.py:192-197`), so no static date
    is tiled. The dates are parsed as float64.
  - `insert_flow_graph` (the note from slice 5): `n_substeps`
    defaults to 24 and is passed through. A new node's nan initial
    state is filled from the channel's padded zeros only for a name
    the channel entry also supplied (`assemble.py:520-523`);
    `coverage_serial.txt:105` lists `:522` as never run.

## Notes for later slices

- Slice 15 (harness): nothing from this slice.
- Slice 16: 14.2 is 3.2 from the translation side; a test of 3.2
  with one short input covers both.
- Slice 22 (translation tests), from `coverage_serial.txt:104-118`.
  Never run:
  - `assemble_mpi.py` (12%; read it with
    `tests/test_prms_translate*_mpi.py`);
  - every raise of `package_parameters` (`:82`, `:107`, `:124`,
    `:135`);
  - in `dyn_param.py` the 4-digit header rule (`:44`), a blank line
    (`:36`), and both raises (`:50`, `:58`);
  - the multi-variable CBH raise (`cbh.py:54`);
  - the `init_vars_from_file` raise (`control.py:282`);
  - in `assemble.py` the missing-humidity raise (`:220`), the
    no-source raise (`:385`), the missing `nhm_seg` raise (`:479`),
    the fill of a new node's state (`:522`), a Map that does not
    target `nsegment` (`:532-533`), and a parameter file without
    `gwstor_init` or `segment_flow_init` (`164->166`, `166->170`);
  - the missing-input raise of `verify_preprocessed`
    (`preprocess.py:191`).
  Ask of each test file: is there a control with a flag the layer
  does not read (14.4), with an entry pyPRMS does not know (14.3),
  with `precip_units = 1` (14.1)? A CBH file shorter than the
  window (14.2)? A dynamic-parameter file whose first date is after
  the start, or out of order (14.6)? An artifact file with its attrs
  removed (14.8)? Is the reader compared with anything but
  pywatershed's (14.6)? `readers.py` has 100% coverage and 14.3 was
  not caught: no test control has an unknown `dyn_*_flag` or a name
  line with trailing whitespace.
- Fix phase, to read before fixing: pyPRMS's parameter metadata for
  the strings `precip_units`, `temp_units`, `elev_units` (14.1) and
  for `tmax_cbh_adj`, `tmin_cbh_adj`, `tmax_allrain_offset`,
  `transp_tmax`, `seg_humidity`, `segshade_sum`, `segshade_win`,
  `gw_tau`, `ss_tau` (14.11); its control metadata for the flags of
  14.4 and for `force_default` (14.14); its reshape order
  (checklist section); `climate_hru.f90` (14.1) and
  `dynamic_param_read.f90` (14.6).
- Fix phase, runs that confirm before fixing: `load_parameters` on
  the drb file with `precip_units` edited to 1 (14.1);
  `model_from_control` with one CBH file cut to the second year
  (14.2); `load_control` on a control with `dprst_flag ` (14.3);
  `verify_preprocessed` on an artifact file rewritten without attrs
  (14.8); peak memory of `assemble_from_control` on the large domain
  (14.10).
