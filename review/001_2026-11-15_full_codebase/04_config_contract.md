<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 4: the yaml loader and the contract graph](#slice-4-the-yaml-loader-and-the-contract-graph)
  - [Findings](#findings)
    - [4.1 `restart_read` is not resolved relative to the yaml file](#41-restart_read-is-not-resolved-relative-to-the-yaml-file)
    - [4.2 A key the loader does not know is dropped without a message; a map's `derivation` cannot be given at all](#42-a-key-the-loader-does-not-know-is-dropped-without-a-message-a-maps-derivation-cannot-be-given-at-all)
    - [4.3 A key written twice in the yaml is accepted; the last value wins](#43-a-key-written-twice-in-the-yaml-is-accepted-the-last-value-wins)
    - [4.4 The discretizations the loader returns cannot be passed to `ModelMPI` when the yaml lists the distributed grid](#44-the-discretizations-the-loader-returns-cannot-be-passed-to-modelmpi-when-the-yaml-lists-the-distributed-grid)
    - [4.5 A discretization entry's `dims` is not compared with its key, and a string is split into letters](#45-a-discretization-entrys-dims-is-not-compared-with-its-key-and-a-string-is-split-into-letters)
    - [4.6 `_load_weights` reads the weights file once per map entry, so maps that share a file each hold their own dense matrix](#46-_load_weights-reads-the-weights-file-once-per-map-entry-so-maps-that-share-a-file-each-hold-their-own-dense-matrix)
    - [4.7 The graph finds a Map's producer by a rule that is not the Model's, and writes `(grid)` as a node when it finds none](#47-the-graph-finds-a-maps-producer-by-a-rule-that-is-not-the-models-and-writes-grid-as-a-node-when-it-finds-none)
    - [4.8 Process, grid and input names are written into dot and mermaid without quoting or escaping](#48-process-grid-and-input-names-are-written-into-dot-and-mermaid-without-quoting-or-escaping)
    - [4.9 `to_dot` raises a bare `KeyError` for a Map whose target grid hosts no process](#49-to_dot-raises-a-bare-keyerror-for-a-map-whose-target-grid-hosts-no-process)
    - [4.10 A time-varying parameter is recognized by two dim names only](#410-a-time-varying-parameter-is-recognized-by-two-dim-names-only)
    - [4.11 The graph classifies parameters itself instead of reading `input_spec`, and the two can disagree](#411-the-graph-classifies-parameters-itself-instead-of-reading-input_spec-and-the-two-can-disagree)
    - [4.12 Yaml content that is not validated fails late, with errors that do not name it](#412-yaml-content-that-is-not-validated-fails-late-with-errors-that-do-not-name-it)
  - [Checklist A, item by item](#checklist-a-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 4: the yaml loader and the contract graph

- pws_phoenix commit: 7312db3 (branch `review_001`; the commits since
  e33e217 are review commits).
- Paths: `pws_phoenix/{config,model_contract_graph}.py`
- Checklist: A. Reviewed 2026-11-15.
- Method: `/code-review high` on the two files, then a read of each
  file against checklist A, with the code they call
  (`Model.input_spec`, `Model._resolve_grid`, `Model._resolve_maps`,
  the `Discretization` and `Map` constructors) and their callers
  (`tests/test_two_grid_mpi.py`, `tests/test_model_contract_graph.py`,
  `examples/00_input_contract.py`,
  `examples_from_pws/01_multi-process_models.py`). The session
  executed no project code; it used text scans (grep) only. A finding
  marked "not run" rests on reading how PyYAML, graphviz or mermaid
  behaves and needs a short run to confirm before it is fixed.

Neither file runs inside a time step. `config.py` decides what the
Model is built from, so its findings come first (4.1-4.6, ordered by
consequence); the graph only draws, so its findings (4.7-4.11) are
wrong or unusable pictures. 4.12 collects the late, unnamed errors.

## Findings

### 4.1 `restart_read` is not resolved relative to the yaml file

- Where: `pws_phoenix/config.py:80-84`, `:155-157`
- Code:

      _CONTROL_PATH_KEYS = (
          "input_file",
          "output_parallel_netcdf",
          "output_serial_zarr",
      )

- Consequence: the module docstring says "every path string is
  resolved RELATIVE TO THE YAML FILE's directory". A yaml with
  `control: {restart_read: restart_dir}`, loaded by a run started in
  another directory, reaches `self._read_restart(pl.Path(restart_read))`
  (`pws_phoenix/model.py:138`, `:1349`) relative to the working
  directory. The run raises "restart_read: no restart files found",
  or, when the working directory holds a directory of that name,
  warm-starts from that other state.
- Source: /code-review
- Outcome: open

### 4.2 A key the loader does not know is dropped without a message; a map's `derivation` cannot be given at all

- Where: `pws_phoenix/config.py:111-113`, `:154`, `:166-170`, and the
  top level (`:136`, `:154`, `:160`, `:107`)
- Code:

      for key in ("topo_order", "topo_one_based"):
          if key in entry:
              kwargs[key] = entry[key]
      ...
      maps[map_name] = _MAP_CLASSES[entry["class"]](
          weights=_load_weights(entry["weights"], base_dir),
          grid=dict(entry["grid"]),
          variable=dict(entry["variable"]),
      )

- Consequence:
  - A discretization entry with `topo_one_base: false` (misspelled)
    is built with the default `topo_one_based=True`. On 0-based
    connectivity that gives another order, and `topological_order`
    checks nothing about the values (2.7).
  - `Map.__init__` takes `derivation=` (`pws_phoenix/maps.py:108`);
    the loader never passes it. For every model loaded from yaml,
    `Model.input_spec(...)["maps"][name]["derivation"]` is None and
    the graph's "N with a known derivation" is absent.
  - `control` is copied whole (`dict(cfg["control"])`), so an unknown
    key reaches the Model, which ignores it (3.12). This answers the
    slice 3 note: the loader does not reject unknown `control` keys.
  - A misspelled top-level section (`map:`, `discretisations:`) is
    read as absent. The Model then builds without Maps or with
    default grids; whether that raises depends on the model.
- Source: /code-review, checklist A (an error that passes silently)
- Outcome: open

### 4.3 A key written twice in the yaml is accepted; the last value wins

- Where: `pws_phoenix/config.py:133`
- Code:

      cfg = yaml.safe_load(ff)

- Consequence: PyYAML does not raise on a duplicate mapping key. Two
  entries under `processes:` with the same name give one process: the
  second entry's content at the first entry's place in the schedule.
  The same holds for two maps of one name and for a `control` key
  given twice. A process dropped this way may raise at assembly (an
  unresolved input) or may not (its output was not needed, or is
  also supplied). Not run.
- Source: checklist A (an error that passes silently)
- Outcome: open

### 4.4 The discretizations the loader returns cannot be passed to `ModelMPI` when the yaml lists the distributed grid

- Where: `pws_phoenix/config.py:107-115`, with
  `pws_phoenix/model.py:1283-1292`
- Code:

      for grid, entry in cfg.get("discretizations", {}).items():
          ...
          discretizations[grid] = Discretization(list(entry["dims"]), **kwargs)

- Consequence: the docstring says the returned tuple "feeds `Model`
  or `ModelMPI` equally", and its own schema has
  `discretizations: hru: dims: [hru]` beside `control: mpi_grid: hru`.
  That tuple, passed to `ModelMPI`, raises "discretizations ['hru']:
  pass Discretizations for SERIAL grids only". The raise is the same
  on every rank. One yaml therefore serves both models only if the
  caller removes the distributed grid's entry by hand.
  `tests/test_two_grid_mpi.py:179` discards the fourth return value
  (`process_dict, control, maps, _ = load_model_yaml(`), and its yaml
  has no `discretizations:` section, so no test passes loaded
  discretizations to either model.
- Source: /code-review; checklist A (a serial / MPI difference beyond
  the table)
- Outcome: open

### 4.5 A discretization entry's `dims` is not compared with its key, and a string is split into letters

- Where: `pws_phoenix/config.py:114`
- Code:

      discretizations[grid] = Discretization(list(entry["dims"]), **kwargs)

- Consequence:
  - `segment: {dims: [nsegment]}` loads. The Model takes the grid's
    dim from `dims[0]` in one place and from the key in another
    (3.22), and fails at the Map weights check with
    `(n_target, n_source) = (None, ...)`.
  - `hru: {dims: hru}` (no brackets) gives `list("hru")` =
    `['h', 'r', 'u']`. The Model's `real_dim` is then `'h'`
    (`pws_phoenix/model.py:732`): an input raises "grid 'hru' expects
    ('h',)", and a grid with no supplied input fails later at
    `grid_ds.sizes[dd]` (`:789`). Not run.
  - More than one dim is accepted; the Model reads only `dims[0]`.
- Source: /code-review (the key), checklist A
- Outcome: open

### 4.6 `_load_weights` reads the weights file once per map entry, so maps that share a file each hold their own dense matrix

- Where: `pws_phoenix/config.py:97-99`, `:167`
- Code:

      with xr.open_dataarray(_resolve_path(spec, base_dir)) as da:
          return da.load().values
      ...
      weights=_load_weights(entry["weights"], base_dir),

- Consequence: the yaml written by
  `examples_from_pws/01_multi-process_models.py:591-604` has 13 maps
  over 5 weights files. The loader reads 13 files and holds 13 dense
  `(nsegment, nhru)` matrices where 5 would do; the in-code
  `make_maps(weights)` of the same example passes one array to every
  map of a family. Each matrix then stays in memory after `bind`
  (2.8), on every rank under MPI. The loader also discards the
  file's dim names, so a matrix stored as `(source, target)` is
  caught only by the shape check of `_resolve_maps`, which cannot see
  it when the two grids have the same size.
- Source: /code-review; checklist A (a hidden copy)
- Outcome: open

### 4.7 The graph finds a Map's producer by a rule that is not the Model's, and writes `(grid)` as a node when it finds none

- Where: `pws_phoenix/model_contract_graph.py:177-188`, `:200`,
  `:313`, `:417-421`
- Code:

      if var in cls.get_var_names(
          include_budget=True
      ) or var in tuple(cls.get_parameters_internal()) + tuple(
          cls.get_inputs()
      ):
          return slot
      return f"({grid})"

- Consequence:
  - `get_inputs()` does not include `kind="mutable_input"`. In the
    sub-model of `examples_from_pws/01_multi-process_models.py:761`
    (soilzone, groundwater, channel, with the three volume Maps),
    `sroff_vol` on `nhru` is declared only by `PRMSSoilzone`, as a
    mutable input (`pws_phoenix/hydrology/prms_soilzone.py:630-631`).
    `_producer_of` returns `"(nhru)"`, and `to_dot()` writes
    `(nhru) -> prms_channel [label="sroff_vol>seg_sroff_vol",
    style=dashed];`, which is not valid dot; `to_mermaid()` writes
    `(nhru) -. "..." .-> prms_channel`, which is not valid mermaid.
    The example draws exactly this graph at `:823-829`. Not run. The
    same happens for a Map whose source is a supplied parameter,
    which `Model._resolve_maps` accepts (its only test is
    `mm.source_var not in source_ds`).
  - The rule returns the FIRST process in schedule order that
    declares the name as a variable OR as an input. A process that
    only reads the variable, placed ahead of its owner (a read of
    the prior step's value), is drawn as the producer.
    `Model._resolve_maps` finds the owner with `get_var_names()`
    alone (`pws_phoenix/model.py:868-872`). No NHM Map source has a
    reader ahead of its owner today (text scan of the declarations),
    so this needs a new configuration to show.
  - `return f"({grid})"` (`:188`) is reached by no serial test
    (`coverage_serial.txt`).
- Source: /code-review
- Outcome: open

### 4.8 Process, grid and input names are written into dot and mermaid without quoting or escaping

- Where: `pws_phoenix/model_contract_graph.py:288`, `:294`, `:299`,
  `:363`, `:371`, `:383`, `:242`
- Code:

      out.append(f'    subgraph {grid}["grid: {grid}"]')
      out.append(f'        {slot}["{label}"]')
      out.append(f'    ext_{name}(["{name}"])')
      ...
      out.append(f"    subgraph cluster_{grid} {{")
      f"        {slot} [shape=plain, "

- Consequence: a process key is free text chosen by the user. Not
  run:
  - A key that is not a bare identifier (`prms-snow`, `2nd_channel`,
    a key with a space) is a syntax error in both outputs.
  - A key that is a keyword (`graph`, `node`, `edge` in dot; `end`,
    `graph` in mermaid) is read as a statement, not as a node.
  - A key equal to a grid's name, or equal to `time`, shares one id
    with the mermaid subgraph or with the Time node.
  - `ext_{name}` is one id per input name, so an external input of
    the same name on two grids is drawn as one node feeding both
    grids; they are two arrays to supply.
  - `<`, `>` or `&` in a key goes unescaped into the HTML-like table
    label (`:242`).
- Source: /code-review
- Outcome: open

### 4.9 `to_dot` raises a bare `KeyError` for a Map whose target grid hosts no process

- Where: `pws_phoenix/model_contract_graph.py:401`, `:309`
- Code:

      first = self.grids[tgt_grid][0][0]
      ...
      out.append(f"    {node} -.- {tgt_grid}")

- Consequence: the graph is built from `Model.input_spec`, which
  does not check the Maps against the processes. A `process_dict` of
  `nhru` processes only, with the full `maps` dict, constructs;
  `to_dot()` then raises `KeyError: 'nsegment'`, and `to_mermaid()`
  writes an edge to a new node named `nsegment`. The Model itself
  raises on an unused Map with a message that names it; the graph
  does not say which Map is the cause. The example avoids this by
  subsetting the Maps by hand (`:791-793`).
- Source: /code-review
- Outcome: open

### 4.10 A time-varying parameter is recognized by two dim names only

- Where: `pws_phoenix/model_contract_graph.py:66`, `:124`
- Code:

      _CYCLIC_DIMS = ("nmonth", "ndoy")
      ...
      elif any(dd in _CYCLIC_DIMS for dd in meta.dims):

- Consequence: `CLAUDE.md` defines a time-varying parameter as one
  with "a time axis that is NOT model time, such as `month`". The toy
  `Upper.param_up_1` is declared `dims=("month", "space")`
  (`pws_phoenix/processes_concrete.py:38`); the graph counts it as
  static. No other module holds such a list (the Model only rejects
  a parameter on `time`, `pws_phoenix/model.py:1303-1312`), so any
  new axis name (a per-year parameter, on the backlog) is static in
  the picture until this tuple is edited.
- Source: /code-review
- Outcome: open

### 4.11 The graph classifies parameters itself instead of reading `input_spec`, and the two can disagree

- Where: `pws_phoenix/model_contract_graph.py:115-132`, `:94-100`
- Code:

      for name, meta in _dict_of_kind(
          entry["class"], "parameter"
      ).items():
          if meta.derivation is not None:
              derivable.append(name)

- Consequence:
  - `input_spec` treats a name as derivable when ANY process on the
    grid declares a derivation for it (`pws_phoenix/model.py:357-365`).
    The graph asks each class alone, so a parameter declared with a
    derivation by one process and without by another is "derivable"
    in one node and "static" in the other.
  - A parameter shared by several processes, or owned by the grid,
    is counted in every node that declares it. The counts in the
    nodes add up to more than the number of arrays to supply;
    `spec["required"][grid]["parameters"]` has the number.
  - `len(entry["class"].get_parameters())` is stored in `self.grids`
    and read nowhere (`_n_params` in both renderers).
  - The graph imports the private `_dict_of_kind` from
    `pws_phoenix/process.py`.
- Source: /code-review
- Outcome: open

### 4.12 Yaml content that is not validated fails late, with errors that do not name it

- Where: `pws_phoenix/config.py:90`, `:107`, `:133-136`, `:151`,
  `:160-161`
- Code:

      return path if path.is_absolute() else base_dir / path
      ...
      else:  # data entries arrive as path strings
          resolved[key] = _resolve_path(val, base_dir)

- Consequence (not run):
  - A process entry value that is not a string (a number, a list)
    reaches `pl.Path(val)` and raises a `TypeError` that names
    neither the process nor the key.
  - A section present but empty (`maps:` with nothing under it) loads
    as None and fails on `.items()`. An empty file fails on
    `cfg["processes"]` with "'NoneType' object is not
    subscriptable".
  - A process entry without `class:` loads, and the Model fails with
    `KeyError: 'class'`. A map entry without `class:` raises
    `KeyError: 'class'` without the map's name.
  - Paths are joined, not normalized. The Model opens a file once
    when two entries give EQUAL paths (`Counter(flat_paths)`,
    `pws_phoenix/model.py:205`), so `params.nc` and `./params.nc`
    are opened twice and the two processes do not share arrays.
    `CLAUDE.md` lists this as open ("normalize paths (`.resolve()`)
    before file dedup"). With a relative `yaml_file` the returned
    paths are relative to the working directory at load time.
  - An inline `weights:` of integers gives an integer matrix
    (`np.asarray(spec)`, `:99`). What `Map` then does with integer
    weights was not checked.
- Source: /code-review, checklist A (an error that passes silently)
- Outcome: open

## Checklist A, item by item

- Hidden copy: 4.6. `np.asarray(spec)` (`config.py:99`) builds the
  matrix from a nested list, once, at load. `da.load().values` is
  one read. The graph holds names and counts only, no data.
- Allocation outside the three allowed classes: 4.6 (at load, not
  per step). Nothing in either file runs in a step.
- `self._obj[name].values` inside a step method: none; neither file
  has a step method.
- A raise on one rank alone: none. `load_model_yaml` holds no
  collective and no rank test; its raises (`:141`, `:162`) depend on
  the yaml and on `Process._registry`, which are the same on every
  rank. A file missing on one node's file system would differ, as
  for any file the Model opens. The graph does not run under MPI.
- A collective inside a rank-0 branch: none.
- Serial / MPI difference beyond the table: 4.4.
- An error that passes silently, or a check that cannot fail: 4.1,
  4.2, 4.3, 4.5, 4.7, 4.10, 4.11.

Checked and not a finding: the loader uses `yaml.safe_load`, so a
yaml cannot construct python objects. A `control` tolerance written
`1e-10`, which PyYAML reads as a string, is cast by the Model
(`float(control.get("budget_rtol", 1e-10))`, `pws_phoenix/model.py:476`).
The edge test in `to_dot` (`self._order.get(src, big) >
self._order.get(dst, big)`, `:414`) marks exactly the edges against
the schedule, which are prior-step reads because every `advance()`
runs before any `calculate()`. Every yaml-loaded grid with
`parameters:` goes through the `Path` branch of `Discretization`,
which opens a file that nothing closes (2.10). A class resolved by
name is the one registered last under that name (1.9); a composed
FlowGraph class must be built in code before the yaml is loaded.

## Notes for later slices

- Slice 16: `pws_phoenix/config.py` has 17% serial coverage. Its one
  test is the yaml style of `tests/test_two_grid_mpi.py` (MPI only),
  with no `discretizations:` section and file weights. No test
  reaches the body of `_load_discretizations`, inline weights, or
  either raise (`:141`, `:162`); only
  `examples_from_pws/01_multi-process_models.py` passes loaded
  discretizations to a Model. Why no test caught 4.1 (a yaml with
  `restart_read`, loaded from another directory) and 4.4.
- Slice 16: `tests/test_model_contract_graph.py` asserts on
  substrings of the dot and mermaid text; no test parses either
  output, and no test draws a sub-model. Why no test caught 4.7
  (`:188` is never reached) and 4.9. The toy graph test
  (`test_two_grid_toy`) does not look at `param_up_1` (4.10). The
  comment at `tests/test_model_contract_graph.py:176` says 13 dashed
  edges; the assertion is 9.
- Fix phase, runs that confirm before fixing: `to_dot()` of the
  example's sub-model through graphviz (4.7); a yaml with one
  process name twice (4.3); `dims: hru` without brackets (4.5).
