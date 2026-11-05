---
name: migrate-from-pywatershed
description: Migrate a Python script or notebook that runs pywatershed to pws_phoenix -- locate each pywatershed construct (Control, parameters, model dict or process list, FlowGraph, run loop, output), map it to its pws_phoenix equivalent or say plainly that it is not ported, build via prms_translate / pywatershed_translate where the mapping is mechanical, and verify against the user's existing pywatershed output.
---

# Migrating a pywatershed run script to pws_phoenix

Use this when a user has a script/notebook that runs pywatershed and
wants the same model in pws_phoenix. The input is THEIR code; the
output is a pws_phoenix script/notebook that runs the same model on
the same data, plus a parity check. Do not port process code -- every
PRMS process pywatershed has is already ported (see `docs/PORTS.md`
"Port inventory"); this skill moves the DRIVER.

Two kinds of thing are involved; keep them apart:

- **Mechanical** (module does it): file readers, class resolution from
  flags, parameter splitting, Map construction, output-name
  translation. Call `prms_translate` / `pywatershed_translate`; never
  re-implement these by hand in the user's new script.
- **Judgment** (you do it): what the script's hand-written parts mean
  (in-memory parameter edits, hand loops, FlowGraph splices, custom
  post-processing) and what is not available. Each decision goes into
  the new script as a comment the user can audit.

## Procedure

1. **Read the whole script first.** Collect every construct in the
   tables below into a checklist before writing anything. A script
   usually has 3-6 of them; a notebook often builds the same model two
   or three ways (in-memory dict, yaml, submodel) -- migrate each.
2. **Classify each item** as C (module), S (judgment), N (not ported).
   Any N stops and asks the user before proceeding: an N that the
   script's result depends on (cascades, per-step restart writes)
   means the migration cannot reproduce the run.
3. **Pick the entry point** (first match wins):
   - script builds from a PRMS control file + `myparam.param` + CBH →
     `prms_translate.assemble.model_from_control(control_file,
     control=...)`. Nothing else to translate.
   - script builds from netcdf parameters (`parameters_<Class>.nc`,
     `parameters_dis_*.nc`) + a model dict / yaml pair →
     `pywatershed_translate` (see "Helpers this skill expects").
   - script hand-assembles (in-memory `Parameters` edits, FlowGraph
     splices, custom carriers) → start from the kit the closest entry
     point gives (`assemble_from_control` / `assemble_from_pywatershed`
     return a `ModelKit`, not a Model) and edit the kit's
     `process_dict` / `maps` / `discretizations` before `.model()`.
4. **Write the new script** as a py:percent notebook if the source was
   a notebook (follow the `notebooks` skill), else a plain script.
   Keep the user's variable names and section order where they still
   make sense; mark every translated decision with a one-line comment
   naming the pywatershed construct it replaces.
5. **Verify** -- ask the user which of three modes, once, up front:
   (1) none: no verification code is written; (2) write the
   verification cell/section (the recipe below) and the user runs it;
   (3) the skill runs it (still asking permission before each run, per
   the repo's ground rules). The target is the pywatershed output the
   user already has (their `netcdf_output_dir`); if none exists, mode 3
   runs pywatershed once (ask first). Per-variable tolerances come from
   `docs/PORTS.md` "Port inventory" -- NOT 1e-13 across the board:
   snow-downstream variables vs pywatershed's fastmath numba output
   need the outlier-fraction criterion (PORTS.md "The snow story").
6. **Report** what is identical, what differs and why, and the N list.

## Mapping tables

### Control

| pywatershed | pws_phoenix | class |
| --- | --- | --- |
| `Control.load_prms(file)` | `model_from_control(file)` (prms_translate). `readers.load_control` DROPS entries the PRMS 5.2.1.1 metadata does not know and prints them (`cbh_binary_flag` in the fgr template control; pyPRMS would raise) -- pywatershed's `warn_unused_options=False` hides the same thing | C |
| `options["prcp_day"]`, `tmax_day`, `tmin_day`, ... (the control's CBH file entries) | read from the control; when the names do not match the files on disk (fgr names `precip.day`, ships `prcp.cbh`; pywatershed never reads these entries, it opens `<input_dir>/<var>.nc`), pass `assemble_from_control(..., cbh_paths={"prcp": path, "tmax": ..., "tmin": ...})` | C |
| `Control.from_yaml(file)` / `Control(start_time, end_time, time_step, options=)` | `pywatershed_translate.load_control_yaml` → `PrmsRunConfig` window + flags; `time_step` must be 24 h (daily is fixed) | C |
| `edit_end_time` / `edit_n_time_steps` | a shorter run = `model.run(dt, n_steps)` with fewer steps; the window itself comes from the control file (`PrmsRunConfig.start_time/end_time`) | C |
| `edit_init_start_times` (a later start) | no override today -- edit the control file's `start_time`, or wait for the planned `window=` kwarg on the assemblers | S |
| `options["input_dir"]` | where CBH / forcing netcdf are read from; pass through; pywatershed output dirs are valid input dirs (submodels) | C |
| `options["parameter_file"]` | read from the control; `cfg.param_files` | C |
| `netcdf_output_dir` + `netcdf_output_var_names` | `control["output_serial_zarr"] = <path>.zarr` + `control["output_var_names"]`; NAMES translate, PATHS never (choose a new path; a pywatershed output dir must not be overwritten -- it is the verification target) | C |
| `imbalance_behavior` (`None`/`"warn"`/`"error"`) | `control["budget"]` same three values; tolerances `budget_rtol`/`budget_atol` default 1e-10 (pywatershed's 1e-5 is looser -- do not copy it) | C |
| `dprst_flag`, `iter_aet_flag`, `stream_temp_flag`, `stream_temp_shade_flag`, `strmtemp_humidity_flag`, `dyn_*_flag` / `*_dynamic`, `AET_cbh_file`, `intcp_changeover_in_net_rain` | class resolution: `prms_translate.control.resolve_classes(modules, flags)` picks NoDprst / Ag / ObsET / stream-temp leaves / TranspFrost(Dyn) | C |
| `calc_method` | drop (numba is the only path) | C |
| `verbosity`, `netcdf_output_separate_files`, `find_input_files`, `warn_unused_options` | drop | C |
| `restart_read=dir` | `control["restart_read"] = dir` (serial; pws_phoenix's own restart files, written by `model.write_restart(dir)`) -- pywatershed restart FILES are not readable here; re-create the warm state by running to the restart time and calling `write_restart` | S |
| `restart_write` + `restart_write_freq="f"` | `model.write_restart(dir)` after `run()` | C |
| `restart_write_freq="d"` (every step) | N -- one restart file per run; re-run in daily chunks (notebook 08 pattern) if that is really wanted | N |
| `init_vars_from_file`, `var_save_file` (PRMS binary restart) | N (raises in `from_control`) | N |
| `cascade_flag`, `cascadegw_flag` | N -- no cascades in pws_phoenix | N |
| `MPI` wanted | `prms_translate.assemble_mpi.mpi_model_from_control` under `mpirun`; output routed by grid (`output_parallel_netcdf` for the distributed grid, `output_serial_zarr` rank-0 for replicated grids) | C |

### Data

| pywatershed | pws_phoenix | class |
| --- | --- | --- |
| `pws.utils.addtl_domain_files.download()`, `pws.utils.get_addtl_domains_dir("fgr_2yr")` | `pws_phoenix.utils.addtl_domain_files.download_addtl_domains()` / `get_addtl_domains_dir(domain)` (same release zip; lands in `pws_phoenix/data/`, gitignored) | C |
| `pkg_root / "data/big_sandy_starfit_parameters.nc"`, `pkg_root / "data/..."` (pywatershed PACKAGE data) | read the same file from the pywatershed clone (`PYWS_ROOT / "pywatershed" / "data" / ...`, with an assert naming it); pws_phoenix commits NO data until a data repo exists | S |
| `pws.utils.gis_files.download()`, `DomainPlot`, folium maps | N today -- drop the map section with a pointer to the pywatershed notebook (`gis_files` port + `pws_phoenix/plot/` are planned) | N |

### Parameters

| pywatershed | pws_phoenix | class |
| --- | --- | --- |
| `PrmsParameters.load(myparam.param)` | handled inside `model_from_control` | C |
| `Parameters.from_netcdf(parameters_<Class>.nc)`, `parameters_dis_{hru,seg,both}.nc`, `Parameters.merge`, `from_ds`, `.subset` | `pywatershed_translate.load_parameters_netcdf(paths)` → one `xr.Dataset` (dims `nhm_id`→`nhru`, `nhm_seg`→`nsegment`, float32→float64, duplicated dis vars deduplicated) → `prms_translate.parameters.package_parameters` splits it by class declarations | C |
| `params.to_xr_ds()` / `to_dd()` then arithmetic (ensembles, perturbations) | edit the kit's per-process parameter datasets BEFORE `kit.model()`; they are the model's working buffers (zero-copy) and are frozen read-only at assembly. Keep the arithmetic verbatim. | S |
| `pws.utils.preprocess_cascade_params` | N | N |
| `PRMSSolarGeometry` parameters | there is no solar-geometry process: `atmosphere.prms_solar_geometry.compute_soltabs(dis_hru)` makes the `soltab_*` tables (a DERIVABLE parameter, done by the assemblers; `prms_translate.preprocess.write_preprocessed` caches it) | C |

### Model structure

| pywatershed | pws_phoenix | class |
| --- | --- | --- |
| `Model([PRMSSolarGeometry, PRMSAtmosphere, ...], control=, parameters=)` (legacy list) | the assemblers' NHM-ordered `process_dict`; class names are identical except: no `PRMSSolarGeometry` (see above); `PRMSStreamTempHumidityCBH` → `PRMSStreamTemp` + a humidity Map (CBH-humidity config = core class + Map, no leaf) | C |
| model dict: `{"control", "dis_hru"/"dis_seg"/"dis_both", "model_order", "<name>": {"class", "parameters", "dis"}}` and `Model.from_yaml(model_dict.yaml)` | `pywatershed_translate.assemble_from_pywatershed(control, model_dict)` → `ModelKit`: `dis` entries → `discretizations` (hru grid bare; segment grid = `Discretization(dims, parameters=dis_seg, topo_order=...)`); `model_order` → `process_dict` order; segment-grid consumers get the three volume Maps, stream temperature gets the ten aggregation Maps + `HumidityCarrier` | C |
| `"input_aliases": {"humidity_hru": "rhavg"}` | CBH rename table (`prms_translate.cbh`: `rhavg`→`humidity_hru` already) | C |
| `"stream_shade_class"`, `"stream_shade_parameters"` | shade is inside `PRMSStreamTemp` (`_compute_shade`); dynamic vs `PRMSStreamTempConstantShade` chosen by `stream_temp_shade_flag`; shade params merge into the stream-temp parameter set | C |
| submodel (a subset of processes with `input_dir` = a full run's output) | same subset in `process_dict`; `Model.input_spec(process_dict, maps)` lists the externals; feed them from the pywatershed output files (they are already `(time, nhm_id)` netcdf -- rename the dim) | C |
| `model.processes[name]["var"]`, `.get_variables()` | `Process` classmethods (`get_variables()`, `get_parameters()`, `get_inputs()`); runtime values live on the grid dataset `model.discretizations[grid].dataset[var]` | S |
| `pws.analysis.ModelGraph(model).SVG()` | `pws_phoenix.model_contract_graph` (`to_dot()` / `to_mermaid()`; notebook `examples/00_input_contract.py`) | C |

### FlowGraph

| pywatershed | pws_phoenix | class |
| --- | --- | --- |
| `prms_channel_flow_graph_to_model_dict(model_dict, prms_channel_dis, prms_channel_params, new_nodes_maker_dict, new_nodes_maker_names/indices/ids, new_nodes_flow_to_nhm_seg, addtl_output_vars)` | `prms_translate.assemble.insert_flow_graph(kit, new_nodes)` on a kit (PRMS control today; `pywatershed_translate` kit later), or `flow_graph_build.splice_flow_graph(dis_seg, channel_params, new_nodes)` -> `FlowGraphKit` when hand-assembling. Same semantics as pywatershed: `NewNode(node_type, flows_to=<nhm_seg>, node_id=<maker_id>, parameters={...})` intercepts above that segment; `flows_to=<another NewNode>` = pywatershed's non-positive in-series index. The collated `names/indices/ids` lists collapse into the `NewNode` list (one per node, in the same order); `node_maker_id` becomes `node_id` = the `nnodes` coordinate (`ds.sel(nnodes=999)`). `addtl_output_vars` dissolve: every node field is an ordinary variable (`node_outflows`, `lake_storage`, `lake_release`, `lake_spill`, ...) -- request them in `output_var_names` | C |
| `StarfitFlowNodeMaker(discretization, sf_params, compute_daily=, imbalance_behavior=)` | `NewNode(StarfitFlowNode, ...)` (hourly, `compute_daily=False`) or `NewNode(StarfitDailyFlowNode, ...)` (`compute_daily=True`; ONE-DAY LAG, matches pywatershed's daily mode, never the hourly reference). `parameters` = the reservoir's row of `sf_params` (`{name: float(row[name]) for kind=="parameter"}`) in the GRAPH's units (cfs graph: storages in Mcf = MCM x `cms_to_cfs`); nan `initial_storage` / `Obs_MEANFLOW_CUMECS` must be filled by the caller (pywatershed fills in-node; `initialize_type` raises): `Obs_MEANFLOW_CUMECS` <- `float(row["inflow_mean"])`; `initial_storage` <- `starfit_flow_node.nor_midpoint_initial_storage(row, init_date)` (MCM; `* cms_to_cfs` in a cfs graph) with `init_date` = the control start MINUS ONE DAY (pywatershed's init time) and `row` ALREADY carrying any capacity scaling (`GRanD_CAP_MCM *= cap_mult` first -- the midpoint depends on it, as it does in pywatershed). `examples_from_pws/06_flow_graph_starfit.py::starfit_node_parameters` is the worked version. `imbalance_behavior` -> `control["budget"]` | S (the fill) |
| `PassThroughFlowNodeMaker`, `ObsInFlowNodeMaker`, `SourceSinkFlowNodeMaker`, `StarfitSourceSinkFlowNodeMaker` | `NewNode(PassThroughFlowNode / ObsInFlowNode / SourceSinkFlowNode / StarfitSourceSinkFlowNode, ...)`; per-node pandas Series inputs become `kind="input"` fields (`node_obs_flow`, `node_source_sink`, `(time, nnodes)`, zero where not applicable) served in lockstep (`missing_data_as_zero` = data prep) | S (the inputs) |
| `prms_channel_flow_graph_postprocess(control, input_dir, ...)` + hand loop | a one-process Model: `splice_flow_graph(...)` for the class/dis/params; externals (`sroff_vol`, `ssres_flow_vol`, `gwres_flow_vol`) from the previous run's output files, mapped by the three volume Maps padded with `kit.pad_weights(...)` | S |
| a kit with stream temperature / hydraulic geometry + a FlowGraph | `insert_flow_graph` raises `NotImplementedError` (they consume `seg_outflow` on `nsegment`); not wired | N |
| `graph_imbalance_behavior` | `control["budget"]` (FlowGraph budgets are declared per node) | C |

### Run loop and output

| pywatershed | pws_phoenix | class |
| --- | --- | --- |
| `model.run(finalize=True)`, `model.run(netcdf_dir=)`, `model.initialize_netcdf()` | `model = kit.model(); model.run(np.float64(86400.0), np.int32(n_days)); model.finalize()` with `output_serial_zarr` set (`dt` is seconds; `finalize` is a separate call -- skipping it loses the buffered tail of the zarr output, with a warning) | C |
| `control.advance(); proc.advance(); proc.calculate(1.0); proc.output()` | never drive a process by hand -- budgets are only checked by the Model. Make a one-process Model. `dt` is seconds (86400.0), not 1.0 | S |
| `pws.base.Output(control, chunked_var_list, chunked_output_file=<zarr>)`, `run(output_obj=)`, `output_obj_kwargs_dict` | `control["output_serial_zarr"]` + `output_var_names` (same buffered-chunk zarr design) | C |
| `Output` statistics (`hoi_stats`, `noi_stats`, `monthly_accumulations`, `percentile_resamp_enclosing`, ...) | N in the framework; show the xarray equivalent on the zarr store after the run | N |
| `model.processes[name].mass_budget` / `energy_budget` repr | budget terms are ordinary variables `{Class}_{budget}_{inputs_sum,outputs_sum,storage_changes_sum,residual,residual_accum}` -- add them to `output_var_names` and inspect from the store | C |
| `PRMSChannel_mass_budget.nc` | N -- the channel is not budgeted (its per-segment form is an identity; module docstring) | N |
| `multiprocessing` / `concurrent.futures` ensembles | unchanged pattern (serial pws_phoenix per worker); or MPI for one big run | S |
| GIS / `pws.utils.gis_files` / plotting | unchanged; variables carry the same names and `nhm_id`/`nhm_seg` coords are preserved on the grid dims | S |

## Verification recipe

```python
import xarray as xr
pws_out = xr.load_dataarray(pywatershed_dir / f"{var}.nc").rename({"nhm_id": "nhru"})
phx = xr.open_zarr(zarr_path)[var]
# tolerance per docs/PORTS.md "Port inventory" for the OWNING process
xr.testing.assert_allclose(phx, pws_out, rtol=rtol, atol=atol)
```

- groundwater / channel / canopy: 1e-12 to 1e-10; runoff / soilzone:
  1e-10; atmosphere 1e-5; snow and everything downstream of it vs a
  pywatershed NUMBA run: outlier-fraction criterion (PORTS.md "The
  snow story") -- pywatershed's fastmath drifts from its own numpy
  path, not from us.
- The first mismatch to suspect is the window: pywatershed's
  `edit_init_start_times` vs `start_time`; both must land on the same
  first output date.
- **Isolate the ported piece.** A full-chain comparison vs a
  pywatershed NUMBA run can only be loose (the snow knife-edges);
  do not chase it tighter. Wherever the script has a stage driven
  from files (a post-process FlowGraph, a submodel), drive OUR stage
  with pywatershed's OWN intermediate output files and pin that
  tight (1e-10 for the graph). Pattern: `06_flow_graph_starfit.py`
  "Checking against pywatershed" -- loose prints + gross-breakage
  asserts for the chains, then the graph on pywatershed's
  `fgr_nhm/*_vol.nc` with zero values outside 1e-10.

## Helpers this skill expects

Existing (`pws_phoenix/prms_translate/`): `assemble.model_from_control`
/ `assemble_from_control` → `ModelKit` (`process_dict`, `control`,
`maps`, `discretizations`, `.model()`); `assemble.insert_flow_graph`;
`control.resolve_classes`; `parameters.package_parameters`;
`cbh.load_cbh`; `preprocess.write_preprocessed`;
`assemble_mpi.mpi_model_from_control`; `readers.load_control`
(unknown-entry filter); `assemble_from_control(..., cbh_paths=)`.
Core: `flow_graph_build.splice_flow_graph` / `NewNode` /
`FlowGraphKit`; `hydrology.starfit_flow_node.
nor_midpoint_initial_storage`; `utils.addtl_domain_files`.

PLANNED (`pws_phoenix/pywatershed_translate/`, not yet written -- when
a mapping above names one of these, hand-build the equivalent from the
existing helpers and tell the user the helper is pending):
`load_control_yaml`, `load_parameters_netcdf`,
`assemble_from_pywatershed(control, model_dict)`; and a
`window=(start, end)` override on the existing assemblers.

## Hard rules

- One-way: the model code never imports the translators.
- Never hand-copy parameter values into the new script; always load
  from the user's files so provenance survives.
- Never guess a tolerance; cite the PORTS.md line.
- Say N plainly, early, with the reason, and stop if the run depends
  on it.
