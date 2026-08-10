<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [pywatershed process ports](#pywatershed-process-ports)
  - [The chain](#the-chain)
  - [Port inventory](#port-inventory)
  - [Port conventions (established over the arc)](#port-conventions-established-over-the-arc)
  - [The snow story (fastmath and knife edges)](#the-snow-story-fastmath-and-knife-edges)
  - [Framework findings along the way](#framework-findings-along-the-way)
  - [drb_2yr domain facts the tests rely on](#drb_2yr-domain-facts-the-tests-rely-on)
  - [The agricultural (Ag) ports (July 2026)](#the-agricultural-ag-ports-july-2026)
  - [The stream-temperature chain (July 2026)](#the-stream-temperature-chain-july-2026)
  - [Not ported / backlog](#not-ported--backlog)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# pywatershed process ports

Status and reference for the pywatershed → pws_phoenix process ports
(the "goal 4" pathway). As of July 2026 **every process in the NHM
model configuration is ported**: the full chain runs from the raw CBH
files (`prcp`/`tmax`/`tmin`) + parameters to channel outflow, serial
and MPI-distributed, with no pywatershed at runtime.

Sources are pywatershed (PRMS 5.2.1 physics; the repo lives at the
mpix meta-repo root). Validation compares against pywatershed's
GENERATED drb_2yr answers (`test_data/drb_2yr/output/`, produced by
its autotest data-generation workflow — not checked in; parity tests
skip cleanly when absent).

## The chain

```
CBH files (prcp, tmax, tmin)          parameters + dis files
        │                                      │
        ▼                                      ▼
PRMSAtmosphere ──► PRMSCanopy ──► PRMSSnow ──► PRMSRunoff ──► PRMSSoilzone ──► PRMSGroundwater
   (solar tables via the                  hru grid: ONE shared dataset,
    compute_soltabs factory)              all fluxes by structural sharing
                                                       │
                                    sroff_vol / ssres_flow_vol / gwres_flow_vol
                                                       │  3 Maps (MapMPI under MPI)
                                                       ▼
                                                  PRMSChannel ──► seg_outflow, seg_lateral_inflow
                                                  (segment grid, replicated        │
                                                   under MPI)                      ▼
   tavgc / swrad / potet / hru_rain / ccov_hru /                    PRMSHydraulicGeometryWidthOnly
   sroff / ssres_flow / gwres_flow / snowmelt /                            │  seg_flow_width
   humidity_hru (CBH, via a 1-var carrier)                                 ▼
        │              10 aggregation Maps                           PRMSStreamTemp
        └──────────────────────────────────────────────────────────────────┘
```

Schedule = NHM order (dict order in `process_dict`). Correctness of
the prior-step back-edges (snow's `pk_ice_prev`/`freeh2o_prev` to
canopy; soilzone's `soil_lower_prev`/`soil_rechr_prev` to runoff)
relies on Model/ModelMPI running ALL `advance()` hooks before any
`calculate()`.

## Port inventory

| Process | Module | Validation standard | Notes |
|---|---|---|---|
| PRMSGroundwater | `hydrology/prms_groundwater.py` | 1e-13 (upstream's own) | First real port; dis-first parameter sourcing |
| PRMSChannel | `hydrology/prms_channel.py` | 1e-13; `seg_stor_change` (1e-7, 1e-4) cancellation carve-out | hru→segment aggregation externalized to Maps; muskingum coefficients = `parameter_derived` |
| FlowGraph (6 node types) | `flow_graph.py` + `hydrology/*_flow_node.py` | 1e-7…1e-10 per mode | Registry dispatch via `literal_unroll`; hourly + daily STARFIT; io_in_cfs |
| PRMSRunoff | `hydrology/prms_runoff.py` | 1e-10 (upstream's own) | dprst-ACTIVE path; `basin_init`/`dprst_init` → `initialize()` + `parameter_derived` |
| PRMSSoilzone | `hydrology/prms_soilzone.py` | 1e-10 observed (upstream's own is 5e-6) | First `mutable_input` (`sroff`/`sroff_vol`: dunnian added in place) |
| PRMSCanopy | `hydrology/prms_canopy.py` | 1e-12 (upstream's own) | `pptmix` = written-never-read mutable input; hru_type hardwired all-LAND upstream |
| PRMSSnow | `hydrology/prms_snow.py` | **bit-identical** to upstream's numpy path (A/B); 1e-3 on upstream's own 5-var list vs answers | See "The snow story" below |
| PRMSSolarGeometry | `atmosphere/prms_solar_geometry.py` | 1e-10 (upstream's PRMS 5.2.1 standard) | A parameter FACTORY, not a Process (static (ndoy, nhru) tables) |
| PRMSAtmosphere | `atmosphere/prms_atmosphere.py` | 1e-5 (upstream's own) | Per-step port of upstream's all-time-vectorized preprocessing |
| NoDprst variants (runoff, soilzone, gw) | same modules as their full classes | bit-identical to the full class with dprst disabled by data (`tests/test_prms_no_dprst.py`); answer parity pending data (see below) | The ADDITIVE inversion of upstream's subtractive subclassing -- see "How variants are done here" |
| PRMSRunoffAg | `hydrology/prms_runoff.py` | 1e-5 + exceptions (upstream's own; GSFLOW Fortran answers, fgr_ag_2yr) | Additive extension of PRMSRunoff; dynamic ag_frac; serial + 4-rank MPI |
| PRMSSoilzoneAg | `hydrology/prms_soilzone_ag.py` | 1e-5 + exceptions (fgr spinup) | SIBLING dual-area family core (see module docstring); serial + 4-rank MPI + live RunoffAg chain |
| PRMSSoilzoneAgObsET | `hydrology/prms_soilzone_ag.py` | 1e-5 + exceptions (fgr analysis: obs-AET iteration + dynamic ag_frac) | Additive extension of PRMSSoilzoneAg (It0 loop + extracted irrigation-adjust kernel) |
| PRMSAtmosphereTranspFrost | `atmosphere/prms_atmosphere.py` | 1e-5; transp_on EXACT (ucb_2yr nhm_transp_frost) | Frost-window leaf of the new PRMSAtmosphereBase family (tindex params are NOT a superset -> shared abstract base, each leaf ADDS its transp kernel); Time.jsol = solar day of year |
| PRMSHydraulicGeometry (WidthOnly + Full) | `hydrology/prms_hydraulic_geometry.py` | 1e-5, all 5 vars (upstream's own test only checks seg_res_time) | WidthOnly = core w/ default depth as derived; Full = declaration override to supplied params (bit-identical pin) |
| Stream shade (shday) | `hydrology/prms_stream_shade.py` | 5e-3 (upstream's own family standard: Fortran trig noise; observed max 2.4e-3, 93% within 1e-5) | Upstream strategy objects -> VERBATIM-extracted njit fns; standalone daily pin vs seg_shade; drb = dynamic mode |
| PRMSStreamTemp | `hydrology/prms_stream_temp.py` | 5e-3, 6 variables (drb nhm_stream_temp); serial + 4-rank MPI (replicated segment grid, LIVE seg_flow_width from PRMSHydraulicGeometryWidthOnly -- `tests/test_prms_stream_temp_mpi.py`) | SEGMENT-grid port: HRU-derived aggregates are INPUTS (upstream is secretly two-grid; drb has 40 no-HRU segments so its seg_close aggregation is graph-based -> chain stage); kernels/equilibrium solver verbatim-extracted; gw/ss silos = python-attr state; energy-flux tracking not ported (upstream excludes it). Shade = the `_compute_shade` hook on abstract `PRMSStreamTempBase` (dynamic/constant parameter sets are disjoint -> base family like PRMSAtmosphereBase); PRMSStreamTemp = the DYNAMIC leaf (shade_flag=0, the answers-validated configuration) |
| PRMSStreamTempConstantShade | `hydrology/prms_stream_temp.py` | behavioral pin only (`tests/test_prms_stream_temp_const_shade.py`): seasonal selection EXACT, svi stays 0, never-flow structure preserved | shade_flag=1 leaf (upstream PRMSStreamShadeConstant, verbatim semantics): seg_shade = segshade_sum on doy 121-273 else segshade_win; svi = 0 (hv longwave term vanishes). pywatershed generates NO drb answers for this mode, hence no Fortran parity |
| PRMSStreamTempSegHumidity | `hydrology/prms_stream_temp.py` | 5e-3 incl seg_humid (drb seg_humid_matrix AND seg_humid_scalar configs, parametrized in one test) | strmtemp_humidity_flag=1 leaf: seg_humid declaration OVERRIDE input -> variable, assigned from the monthly seg_humidity parameter (matrix and scalar are the SAME upstream code path -- one leaf serves both). DATA NOTE: parameters_PRMSStreamTemp.nc carries a UNIFORM seg_humidity (0.627) that does NOT match either config's myparam -- the test reconstructs the true monthly parameter exactly from each config's seg_humid answers |
| hru->segment aggregation (LIVE, via Maps) | `hydrology/prms_stream_temp.py` | kernels vs Fortran 1e-5 (all 10 aggregates, all days) + weights vs kernels 1e-12 (`tests/test_prms_stream_temp_aggregates.py`); live chains below | Verbatim-extracted kernels + `resolve_aggregation_topology` (segment_up / auto-seg_close incl. the route-order fallback; the 40 no-HRU segments' fallbacks and drb's ONE order-dependent segment all match Fortran under our dis topo order) + `derive_aggregation_weights()` (weights by basis-vector PROBING of the kernels) + `AGGREGATION_MAP_SPEC`. See "The stream-temperature chain" below |
| ccov_hru (relocated cloud cover) | `atmosphere/prms_atmosphere.py` | EXACT vs the reference block on the model's own swrad (`test_prms_atmosphere.py`) | Upstream computes cloud cover INSIDE stream temp's aggregation loop; relocated to PRMSAtmosphereBase (it is HRU meteorology, and Maps never originate variables) -- the one per-HRU nonlinear step, whose removal makes every aggregate exactly linear-static |
| Stream-temp live chains | `tests/test_prms_stream_temp_chain.py` (Maps chain), `..._full_chain.py` + `..._full_chain_mpi.py` (the COMPLETE NHM from raw CBH through stream temperature) | Maps chain: aggregates 1e-5, temps 5e-3. Full chain (13 Maps, serial + 4-rank MPI): 5e-3 with outlier fraction <= 5e-3 (measured 0.16% worst; gw/ss/shade 0.0) + bit-identical cross-rank replication | See "The stream-temperature chain" below for the tolerance story (the two pywatershed answer generations differ from EACH OTHER) |

Every port has a standalone serial parity test and a 4-rank MPI test
(`tests/test_prms_<name>{,_mpi}.py`); the chain is validated end to
end in `tests/test_prms_channel{,_mpi}.py`.

## Port conventions (established over the arc)

- **Names verbatim** (parameters / inputs / variables) — the pathway
  back to pywatershed domains and answer files. This includes
  upstream's underscore-private derived arrays (e.g. soilzone's
  `_sat_threshold`).
- **Kernels**: pywatershed's allocate-and-return-tuple
  `_calculate_numpy` becomes an in-place, out-first `@numba.njit`
  kernel — an EXPLICIT element loop with scalar temporaries (numba's
  expression fusion does not eliminate NAMED intermediate arrays).
  Per-element operation order is kept identical to upstream's
  vectorized expressions. Upstream's scalar helper functions
  (`compute_infil`, `calc_calin`, …) stay separate njit functions with
  verbatim signatures, called directly (not passed as arguments).
- **Init-time work** (`basin_init`, `_initialize_soilzone_data`,
  muskingum coefficients, solar tables) → `initialize()` +
  `kind="parameter_derived"` (frozen after) or the factory pattern;
  init-time numpy staging is fine there.
- **NOT ported, everywhere**: Budget/ConservativeProcess (backlogged
  design pass), adapters, restart, `calc_method` switch (numba is THE
  path), `verbose`, `fastmath=True` (strict IEEE here — see below),
  unused declared parameters, dead upstream code.
- **Upstream edits of true parameters** (runoff's clos-flag zeroing,
  soilzone's `soil_moist_max` clamp, channel's `seg_slope` clamp,
  snow's nonzero-`snowpack_init` block) are either proven no-ops and
  documented, or guarded with `NotImplementedError` — frozen
  parameters are never edited.
- **Time-varying parameters**: `(nmonth, space)` params indexed by
  `time.month - 1` in-kernel; static `(ndoy, space)` tables indexed
  by `time.doy - 1` (pywatershed's netCDF-reader semantics);
  `('scalar',)` params extracted as kernel floats.
- **Tolerances**: always pywatershed's OWN autotest standard for that
  process (they vary enormously: 1e-13 gw/channel, 1e-12 canopy,
  1e-10 runoff, 5e-6 soilzone, 1e-3 snow-on-5-vars, 1e-5
  atmosphere) — tightened where the observed level allows.

## The snow story (fastmath and knife edges)

pywatershed compiles snow (and soilzone, canopy) with numba
`fastmath=True`; its GENERATED answers carry that arithmetic. Snow is
branchy accumulated state, so the ~ulp-level fastmath differences
grow to ~1e-8-relative state drift within days and flip pack-survival
knife edges on ~0.02% of hru-days — `tcal` (pack energy, O(100)
cal/cm² regardless of pack size) shows O(400) excursions at those
flips. pywatershed's own strict-numpy path shows the SAME excursions
against its own answers.

Our snow port is **bit-identical to pywatershed's strict-IEEE numpy
path** (`tests/test_prms_snow_ab_numpy.py`: 120-day A/B, 14 state
variables, exact equality). That A/B carries the precision guarantee;
the answers-based test mirrors upstream's own 5-variable/1e-3 list
with an outlier-fraction criterion for the knife-edge-amplified
`tcal`/`through_rain`.

Downstream consequence: with snow live, the seasonal 1e-8 noise feeds
every flux and muskingum smears it (measured: 15%/1.6%/0.015% of
`seg_lateral_inflow` segment-days outside 1e-10/1e-8/1e-2). Hence the
chain test's TWO modes:

- **snow_disk** — canopy→runoff→soilzone→gw with atmosphere+snow
  products from disk; STRICT 1e-10. The sensitive plumbing canary.
- **snow_live** — the full 7-process chain from CBH; (1e-2, 1e-2)
  with an outlier fraction. The fastmath-answers ceiling; per-process
  tests carry precision.

**The ceiling confirmed against pywatershed itself** (July 2026): the
`output/` and `output_stream_temp/` answer generations — two
pywatershed runs of the same chain — differ from EACH OTHER at
exactly this level (identical atmosphere; snow knife-edge flips from
day 80 cascade to 68% of seg_outflow seg-days, max 59 cfs). Details
and the resulting stream-temp criterion: "The stream-temperature
chain" below.

## Framework findings along the way

- `kind="mutable_input"` (soilzone edits runoff's `sroff`; canopy
  edits atmosphere's `pptmix`) and the advance-all-then-calculate
  ordering carry pywatershed's cross-process in-place edit semantics.
- Shared disk forcings are fed ONCE (to the first consumer in the
  schedule); overlapping parameters across process files merge
  (identical NHM values).
- mpixarray cannot declare multi-dim derived buffers (its buffer
  creation decomposes EVERY declared dim) — hence per-element unit
  conversions in kernels (snow's `tmax_allsnow` F→C) and the solar
  FACTORY pattern instead of (ndoy, space) `parameter_derived`.
- pywatershed hardcodes `dnearzero = 2.23e-16` (NOT
  `np.finfo(float64).eps`) — threshold branches differ if you get
  this wrong.
- Test hygiene: tests use `xr.load_dataset`/`load_dataarray`
  (open-load-close). Past ~128 open netCDF handles, xarray's
  file-manager LRU churns reopen/evict on every access — a de-facto
  hang at 255% CPU.

## drb_2yr domain facts the tests rely on

- 765 HRUs / 456 segments × 731 daily steps (1979–1980); uneven
  under 4 ranks (192/191/191/191) — a deliberate decomposition probe.
- `sat_threshold >= 999` → dunnian flow ≡ 0 (why runoff's `sroff` is
  unmodified by soilzone; upstream's own runoff autotest skips
  otherwise).
- `dprst_frac_open == 1` everywhere → no closed depressions (the
  clos-flag parameter-zeroing no-op).
- `snowpack_init == 0` everywhere (the buggy upstream init block is
  never exercised).
- Output files put variables on `nhm_id`; parameter files use
  `nhru` — rename on assembly (the framework rejects mismatches).
- CBH files are float32 — widened exactly to f64 on input.

## The agricultural (Ag) ports (July 2026)

Domain/answers: **fgr_ag_2yr** (GSFLOW; converted Fortran answers,
partly single precision; upstream's own standard = 1e-5 with
per-variable exceptions). Two configurations: `spinup` = static
ag_frac, no iteration; `analysis` = obs-AET iteration + DYNAMIC
ag_frac (annual Jan-1 changes via dyn_ag_frac.param).

- **PRMSRunoffAg(PRMSRunoff)** -- additive extension (upstream
  interface is a strict superset). `hru_perv`/`hru_frac_perv` are
  redeclared per-step variables (declaration override); the kernel
  sees the previous step's areas while `ag_frac` scalars are current
  (upstream ordering). `intcp_changeover_in_net_rain` became a
  RUNTIME kernel argument across the runoff family (upstream's own
  signature): GSFLOW mode (True, fgr) skips the changeover block and
  flips the rain-on-snow check to `not (net_ppt - net_snow > 0)`.
- **PRMSSoilzoneAg(Process)** -- sibling family core (see the module
  docstring for the full extend-vs-sibling rationale). One shared
  dual-area kernel with a runtime `iter_aet_flag`; reuses the four
  plain-soilzone njit helpers.
- **PRMSSoilzoneAgObsET(PRMSSoilzoneAg)** -- adds observed-AET
  iteration: It0 state save/restore around the SAME kernel, with the
  irrigation-adjustment block extracted to its own njit kernel and
  the convergence diagnostics.

**The change-variables carve-out (REVIEW NOTE, Jul 27).** The first
analysis-parity run failed on `ag_soil_moist_change` at exactly
timestep 366 -- Jan 1, the first dynamic-ag_frac change date (82 of
447,372 points, O(0.5-5) inches, ours ~0 vs Fortran large-negative).
This is NOT a port bug: when ag_frac changes, water is redistributed
between the ag store and slow_stor / rescaled over the new areas.
pywatershed's kernel (and this port, verbatim) computes the five
change variables as `current - prev - redistribution` so the mass
budget excludes area-accounting transfers; the Fortran-postprocessed
answer files are plain `current - prev` and INCLUDE them. pywatershed
documents this divergence in its kernel comments and its own autotest
EXCLUDES the five (`ag_soil_moist_change`, `ag_soil_rechr_change`,
`slow_stor_change`, `soil_lower_change`, `soil_rechr_change`) from
external comparison, relying on its mass budget instead. Mirrored
here: `tests/test_prms_soilzone_ag_obs_et.py` (analysis) excludes the
five; `tests/test_prms_soilzone_ag.py` (spinup, static ag_frac ->
redistributions identically zero) still compares all five -- a
stronger pin upstream doesn't have. When the Budget design pass lands
(backlog), these variables get their mass-budget validation here too.

Also mirrored from upstream's ag tests: the pervious-zone variable
list compared only on non-ag HRUs (upstream's mask; note its
dangling-else makes the ag-HRU mask inert -- everything else is
compared on all HRUs), and `sroff_vol` excluded from runoff-ag
comparison (single-precision Fortran error scales with hru_area).

## The stream-temperature chain (July 2026)

Upstream PRMSStreamTemp is secretly a TWO-GRID process: it takes HRU
inputs and aggregates them to segments in-process. Here that
aggregation is externalized, governed by one principle (JLM):
**a Map is a grid-to-grid correspondence, per variable -- one in, one
out, renaming allowed, but a Map never ORIGINATES a quantity.**
Calculations belong to processes. Consequences:

- **ccov relocated**: upstream computes cloud cover inline in the
  aggregation loop (nonlinear per-HRU: `1 - swrad/soltab·cossl`,
  clamped). That is HRU meteorology -> `ccov_hru` on
  `PRMSAtmosphereBase` (verbatim block, exact pin). Its removal
  leaves every aggregate exactly LINEAR in its hru input with static
  coefficients -- so the whole aggregation rides plain static-weights
  Maps, `MapMPI` inherited unchanged. It also removes the only
  time-dependence (the soltab doy row): Maps stay time-free.
- **Weights by kernel probing** (`derive_aggregation_weights()`): the
  matrices are recovered by driving the pinned verbatim kernels with
  basis vectors -- nothing re-derived by hand (notably the
  ORDER-DEPENDENT seginc_swrad fallback rows, where the
  numerical-order normalization pass can re-divide an
  already-normalized value; probing captures whatever the kernel
  does). Three matrices (flow = area·cfs_conv; met = area-average
  with seg_close copy rows; humid = 0.01·met) serve the ten Maps by
  reference (`AGGREGATION_MAP_SPEC`). A zero-input probe guards the
  -99.9 no-HRUs-anywhere marker (an affine constant no pure-matmul
  Map can produce): loud NotImplementedError; drb is marker-free.
  Pinned: weights vs kernels at 1e-12, all days. Derived at build;
  file-backed weights = the pre-processing-suite backlog item.
- **CBH humidity needs no leaf**: `humidity_hru -> seg_humid` via the
  humid Map IS the strmtemp_humidity_flag=0 configuration (the base
  nhm_stream_temp config, per the aggregates pin), served by the CORE
  PRMSStreamTemp. The SegHumidity leaf covers flag=1. humidity_hru
  itself is external data everywhere -> a 1-variable carrier process.

**Answer-generation finding** (Jul 2026, worth remembering): the
output_stream_temp generation is a DIFFERENT pywatershed run than
output/, and the two differ from EACH OTHER -- identical atmosphere,
but snow knife-edge flips (the fastmath build-reproducibility story,
first at day 80) cascade: 0.2% of snowmelt hru-days (max 1.78 in),
7% of sroff hru-days, smeared by muskingum over 68% of seg_outflow
seg-days (max 59 cfs). pywatershed's own answers disagree at exactly
the level our snow_live criteria allow -- direct confirmation of the
fastmath-answers ceiling. Full-chain stream-temp criterion: family
5e-3 with outlier fraction <= 5e-3 (measured: seg_tave_water 0.16%,
seg_tave_upstream 0.14%, seg_tave_lat 0.025%, and the
air-temperature-driven seg_tave_gw/ss + seg_shade 0.0 -- only
flow-driven variables see the knife-edge cascade).

## Not ported / backlog

- Ag odds and ends: an ObsET standalone MPI test (deferred -- the
  iteration is per-element with a local exit that is provably
  equivalent to upstream's global transp_on.any(); the analysis
  serial test carries the iteration parity); a full 7-process ag
  chain model; PRMSSoilzoneNoDprstAg (upstream doesn't test it
  either); mixed-pref_flow_den domains under MPI (the scalar
  _pref_flow_flag local-any caveat -- see the module comment).
- Stream-temperature chain: DONE (July 2026) -- see "The
  stream-temperature chain" section. Remaining niceties are
  framework-level and already flagged: the batched multi-variable Map
  (10 aggregation Maps share 3 weight matrices by reference today)
  and the weights save/read file option (pre-processing suite, below).
- Legacy/superseded upstream classes NOT planned: standalone Starfit
  (the FlowNode is its successor), PRMSEt.
- Cascades; glaciers; frozen ground; water use.
- **Restart: SERIAL DONE (Aug 2026)**. Design (JLM):
  ``DataArrayMeta(restart=True)`` marks the PROGNOSTIC state (the
  natural cut for state-updating/DA); ``get_restart_variables()``
  derives from the flags (pywatershed's hand lists = the
  cross-check); ``get/set_restart_state()`` hooks carry evolving
  python-attr state (stream temp's gw/ss silos + indices -- the only
  case). ``Model.write_restart(dir)`` writes self-locating per-grid
  files (state timestamp in attrs); ``control["restart_read"]``
  restores, fast-forwards inputs, resumes at the following step
  (istep0 blocks are index-gated and skip naturally). Flags on
  CURRENT variables only -- advance() regenerates the ``*_prev``
  copies. Validation = PERFECT-RESTART tests
  (tests/test_restart.py framework + tests/test_restart_processes.py:
  gw/snow/atmosphere/stream-temp standalones + THE FULL NINE-PROCESS
  CHAIN, all BIT-identical in every variable -- far stronger than
  upstream's own standard). FINDING: pywatershed's snow list is
  incomplete for bitwise restart -- the season/albedo memory
  (iso/lso/mso/int_alb/lst/salb/snsv/ai/albedo) AND **pk_temp**
  (read at step start, not recomputed from pk_def; caught by the
  full-chain test at one knife-edge HRU) are prognostic; ours flags
  all of them.
  **Ag family DONE (Aug 2026)** (tests/test_restart_processes_ag.py,
  fgr): the ag wrinkle is the PER-STEP AREAS -- under time-varying
  ag_frac both processes read the PREVIOUS step's areas at step
  start and the istep0 area blocks are time-zero-gated, so
  hru_perv/hru_frac_perv/ag_area (RunoffAg) and
  hru_area_perv/ag_area (SoilzoneAg) are flagged as prognostic
  markers though not storages; SoilzoneAg storages mirror plain
  soilzone (incl. derived soil_lower; ag_soil_lower for symmetry);
  ObsET adds NOTHING (It0 buffers = per-step scratch, overwritten
  before any read). Tests: flag pins + the live RunoffAg->SoilzoneAg
  chain (spinup/static ag_frac, 90 d) + ObsET standalone (DYNAMIC
  ag_frac + AET iteration, 120 d), all bit-identical.
  **FlowGraph node types DONE (Aug 2026)**
  (tests/test_restart_flow_graph.py): flags live on the node types'
  shared ``fields`` DataArrayMeta objects, so every composed class
  derives them. prms_channel: seg_inflow + outflow_ts (=
  pywatershed's channel list; advance_type regenerates
  inflow_ts_prev, prepare zeroes the rest). STARFIT family:
  lake_storage_sub (hourly marching storage) + lake_storage (daily
  marching storage; read by starfit_advance_type --
  lake_storage_old is regenerated every advance); starfit_daily
  adds lake_outflow_sub_next (the one-day-lag rate). pass_through/
  obsin/source_sink/combined: NO prognostic state (accumulators
  zeroed in prepare; obsin's latch is within-day). Tests: all-types
  flag pin + drb pure-channel graph (60 d) + the 115 reference
  reservoirs hourly (n_substeps=1) AND daily (n_substeps=24), all
  bit-identical.
  **MPI restart DONE (Aug 2026) -- the restart arc is COMPLETE**
  (tests/test_restart_mpi.py). Design: restart files are
  SERIAL-FORMAT and FULL-extent in both paths, so serial and MPI
  runs warm-start each other (tested). ``ModelMPI.write_restart`` =
  gather-then-write (the distributed grid's flagged vars are
  allgathered -- contiguous rank-ordered blocks -> concatenate =
  global order -- and rank 0 writes, mirroring the rank-0 zarr
  Output stance; collective-uniform loop, Barrier at the end so the
  files are complete before any rank proceeds). Read = every rank
  loads the full-extent files and restores its OWN block
  (``_restore_grid_var`` seam on Model; SPMD-uniform, no
  collectives); serial (replicated) grids restore whole on every
  rank. Resume = a LAZY TIME-SLICE of the input dataset before
  ``parallelize``/``set_streaming`` (JLM's call; mpixarray takes the
  isel view without complaint), so the stream simply BEGINS at the
  resume step: the superset input file serves any restart (no
  duplicate files, no wasted reads) and the warm run's streamed
  output covers exactly the computed window. The resume index is
  located BEFORE the stream is built (``_peek_restart_resume_index``
  reads state_time from the files' attrs; ``_read_restart`` then
  fully validates + restores once the buffers exist). Model time
  stays GLOBALLY indexed (``run()`` offsets stream-local ``tt`` by
  the start index) -- istep0-gated blocks must NOT re-fire at the
  resume step. Python-attr hook
  state on a DISTRIBUTED-grid process raises NotImplementedError
  (no such process exists today; stream temp's silos live on the
  replicated segment grid). Tests: the two-grid toy (distributed
  Upper -> MapMPI -> replicated Lower; + the SERIAL-warm-starts-
  from-MPI-files interop leg) and drb PRMSGroundwater (765 HRUs =
  UNEVEN blocks), truncated-input-file recipe, bit-identical per
  rank in every variable.
  REMAINING (backlog only): write frequencies (pywatershed's
  y/m/d/f) if ever needed -- the recipe needs only final-write.
- Weights save-to/read-from file (JLM, Jul 2026): derive-at-build is
  the current decision for the aggregation weight matrices; a
  file-backed option belongs to a future PRMS PRE-PROCESSING SUITE,
  which must be cleanly separated from this code base. Decided so
  far (Jul 2026): strictly ONE-WAY (PRMS native files ->
  pws_phoenix inputs; no round-trip); lives IN this repo as a module
  for now (separation is logical -- the model code never imports the
  translator -- so it can split out later); pws_phoenix defines the
  input CONTRACT and the suite targets it. The contract is BUILT
  (July/Aug 2026): `Model.input_spec()` (dry-run classmethod from the
  DataArrayMeta declarations; pinned by tests/test_input_spec.py,
  data-free), hierarchy per JLM: required -> grid -> {external
  inputs, parameters, initial values} first; the informational half
  (internal inputs, derived params, map-fed) only on
  `include_optional=True`. Rendered by
  **`examples/00_input_contract.py`** -- a
  py:percent NOTEBOOK (Jupyter/VS Code cells; also runs as a plain
  script) whose visible cells ARE the provenance, incl. the drb
  source-file scan. Translation facts surfaced: humidity CBH =
  `cbh.nc:rhavg` (pywatershed renames it humidity_hru);
  hru_slope/hru_lat/etc live in BOTH parameters_dis_both.nc and
  parameters_dis_hru.nc; CBH float32 -> f64 widening happens at
  input preparation (exact; a translator responsibility). IC status:
  only TWO `initial=` seams (gwstor_init, segment_flow_init); other
  ICs are ordinary `*_init*` PARAMETERS consumed by initialize().

  **prms_translate/ BUILT (Aug 2026) -- the legacy endpoint reached.**
  The translation layer lives at `prms_translate/` under the RULE
  (JLM): decoding leans on **pyPRMS exclusively** (never
  pywatershed; pywatershed = test ORACLE only), so the generic
  capabilities can migrate INTO pyPRMS later. Modules: `readers.py`
  (control + parameters at FULL precision -- pyPRMS parses PRMS
  float params as float32 at the text line, upstream ask #1; the
  patch + metadata-widen combination is pinned EXACT vs
  pywatershed's float64 parse -- plus (nmonth, nhru) transpose,
  int64 widen); `control.py` (modules+flags -> classes, pure
  function, NHM order; dprst -> NoDprst trio; stream-temp flags ->
  leaf; loud NotImplementedError on anything unported;
  init_vars_from_file raises toward our own restart); `cbh.py`
  (ASCII -> f64 (time, nhru) at pws names; pyPRMS itself does
  rhavg -> humidity_hru); `dyn_param.py` (dynamic-parameter reader
  REWRITTEN pyPRMS-shaped -- pyPRMS has none, upstream ask #2 --
  pinned vs pywatershed's); `parameters.py` (contract-driven split
  of the flat namespace by the class declarations -- no hand lists;
  positional dim-rename rule gives one->scalar / nmonths->nmonth
  for free; derives hru_in_to_cf = hru_area*43560/12 [pinned
  exact]; collapses PRMS's nhru-dimensioned snow densities to the
  declared scalars IFF uniform, raises otherwise;
  volume_map_weights). Validation: tests/test_prms_translate.py
  (resolution pins data-free + bitwise decode parity). ENDPOINT:
  `examples/01_prms_legacy_translation.py` -- the complete NHM +
  stream-temp model assembled and run from NOTHING but
  nhm_stream_temp.control + myparam.param + the four .cbh ASCII
  files, contract-driven (input_spec loops), verified vs the PRMS
  answers (day-5 means match, 5e-3 elementwise, sentinel segment
  reproduced) -- ran green on the FIRST try. NOT yet wired:
  dynamic-parameter/ag assembly (fgr resolves loudly), sf_data
  (obsin), yaml serialization of the assembled config (converges
  with Options/from_yaml backlog).

  **How variants are done here** (deliberate stance): pywatershed
  derives these by SUBTRACTIVE subclassing — the parent is the
  kitchen sink and children remove fields (re-declaring the whole
  interface, double-running init, and feeding the parent kernel ~30
  per-step zero arrays for the disabled physics). Not here. A variant
  is either (a) a sibling leaf class sharing module-level njit
  kernels (the STARFIT node-type precedent), (b) an option-composing
  factory (the `make_flow_graph` precedent), or (c) if a real family
  core emerges, a MINIMAL base that variants extend by ADDING fields
  (`DataArrayMeta` class attributes inherit additively; subtracting
  an inherited field is nearly unwritable — a feature). The general
  case must never depend on the specific one.

  **Realized for NoDprst** (July 2026, pattern (c)):
  `PRMSGroundwaterNoDprst` / `PRMSRunoffNoDprst` /
  `PRMSSoilzoneNoDprst` are now the minimal BASE classes in the same
  modules; the full classes EXTEND them by adding dprst declarations
  and overriding `initialize` (runoff; soilzone via the ONE
  `_set_hru_frac_perv` hook), `advance` (runoff), and the kernel
  (each class owns its kernel over the shared module njit helpers;
  the full kernels' bodies are untouched by the restructure). The
  framework walks the MRO for `DataArrayMeta` declarations, so the
  additions are pure. Validation: `tests/test_prms_no_dprst.py` pins
  each NoDprst class bit-for-bit (assert_array_equal, full drb
  period) against its full class run with dprst disabled by data
  (`dprst_frac = 0` / zero-dprst inputs) — no external answers
  needed. `tests/test_prms_no_dprst_parity.py` compares against
  pywatershed's nhm_no_dprst simulation answers and SKIPS until those
  are generated into `test_data/drb_2yr/output_no_dprst/`.
- Budget / ConservativeProcess: a deliberate later design pass.
- Southern-hemisphere domains (atmosphere `is_summer`).
- Nonzero `snowpack_init` (upstream's implementation is faulty).
- Annual (1-month) CBH-adjustment parameter variants.
