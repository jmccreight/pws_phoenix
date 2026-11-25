<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Maintenance ledger](#maintenance-ledger)
  - [PRMSSnow: `calc_snowevap` hands mass to `pkwater_equiv` but not to `pk_ice` / `freeh2o`](#prmssnow-calc_snowevap-hands-mass-to-pkwater_equiv-but-not-to-pk_ice--freeh2o)
  - [PRMSCanopy: bare-soil / lake HRUs drop stored water outside the budget](#prmscanopy-bare-soil--lake-hrus-drop-stored-water-outside-the-budget)
  - [PRMSSoilzone: `compute_interflow` silently gives zero interflow from an inconsistent start state](#prmssoilzone-compute_interflow-silently-gives-zero-interflow-from-an-inconsistent-start-state)
  - [PRMSRunoffAg: areas from the current step's `ag_frac`, before the kernel (departure #3)](#prmsrunoffag-areas-from-the-current-steps-ag_frac-before-the-kernel-departure-3)
  - [Report upstream (pywatershed): SourceSinkFlowNode harvests the last substep's outflow](#report-upstream-pywatershed-sourcesinkflownode-harvests-the-last-substeps-outflow)
  - [Report upstream (pywatershed): `segment_flow_init` is ASSIGNED, not summed, at confluences](#report-upstream-pywatershed-segment_flow_init-is-assigned-not-summed-at-confluences)
  - [Report upstream (pywatershed): a supplied `pref_flow_infil_frac` is not zeroed on inactive, lake and swale HRUs](#report-upstream-pywatershed-a-supplied-pref_flow_infil_frac-is-not-zeroed-on-inactive-lake-and-swale-hrus)
  - [Report upstream (mpixarray): the NetCDF writer re-emits a `_FillValue` inherited on a coordinate](#report-upstream-mpixarray-the-netcdf-writer-re-emits-a-_fillvalue-inherited-on-a-coordinate)
  - [A Model that raises during construction leaves its input files open](#a-model-that-raises-during-construction-leaves-its-input-files-open)
  - [Idea (later): one-off unit check of a whole configuration with `pint`](#idea-later-one-off-unit-check-of-a-whole-configuration-with-pint)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Maintenance ledger

Things to watch for and potentially fix later: known hazards kept
verbatim for parity with pywatershed/PRMS, each with the condition
that makes it bite and the detector that would catch it. One entry
per item; delete the entry when the item is resolved.

## PRMSSnow: `calc_snowevap` hands mass to `pkwater_equiv` but not to `pk_ice` / `freeh2o`

`pws_phoenix/hydrology/prms_snow.py`, end of `calc_snowevap` (PRMS
`snowcomp.f90` verbatim, via pywatershed):

```python
    avail_et = potet - hru_intcpevap - snow_evap
    if avail_et < 0.0:
        snow_evap = snow_evap + avail_et
        pkwater_equiv = pkwater_equiv - avail_et
```

(and the `snow_evap < 0.0` correction just above it). The kernel
tracks `pkwater_equiv` SEPARATELY from `pk_ice + freeh2o`; these
corrections move mass into only one of the two, so the mass budget
(storage terms = `pk_ice_change`, `freeh2o_change`) opens by
`|avail_et|` on any HRU-day that reaches the branch.

- **Reachable only if** `potet_sublim * snowcov_area > 1` or
  `hru_intcpevap > potet`: with `ez = potet_sublim * potet *
  snowcov_area - hru_intcpevap`, `avail_et >= 0` in every branch
  otherwise. PRMS's `potet_sublim` range is below 1 and PRMSCanopy
  bounds `hru_intcpevap` by `potet` (epan_coef hardwired 1.0), so it
  is dead on nhm domains.
- **Detector:** `control["budget"] = "error"` (date, max |residual|,
  indices) -- the PRMSSnow mass budget is declared (Oct 2026).
- **Not fixed** because PRMS and pywatershed carry the same lines and
  no data reach it; changing `pk_ice`/`freeh2o` here is a physics
  decision that would diverge from both on any domain that does. The
  broader irritant -- derived `pkwater_equiv` tracked beside the
  storages it should equal -- is upstream's design. The same
  tracker split DID bite once, in `calc_ppt_to_pack` 1.1.3 (`freeh2o
  = net_rain - pndz`); that one was reachable on drb and is fixed
  (PORTS.md "The snow story").

## PRMSCanopy: bare-soil / lake HRUs drop stored water outside the budget

`pws_phoenix/hydrology/prms_canopy.py`, top of the per-HRU loop in
`_calculate` (PRMS `intcp.f90` verbatim, via pywatershed):

```python
            if hru_type[ii] == _LAKE or cov_type[ii] == _BARESOIL:
                if cov_type[ii] == _BARESOIL and intcpstor > 0.0:
                    extra_water = intcp_stor[ii]
                intcpstor = 0.0
```

`extra_water` is per-canopy (not `* cov`) but lands in the HRU-scale
output `intcp_changeover`, while the store loses `intcpstor * cov`;
the LAKE case zeroes the store with no output term at all. Either
opens the mass budget (declared Oct 2026) by the stored amount on
the HRU-day it runs.

- **Reachable only if** a bare-soil HRU holds water (every
  interception branch is guarded by `cov_type > GRASSES` /
  `== GRASSES`, so its store never leaves zero -- a nonzero
  `intcp_stor` restart on such an HRU is the one way in) or an HRU
  is a LAKE (`hru_type` is hardwired LAND here, as upstream).
- **Detector:** `control["budget"] = "error"`.
- **Not fixed** for the same reason as the snow entry: verbatim
  lines, no data reach them.

## PRMSSoilzone: `compute_interflow` silently gives zero interflow from an inconsistent start state

`pws_phoenix/hydrology/prms_soilzone.py`, `compute_interflow` (PRMS
`soilzone.f90` verbatim, via pywatershed):

```python
        c3 = np.sqrt(coef_lin**2.0 + 4.0 * coef_sq * ssres_in)
        ...
        if (1.0 + c1 * c2) > 0.0:
            inter_flow = ssres_in + ((sos * (1.0 + c1) * c2) / (1.0 + c1 * c2))
        else:
            inter_flow = ssres_in
```

For `ssres_in < -coef_lin**2 / (4 coef_sq)` the square root is nan,
every comparison is False, the `else` sets `inter_flow = ssres_in`
(negative) and the clamp makes it 0: no interflow that step, no
message. For any real `c3` and `storage > 0`, `1 + c1*c2 > 0`, so the
`else` is reached ONLY through the nan. (The `c3 == 0.0` raise needs
`ssres_in` exactly `-coef_lin**2 / (4 coef_sq)`.) The mass budget
stays closed (the storage keeps the water).

- **Reachable only if** a step starts with `slow_stor >
  pref_flow_thrsh` (slow path: `ssresin = pref_flow_thrsh -
  slow_stor`) or `pref_flow_stor > pref_flow_max` (fast path).
  `init_state()` never produces that state and each step preserves
  it, so the one way in is a supplied state (restart, or an initial
  state given under its own name). The Ag family shares the helper.
- **Detector:** none yet. Planned: warn ONCE at run start where
  supplied state enters (placed with the initial-state-eradication
  arc), not a per-step check -- the state cannot arise mid-run.
- **Not changed** in the kernel: verbatim lines; a raise or warning
  inside njit code is not possible (warn) or a departure (raise).

## PRMSRunoffAg: areas from the current step's `ag_frac`, before the kernel (departure #3)

pywatershed's PRMSRunoffAg updates `ag_area`, `hru_perv` and
`hru_frac_perv` AFTER its kernel, so on a step where `ag_frac`
changes its kernel still sees the previous step's areas. Here the
kernel `_areas` computes them from the current step's `ag_frac`
BEFORE the main kernel (`PRMSRunoffAg` docstring in
`pws_phoenix/hydrology/prms_runoff.py`). Reason: a PRMS dynamic
parameter is an input, and a result of an input belongs to the step
that input is for; and with the areas held over from the previous
step, a live RunoffAg -> SoilzoneAg chain never redistributed ag
storage (`docs/design_description.md` "Ag variants").

- **Bites only if** `ag_frac` differs from the previous step's AND
  the changed hrus have rain, melt, infiltration or runoff on that
  step. Otherwise the two orders give the same numbers.
- **Not settled:** which order GSFLOW uses. The fgr_ag_2yr answers
  cannot tell: on the one change date (step 366, 2001-01-01) the 124
  changed hrus are dry. JLM suspects pywatershed's order is the wrong
  one. The GSFLOW Fortran source (where it reads a dynamic parameter
  relative to that step's runoff computation) would settle it.
- **Detector:** `test_runoff_ag_supplied_state` in
  `tests/test_restart_processes_ag.py` pins the order chosen here (it
  fails with pywatershed's order). Nothing compares it with GSFLOW.
- **Status:** not yet reported to pywatershed.

## Report upstream (pywatershed): SourceSinkFlowNode harvests the last substep's outflow

`pywatershed/hydrology/source_sink_flow_node.py:118,134`:

```python
        self._seg_outflow = outflow          # per substep
    ...
    def outflow(self) -> np.float64:
        return self._seg_outflow             # harvested after the last
```

while `FlowGraph.node_upstream_inflows` is the substep MEAN and
`sink_source` is `_sink_source_sum / nsubsteps`. PassThroughFlowNode
reports the mean (`_accum_inflow / (isubstep + 1)`); obsin's outflow
is constant within a day so last == mean. For a source_sink node
spliced into a muskingum network, the daily `node_outflows` therefore
disagrees with `node_upstream_inflows + inflows + node_sink_source`
by the last-minus-mean difference (4.5 cfs on drb day 1). Routing is
unaffected (the substep value is what flows downstream); only the
harvested daily variable is off. Fixed here (departure #2,
`source_sink_flow_node.py` docstring); pywatershed's own global
budget would see it for a source_sink node at an outlet.

- **Detector:** `control["budget"] = "error"` on any composed graph.
- **Status:** not yet reported.

## Report upstream (pywatershed): `segment_flow_init` is ASSIGNED, not summed, at confluences

`pywatershed/hydrology/prms_channel.py`, channel initialization,
carried verbatim in
`pws_phoenix/prms_translate/assemble.py:seg_inflow_from_segment_flow_init`:

```python
    for iseg in range(tosegment0.shape[0]):
        jseg = tosegment0[iseg]
        if jseg < 0:
            continue
        seg_inflow[jseg] = segment_flow_init.values[iseg]
```

A segment with several upstream segments starts with the initial flow
of the LAST one in index order, not their sum.

- **Reachable only if** `segment_flow_init` is nonzero on two or
  more segments that share a downstream segment. It enters the
  first step only (`inflow_ts_prev`, the c1 term of the muskingum
  equation).
- **Detector:** none (the channel declares no budget).
- **Not fixed** because pywatershed's answers carry it; `+=` is the
  one-character fix once upstream agrees.
- **Status:** not yet reported.

## Report upstream (pywatershed): a supplied `pref_flow_infil_frac` is not zeroed on inactive, lake and swale HRUs

PRMS 5.2.1.1, `soilzone.f90`, in the init HRU loop:

```fortran
        IF ( Hru_type(i)==INACTIVE .OR. Hru_type(i)==LAKE .OR. Hru_type(i)==SWALE) THEN
          Pref_flow_den(i) = 0.0
          Pref_flow_infil_frac(i) = 0.0
        ENDIF
```

pywatershed's `PRMSSoilzone` zeroes its copy of `pref_flow_den` where
the HRU is not land and leaves `pref_flow_infil_frac` as supplied;
`pws_phoenix/hydrology/prms_soilzone.py` carries that verbatim, and
its kernel reads `pref_flow_infil_frac` on every HRU.

- **Reachable only if** a parameter file supplies a nonzero
  `pref_flow_infil_frac` on an inactive, lake or swale HRU. The
  effect on such an HRU has not been analyzed.
- **Not reachable** when the file leaves the parameter "not
  specified" (negative on every HRU):
  `prms_translate.prms_parameter_defaults` then builds it from
  `pref_flow_den` WITH PRMS's zeroing.
- **Also wrong upstream:** pywatershed's `PRMSSoilzone` docstring
  says zeros for all HRUs equal "not supplying the parameter to
  PRMS". PRMS uses `pref_flow_den` in that case (`IF (
  Pref_flow_infil_frac(1)<0.0 ) Pref_flow_infil_frac =
  Pref_flow_den`); the two agree only where `pref_flow_den` is zero.
- **Detector:** none.
- **Not fixed** because pywatershed's answers carry it.
- **Status:** found Nov 18, 2026 (the CONUS NHM parameter file leaves
  the parameter at -1); not yet reported.

## Report upstream (mpixarray): the NetCDF writer re-emits a `_FillValue` inherited on a coordinate

The chain: pywatershed stores `time` as float32; xarray's `to_netcdf`
adds `_FillValue = nan` to any float-encoded variable, the `time`
coordinate included; `mpixarray.open_dataset` hoists it into
`time.attrs`. Then in
`mpixarray/file_io/netcdf/nc_DataArray.py:241-246`
`packing.encoded_fill_value` returns it, which bypasses the
coordinate / datetime exclusion (that exclusion guards only the
DEFAULT fill):

```python
np.asarray(nan, dtype=int64)   # "invalid value encountered in cast"
```

a RuntimeWarning at write on every rank, and `cf_pack_attrs` (about
line 287) stamps `time:_FillValue = NaNf` on the int64 time variable,
so xarray warns at read-back ("non-conforming '_FillValue' ...
dropping", rank 0 only).

- **Effect:** warnings only; results are unaffected.
- **Fix belongs in mpixarray:** apply the coordinate / datetime
  exclusion to inherited values too (CF forbids a fill value on a
  coordinate, and xarray drops it anyway).
- **Not worked around here** (decided Oct 16, 2026): a guard in
  `ModelMPI._build`, or `encoding={"time": {"_FillValue": None}}` at
  every `to_netcdf`, would hide it.
- **Status:** diagnosed with a verified reproducer Oct 16, 2026; not
  yet reported.

## A Model that raises during construction leaves its input files open

`Model.__init__` opens files before assembly finishes: file-backed
process_dict entries (`_load_paths_to_data` -> `self._opened_files`),
`Input(path)` (`xr.open_dataarray`) and a `Discretization` given a
parameter path (`xr.open_dataset`). `finalize()` is what closes
them. When construction raises (an unresolved input, a dim mismatch,
supplied state + `restart_read`), no object is returned, so nothing
closes the handles.

Seen Nov 2026: a serial pytest run in which many Models raised
during construction (a since-removed check on unset state) then hung
at ~56% CPU and, on another run, segfaulted, with the stack in
xarray `file_manager._acquire_with_cache_info` -- the same place as
the >128-open-handles hang recorded in PORTS.md. That the leaked
handles were the cause was inferred, not isolated.

- **Reachable only if** many Models fail in construction in one
  Python session.
- **Detector:** a pytest run that stops progressing with the process
  still on the CPU.
- **Fix (not done):** close what was opened when `Model.__init__` /
  `ModelMPI.__init__` raise (a `try/except BaseException: <close>;
  raise`); under MPI the cleanup must not add a collective call.

## Idea (later): one-off unit check of a whole configuration with `pint`

Every budget term now carries `DataArrayMeta.units` (free text), and
the budget check only demands that one budget's terms AGREE. A
stronger, once-per-configuration check: in a serial run, wrap each
field in a `pint` quantity from its `units` and let the kernels'
arithmetic (or a dimensional shadow of it) verify every expression --
e.g. `hru_in_to_cf` conversions, `s_per_time`, `cms_to_cfs` -- rather
than trusting the descriptions. Not a runtime feature (numba kernels
take raw numpy); a diagnostic pass, probably driven from the field
declarations plus the budget/Map wiring, run once when a model is
assembled. To be worked out; parked here on purpose.
