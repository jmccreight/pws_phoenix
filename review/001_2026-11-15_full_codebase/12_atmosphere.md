<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 12: atmosphere (PRMSAtmosphere family, solar geometry)](#slice-12-atmosphere-prmsatmosphere-family-solar-geometry)
  - [Findings](#findings)
    - [12.1 The first-step block of `_transp_tindex` tests the day of the MONTH; pywatershed tests the day of the YEAR; the difference has no comment and no entry in `docs/MAINTENANCE.md`](#121-the-first-step-block-of-_transp_tindex-tests-the-day-of-the-month-pywatershed-tests-the-day-of-the-year-the-difference-has-no-comment-and-no-entry-in-docsmaintenancemd)
    - [12.2 Three temperature differences are declared with the absolute unit `degF`](#122-three-temperature-differences-are-declared-with-the-absolute-unit-degf)
    - [12.3 `transp_tmax` is declared `degF` and is also converted by `temp_units` in `initialize()`: two mechanisms for one conversion](#123-transp_tmax-is-declared-degf-and-is-also-converted-by-temp_units-in-initialize-two-mechanisms-for-one-conversion)
    - [12.4 In `compute_soltab` `t3 = t7` and `t2 = t6` are the same arrays, not copies, so the two "second daylight period" blocks can never run; pywatershed's quirk, kept with no comment](#124-in-compute_soltab-t3--t7-and-t2--t6-are-the-same-arrays-not-copies-so-the-two-second-daylight-period-blocks-can-never-run-pywatersheds-quirk-kept-with-no-comment)
    - [12.5 The hemisphere guard asks whether ANY HRU is north of the equator](#125-the-hemisphere-guard-asks-whether-any-hru-is-north-of-the-equator)
    - [12.6 `transp_on` is `restart=True` in the frost leaves, where the kernel assigns it on every HRU every step](#126-transp_on-is-restarttrue-in-the-frost-leaves-where-the-kernel-assigns-it-on-every-hru-every-step)
    - [12.7 The kernels index rows by day of year and by month with no check of the first dimension; the docstring's "the declared dims reject it loudly" is not what happens](#127-the-kernels-index-rows-by-day-of-year-and-by-month-with-no-check-of-the-first-dimension-the-docstrings-the-declared-dims-reject-it-loudly-is-not-what-happens)
    - [12.8 `compute_soltab` builds four tiled `(366, nhru)` arrays in every `func3` call and evaluates the flat-surface table for every HRU](#128-compute_soltab-builds-four-tiled-366-nhru-arrays-in-every-func3-call-and-evaluates-the-flat-surface-table-for-every-hru)
    - [12.9 Eight statements in docstrings, descriptions and comments do not match the code](#129-eight-statements-in-docstrings-descriptions-and-comments-do-not-match-the-code)
    - [12.10 Cleanup: a second pass over the HRUs for cloud cover, constants repeated across modules](#1210-cleanup-a-second-pass-over-the-hrus-for-cloud-cover-constants-repeated-across-modules)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Questions handed to this slice](#questions-handed-to-this-slice)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 12: atmosphere (PRMSAtmosphere family, solar geometry)

- pws_phoenix commit: 359563f (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Paths: `pws_phoenix/atmosphere/prms_atmosphere.py`,
  `prms_solar_geometry.py`, `solar_constants.py`, `__init__.py`
- Checklist: B. Reviewed 2026-11-18.
- Method: `/code-review high` on the directory (10 candidates and 4
  left out by its cap). Kept: 9, as 12.1-12.7, 12.9, 12.10. Changed:
  its candidate "`func3(x2, x1, t3, t2)` is evaluated three times" is
  wrong, because two of the three calls sit in blocks that never run
  (12.4); what is left of it is 12.8. Its inactive-HRU candidate is
  in the notes, not a finding (7.2). Then both modules beside their
  pywatershed sources (`pywatershed/atmosphere/prms_atmosphere.py`
  `:171-209`, `:324-805`, `:936-1188`; `prms_solar_geometry.py`
  `:138-420`; `solar_constants.py`; `base/control.py:474-482`;
  `base/active_hru_mixin.py:1-80`). Outside the slice, through
  CodeGraph and direct reads: `pws_phoenix/units.py:56-290`,
  `pws_phoenix/model.py:480-513`,
  `pws_phoenix/prms_translate/prms_units.py:36-150`,
  `pws_phoenix/hydrology/prms_stream_temp.py:1724-1735`, the
  `mutable_input` declarers, `docs/MAINTENANCE.md`, `docs/PORTS.md`
  (the atmosphere entries), findings 3.1, 3.7, 3.9, 7.2, 8.1, 11.2,
  11.4, 11.5, `coverage_serial.txt:73-74`.
- Not read: the PRMS Fortran (`transp_tindex.f90`, `soltab.f90`).
  Where a finding says what PRMS does, it says so from memory and
  the fix phase must read the source.
- Nothing was run. Every consequence below is from reading.

## Findings

### 12.1 The first-step block of `_transp_tindex` tests the day of the MONTH; pywatershed tests the day of the YEAR; the difference has no comment and no entry in `docs/MAINTENANCE.md`

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:801-808`, `:855`
- Code, the port:

      if istep0 == 1:
          # upstream "time zero calculations" (start = this step)
          motmp = current_month + 12
          if current_month == transp_beg[ii]:
              if current_dom > 10:
                  transp_on[ii] = 1.0
              else:
                  transp_check[ii] = 1
      ...
      np.int64(time.day_of_month),

  pywatershed (`prms_atmosphere.py:728-740`, `base/control.py:474-477`):

      start_day = self.control.start_doy
      ...
      if start_month == self.transp_beg[hh]:
          # rsr, why 10? if transp_tmax < 300, should be < 10
          if start_day > 10:
              self.transp_on.data[0, hh] = 1
          else:
              transp_check[hh] = 1

      def start_doy(self) -> int:
          """The simulation start day of year."""
          return datetime_doy(self._start_time)

- Consequence: the two agree when the run starts in January (day of
  year == day of month) or after the 10th of any month. They differ
  for a run that starts on day 1-10 of a month other than January, on
  every HRU whose `transp_beg` is that month. Start 1980-04-05 with
  `transp_beg == 4`: pywatershed has `start_doy = 96 > 10` and sets
  `transp_on = 1` on step 0; the port sets `transp_check = 1` and
  leaves `transp_on = 0` until `tmax_sum` passes `transp_tmax_f`.
  `transp_on` differs for days to weeks, and with it the canopy and
  soil zone evapotranspiration. The port's choice is probably PRMS's
  (the day of the month; not read here), and pywatershed's probably
  the bug, but the comment says `# upstream "time zero
  calculations"` and nothing records a departure. No test sees it:
  `coverage_serial.txt:73` lists `:806` (`transp_on[ii] = 1.0`) and
  `:809-818` as never run, so every test starts on day 1-10 of a
  month equal to `transp_beg` on every HRU.
- Source: /code-review
- Outcome: open

### 12.2 Three temperature differences are declared with the absolute unit `degF`

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:129-142`,
  `:150-156`, `:517-518`, `:526`; `pws_phoenix/units.py:86-91`
- Code:

      tmax_cbh_adj = DataArrayMeta(
          ...
          units="degF",
          description="Monthly adjustment to CBH maximum temperature [degF]",
      tmax_allrain_offset = DataArrayMeta(
          ...
          units="degF",
          description="Offset above tmax_allsnow for all-rain [degF]",
      ...
      tmaxf[ii] = tmax[ii] + tmax_cbh_adj[mm, ii]
      tminf[ii] = tmin[ii] + tmin_cbh_adj[mm, ii]
      tmax_allrain = tmax_allsnow[mm, ii] + tmax_allrain_offset[mm, ii]

  the vocabulary:

      "degF": "degF",
      "degC": "degC",
      "delta_degC": "delta_degC",
      "1/delta_degF": "1 / delta_degF",

- Consequence: `tmax_cbh_adj`, `tmin_cbh_adj` and
  `tmax_allrain_offset` are added to a temperature, so they are
  differences. `CLAUDE.md` ("Units mode"): "A temperature difference
  is `delta_degF` / `delta_degC`"; `prms_units.py:64-65` already
  maps `lat_temp_adj` to `delta_degC` for this reason ("an adjustment
  ADDED to a temperature"). With `control["units"] = "convert"` and
  one of the three labelled `degC`, `units.affine("degC", "degF")`
  is (1.8, 32): an adjustment of 0 becomes 32 degF added to every
  `tmax`, with no message. The vocabulary has no `delta_degF` key,
  so an array correctly labelled as a difference in degF cannot be
  supplied in units mode at all. PRMSSnow does not declare
  `tmax_allrain_offset`, so only this module needs the change; the
  PRMS boundary then needs three by-name entries, because PRMS's
  metadata gives these the same string as an absolute temperature
  (to confirm in slice 14). The future kernel units test also
  depends on it: pint is not expected to add `degF` to `degF`.
- Source: /code-review
- Outcome: open

### 12.3 `transp_tmax` is declared `degF` and is also converted by `temp_units` in `initialize()`: two mechanisms for one conversion

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:727-739`,
  `:775-781`
- Code:

      transp_tmax = DataArrayMeta(
          ...
          units="degF",
          description="Cumulative tmax to trigger transpiration [temp_units]",
      ...
      if int(refs["temp_units"][0]) == 0:
          refs["transp_tmax_f"][:] = refs["transp_tmax"]
      else:
          refs["transp_tmax_f"][:] = (
              refs["transp_tmax"] * (9.0 / 5.0)
          ) + 32.0

- Consequence: `temp_units = 1` and `transp_tmax` supplied in degC.
  With `control["units"] = "convert"` and the array labelled `degC`,
  the Model converts the resident array to degF in place at build
  (`units.reconcile`), then `initialize()` converts it again: 260
  becomes 500, then 932, and transpiration starts weeks late or not
  at all. With `control["units"] = "check"` the correctly labelled
  array raises. So the `else` branch gives the right answer only
  with units mode off, or with a degC array labelled `degF`. It is
  pywatershed's branch verbatim (`:722-725`, "candidate for worst
  code lines"), and it is the only parameter `temp_units` converts:
  with `temp_units = 1` every other temperature parameter and the
  CBH inputs would be degC too, and nothing converts them. The
  branch is never run (`coverage_serial.txt:73`, `:779`). If PRMS's
  metadata string for `transp_tmax` is `temp_units` (to confirm in
  slice 14), `prms_units.resolve` returns `degC` for a metric file
  and `package_parameters` raises against the declaration, which is
  the consistent behavior: the fix is to drop `temp_units` and the
  branch, or to guard `temp_units != 0` with `NotImplementedError`.
- Source: /code-review
- Outcome: open

### 12.4 In `compute_soltab` `t3 = t7` and `t2 = t6` are the same arrays, not copies, so the two "second daylight period" blocks can never run; pywatershed's quirk, kept with no comment

- Where: `pws_phoenix/atmosphere/prms_solar_geometry.py:154-190`
- Code (identical in pywatershed, `prms_solar_geometry.py:278-317`):

      t3 = t7
      wh_t7_gt_t1 = np.where(t7 > t1)
      if len(wh_t7_gt_t1[0]) > 0:
          t3[wh_t7_gt_t1] = t1[wh_t7_gt_t1]

      t2 = t6
      wh_t6_lt_t0 = np.where(t6 < t0)
      if len(wh_t6_lt_t0[0]) > 0:
          t2[wh_t6_lt_t0] = t0[wh_t6_lt_t0]

      t6 = t6 + two_pi
      t7 = t7 - two_pi
      ...
      wh_t7_gt_t0 = np.where(t7 > t0)
      if len(wh_t7_gt_t0[0]):
          solt[wh_t7_gt_t0] = (
      ...
      wh_t6_lt_t1 = np.where(t6 < t1)
      if len(wh_t6_lt_t1[0]):
          solt[wh_t6_lt_t1] = (

- Consequence: the assignment through `t3` also changes `t7`, so
  `t7 <= t1 <= pi` everywhere before `t7 = t7 - two_pi`; the new `t7`
  is `<= -pi <= t0`, and `t7 > t0` is never true. In the same way
  the old `t6 >= t0 >= -pi`, the new `t6 >= pi >= t1`, and `t6 < t1`
  is never true. Both blocks are dead in both codes:
  `coverage_serial.txt:74` lists `:178-182` and `:186-190` as never
  run. With copies (`t3 = t7.copy()`), `t6 + two_pi < t1` is possible
  where `t6 < t1 - two_pi`, which needs `x2 > pi / 2`, that is
  `d1 < 0`: `hru_slope * tan(hru_lat) * cos(hru_aspect) > 1`, a
  steep slope facing the pole at high latitude (slope above 0.87 at
  49 N facing due north; above 0.58 at 60 N). For those HRUs the
  tables hold the single-period value. From memory, PRMS adds 2 pi
  to the unclamped `t6` and `t7` and does take these branches; if
  so, pywatershed and the port differ from PRMS there (`:141`, the
  `d1 < 0` line, is never run on the test domains either). The port
  matches pywatershed, so its 1e-10 test passes. `CLAUDE.md` asks
  that a kept upstream quirk carry a comment; here the docstring
  says "pywatershed compute_soltab verbatim" and nothing marks the
  aliasing or the unreachable blocks.
- Source: /code-review (the alias); checklist B and coverage (that
  the blocks cannot run)
- Outcome: open

### 12.5 The hemisphere guard asks whether ANY HRU is north of the equator

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:447-452`, `:513`
- Code:

      # upstream's is_summer logic assumes the northern hemisphere
      if not (refs["hru_lat"] > 0.0).any():
          raise NotImplementedError(
              "PRMSAtmosphere: southern-hemisphere domains are not "
              "implemented (upstream raises the same)"
          )
      ...
      is_summer = (current_doy >= 79) and (current_doy <= 265)

  pywatershed (`prms_atmosphere.py:548-554`):

      northern_hemisphere = hru_lat > zero
      if not any(northern_hemisphere):

- Consequence: serial, a domain with one northern HRU and the rest
  southern passes, and the kernel applies the northern `is_summer`
  to every HRU: `radj_sppt` and `radj_wppt` are swapped on the
  southern ones with no message. A domain with `hru_lat == 0.0`
  everywhere (a toy) raises as "southern-hemisphere". This is
  pywatershed's `any`, kept; the comment does not say that the test
  is `any` and not `all`. The MPI side (one rank's block, a raise on
  one rank alone) is 3.7 and is not repeated here. `:449` is never
  run (`coverage_serial.txt:73`).
- Source: /code-review
- Outcome: open

### 12.6 `transp_on` is `restart=True` in the frost leaves, where the kernel assigns it on every HRU every step

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:413-420`,
  `:892-896`
- Code:

      transp_on = DataArrayMeta(
          kind="variable",
          restart=True,
      ...
      for ii in range(transp_on.shape[0]):
          if (jsol >= spring_frost[ii]) and (jsol <= fall_frost[ii]):
              transp_on[ii] = 1.0
          else:
              transp_on[ii] = 0.0

- Consequence: in PRMSAtmosphereTranspFrost and
  PRMSAtmosphereTranspFrostDyn `transp_on` is not state. It is
  written to every restart file and read back for nothing; a
  `transp_on` supplied as initial state is accepted, checked for
  units and discarded by the first `calculate()`; the model contract
  lists it as initial state. The same pattern as 11.4 (`ai`).
  pywatershed's frost classes inherit `["tmax_sum", "transp_on"]`
  too, so this is upstream's list, kept. In the tindex leaf all
  three (`transp_on`, `tmax_sum`, `transp_check`) are state; the
  port adds `transp_check`, which pywatershed holds in a local
  variable and does not save. Fix shape: declare `transp_on` without
  `restart` in the base and redeclare it in PRMSAtmosphere.
- Source: /code-review
- Outcome: open

### 12.7 The kernels index rows by day of year and by month with no check of the first dimension; the docstring's "the declared dims reject it loudly" is not what happens

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:34-37`,
  `:607-610`, `:695`, `:517-518` (and eleven more `[mm, ii]` reads)
- Code:

      swrad[ii] = (
          soltab_potsw[current_doy - 1, ii] * radadj / hru_cossl[ii]
      )
      orad_hru[ii] = radadj * soltab_horad_potsw[current_doy - 1, ii]
      ...
      tmaxf[ii] = tmax[ii] + tmax_cbh_adj[mm, ii]

  the docstring:

      the "1-month" (annual) cbh-adjustment parameter variant
      upstream supports is NOT ported (nhm domains are monthly) -- the
      declared dims reject it loudly.

- Consequence: the atmosphere case of 3.1 and 11.5. A supplied
  parameter's dims are never compared with its declaration, and
  numba does not check bounds. Solar tables supplied with 365 rows
  (consistent with each other and with snow's, so xarray sees no
  size conflict): on Dec 31 of a leap year `time.doy == 366`, and
  `swrad`, `orad_hru`, `potet` and `ccov_hru` take whatever memory
  follows the buffer. `compute_soltabs` always returns 366 rows, so
  this needs tables from another source. A one-row monthly parameter
  is stopped only if its dim is named `nmonth` and another parameter
  of the grid has 12 (xarray's size conflict at assembly, expected,
  not run); under any other dim name it is read past its end from
  February on. `_SOLF[kp]` is safe: `1 <= kp <= 25` on a table of
  26.
- Source: /code-review
- Outcome: open

### 12.8 `compute_soltab` builds four tiled `(366, nhru)` arrays in every `func3` call and evaluates the flat-surface table for every HRU

- Where: `pws_phoenix/atmosphere/prms_solar_geometry.py:57-58`,
  `:65-66`, `:93-97`, `:193-197`, `:232-238`
- Code:

      vv = np.tile(v, (NDOY, 1))
      ww = np.tile(w, (NDOY, 1))
      rr = np.transpose(np.tile(r1, (nhru, 1)))
      dd = np.transpose(np.tile(solar_declination, (nhru, 1)))
      ...
      mask_sl_lt_dnearzero = tile_space_to_time(np.abs(sl)) < _DNEARZERO
      solt = np.where(
          mask_sl_lt_dnearzero, func3(np.zeros(nhru), x0, t1, t0), solt
      )

- Consequence: build time and memory only; the values are right. One
  `(366, nhru)` float64 array is 336 MB at 114,958 HRUs. During the
  `func3` call at `:173` of the second `compute_soltab` call about 15
  such arrays are alive (the first call's table, `t0`, `t1`, the two
  clamped arrays, the new `t6` and `t7`, the four tiles, the
  expression temporaries): roughly 5 GB, counted from the code, not
  measured. `func3` runs four times per `compute_soltabs` (two per
  `compute_soltab`; the calls at `:179-180` and `:187-188` never
  run, 12.4). Broadcasting (`v[None, :]`, `r1[:, None]`,
  `solar_declination[:, None]`) gives the same values element for
  element with no tiles. The module docstring says the numpy is
  "verbatim" and init-time, so the per-step rules do not apply; the
  point here is the NHM-scale build.
- Source: /code-review, corrected by the read
- Outcome: open

### 12.9 Eight statements in docstrings, descriptions and comments do not match the code

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:36-37`, `:42-43`,
  `:59-62`, `:207-220`, `:461`, `:732`, `:802`;
  `pws_phoenix/atmosphere/prms_solar_geometry.py:33-36`
- Code and consequence, one by one:
  1. `Deliberately NOT ported: Budget; adapters; restart (upstream's
     own comment: "The restart capability in PRMS does not work")`.
     Restart is ported: `transp_on`, `tmax_sum`, `transp_check` are
     `restart=True` (`:415`, `:753`, `:762`), one more than
     pywatershed's `["tmax_sum", "transp_on"]`. And pywatershed's
     atmosphere has no budget to leave out (no "budget" in
     `pywatershed/atmosphere/`).
  2. `unused declared params (doy, hru_area, hru_aspect)`, and in the
     solar module `the unused doy / radj_sppt / radj_wppt / hru_area
     parameter declarations`: both lists leave out `hru_type`, which
     pywatershed declares in both classes and USES
     (`_mask_inactive_hrus`). See the notes (7.2).
  3. `the declared dims reject it loudly`: 12.7.
  4. `Southern-hemisphere domains raise NotImplementedError`: only a
     domain with no northern HRU at all (12.5).
  5. `radadj_intcp`: `units="dday"`, description `[-]`.
     `radadj_slope`: `units="dday/delta_degF"`, description
     `[1/degF]`. The units follow PRMS's metadata
     (`prms_units.py:61-62`); the kernel uses the result as a
     fraction: `pptadj = radadj_intcp + radadj_slope * (tmaxf -
     tmax_index)`, compared with 1.0, replaced by `radj_sppt` /
     `radj_wppt` (`fraction`), multiplied into `radadj`.
  6. `transp_tmax`: `units="degF"`, description `[temp_units]`
     (12.3).
  7. `pass  # transp_on/tmax_sum/transp_check persist` in
     `PRMSAtmosphereBase.advance`: the base has only `transp_on`.
  8. `# upstream "time zero calculations"`: not upstream's test
     (12.1).
- Source: /code-review, extended by the read
- Outcome: open

### 12.10 Cleanup: a second pass over the HRUs for cloud cover, constants repeated across modules

- Where: `pws_phoenix/atmosphere/prms_atmosphere.py:89-91`,
  `:667-704`; `pws_phoenix/atmosphere/prms_solar_geometry.py:54`
- Code:

      self._compute_ccov_hru(
          refs["ccov_hru"],
          refs["swrad"],
          refs["soltab_potsw"],
          refs["hru_cossl"],
          np.int64(time.doy),
      )
      ...
      @numba.njit(cache=NUMBA_CACHE)
      def _compute_ccov_hru(

- Consequence: none for the answers. `_compute_ccov_hru` is a second
  loop over every HRU each step, not parallel, reading the same row
  of `soltab_potsw` the main kernel just read; its five lines fit
  inside the `prange` loop after `swrad[ii]`. It is kept separate as
  the "VERBATIM" copy of the stream-temperature block, which still
  exists at `pws_phoenix/hydrology/prms_stream_temp.py:1724-1735`
  (slice 13). `_NEARZERO`, `_INCH2CM` and `_DNEARZERO` repeat
  constants defined in other modules (11.8).
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations: the base kernel follows pywatershed's
  `adjust_temperature`, `adjust_precip`, `_ddsolrad_run` and
  `_potet_jh_run` expression for expression. The masks applied in
  reverse upstream collapse to the port's sequential `if`s (`prmx`:
  clip, all-rain, all-snow, no precipitation; amounts: all-snow,
  all-rain, mixed; `pptadj`: the four nested conditions). A nan
  `prmx` takes the mixed branch in both codes. `_SOLF` equals
  `solf`; `solar_constants.py` equals pywatershed's apart from
  `NDOY` being local. `_transp_tindex` follows `calc_transp_tindex`
  apart from 12.1; the copy of step `tt - 1` is the persistent
  array. `_transp_frost` is `np.where(...)` per element.
  `compute_soltab`, `compute_t` and `func3` are pywatershed's text
  (12.4 included).
- Allocation, return, fastmath: no kernel returns a value or
  allocates; scalars assigned inside the `prange` loop are private
  to an iteration. No fastmath. `calculate()` builds three or four
  `np.int64` scalars per step. `compute_soltabs` is build-time
  (12.8).
- Frozen parameter edited: none. `hru_cossl` and `transp_tmax_f` are
  `parameter_internal`.
- Units: every float field declares one; three are the wrong kind
  of temperature unit (12.2), `transp_tmax` is 12.3,
  `radadj_intcp` / `radadj_slope` are 12.9 item 5. Budget: none
  declared, and pywatershed has none.
- Quirks and departures: `docs/MAINTENANCE.md` has no atmosphere
  entry. One unlogged departure (12.1); quirks kept with no comment
  or a partial one: 12.4, 12.5.
- Restart: none missing; `transp_on` is extra in the frost leaves
  (12.6).

## Questions handed to this slice

- 3.7 (the `hru_lat` guard per rank): it is the only `raise` in the
  package (`:449`); `compute_soltab` only warns (`:205`). Nothing to
  add beyond 12.5.
- 3.9 (int64 state is float64 with a nan fill under MPI):
  `transp_check` is the only int state. `initialize()` sets it to 0
  before any read; it is only compared with 1 and assigned 0 or 1,
  which is exact in float64. `transp_beg` / `transp_end` are
  compared with `==` only.
- 8.1 ("Slice 12 (or the fix of 8.1)"): the changeover flag is the
  canopy's and runoff's. Nothing in the atmosphere reads or writes
  it; the note belongs to the fix of 8.1.
- 11.2 (`orad_hru` and `swrad` on a day with a zero solar table):
  the atmosphere divides by no table value. `swrad` divides by
  `hru_cossl = cos(arctan(hru_slope))`, which is above 0 for a
  finite slope; `elh` is 0 only at 1056 degC; `tdiff` is guarded.
  With a zero row: `swrad = 0`, `orad_hru = 0`, `potet = 0` (or
  `-0.0`, which `potet < 0.0` does not catch), `ccov_hru = 1`. From
  `compute_soltabs`, `soltab_horad_potsw` is 0 only where `compute_t`
  returns 0 (`tx > 1`), that is polar night, a latitude above about
  66.6 N. So 11.2's `ZeroDivisionError` in snow needs such an HRU or
  a table from another source.

## Notes for later slices

- Inactive HRUs (7.2; no finding per process). The atmosphere needs:
  a `hru_type` declaration (there is none in the package); the four
  loops cover every HRU (`:515`, `:694`, `:800`, `:892`);
  `compute_soltabs` takes no `hru_type`. pywatershed computes every
  HRU and then masks (`prms_atmosphere.py:204`,
  `prms_solar_geometry.py:182`), so the inactive HRUs hold the fill
  value in every atmosphere variable and in the three tables.
  Docstrings to correct with the feature: 12.9 item 2.
- Slice 13 (stream temperature): the cloud-cover block is still in
  `prms_stream_temp.py:1724-1735` beside `_compute_ccov_hru`
  (12.10); `ccov_hru` is 1.0 on a zero-table day (above).
- Slice 14 (translation): PRMS's metadata strings for
  `tmax_cbh_adj`, `tmin_cbh_adj`, `tmax_allrain_offset` (12.2) and
  `transp_tmax` (12.3); `temp_units` is translated as a parameter
  and as a flag. PRMSAtmosphereTranspFrostDyn makes BOTH frost
  dates inputs, where pywatershed lets either one be dynamic
  (`prms_atmosphere.py:1144-1152`): check what
  `prms_translate/control.py:189-204` does when only one
  `dyn_*frost_flag` is set, and who tiles the static one.
  pywatershed reads the dynamic frost dates as int; the port's
  inputs are float64.
- Slice 15: no kernel units test for the atmosphere (known before
  the start). Before one can be written: 12.2 (`degF + degF`),
  `dday_slope[mm, ii] * tmaxf[ii]` (`1/delta_degF` times an absolute
  `degF`), 12.9 item 5, the Jensen-Haise line (degC and degF mixed,
  as its comment says), and bare literals assigned into or compared
  with dimensional values: `tdiff = 1.0e-4` (`:529`), `hru_rain[ii]
  = 0.0` (`:552`), `hru_snow[ii] = 0.0` (`:556`), `potet[ii] = 0.0`
  (`:620`), `tmax_sum[ii] = 0.0` (`:827`, `:831`, `:841`),
  `tmaxf[ii] > 32.0` (`:836`), `hru_ppt[ii] > 0.0` (`:563`).
- Slice 17: is there a restart test of the tindex leaf that stops
  inside a checking period (`transp_check == 1`, `tmax_sum > 0`)?
  That is the case `transp_check` was added for (12.6).
- Slice 19: `coverage_serial.txt:73` (92%) and `:74` (85%). Never
  run in `prms_atmosphere.py`: the guard (`:449`), `ccov = 1.0`
  (`:703`), the `temp_units` branch (`:779`), `transp_on = 1` after
  day 10 (`:806`), the whole `elif` / `else` of the first-step block
  (`:809-818`), and the end of the season (`:825-827`): no test run
  reaches a first of the month equal to `transp_end`, so no test
  turns transpiration off in the tindex leaf. Never run in
  `prms_solar_geometry.py`: `d1 < 0` (`:141`), `t3 < t2`
  (`:169-170`), the two dead blocks (`:178-182`, `:186-190`, 12.4),
  negative radiation (`:204-205`). Why no test caught 12.1: every
  run starts on a day where the two tests agree. 12.4 cannot be
  caught by a comparison with pywatershed.
- Fix phase, to confirm before fixing: read `transp_tindex.f90`
  (12.1) and `soltab.f90` (12.4); a one-process PRMSAtmosphere Model
  started on day 5 of the `transp_beg` month beside pywatershed
  (12.1); `compute_soltabs` on one HRU at 60 N, slope 1.0, aspect 0
  with and without `.copy()` (12.4); peak memory of
  `compute_soltabs` at 114,958 HRUs (12.8).
