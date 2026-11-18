<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 11: snow (PRMSSnow)](#slice-11-snow-prmssnow)
  - [Findings](#findings)
    - [11.1 PRMSSnow declares no state group, though a pack is only consistent as a set; and a nonzero `snowpack_init` raises, so supplied state is the only way to start with snow](#111-prmssnow-declares-no-state-group-though-a-pack-is-only-consistent-as-a-set-and-a-nonzero-snowpack_init-raises-so-supplied-state-is-the-only-way-to-start-with-snow)
    - [11.2 `trd` is a scalar division inside the kernel: where `soltab_horad_potsw` is 0 the run stops with `ZeroDivisionError`; pywatershed divides arrays and continues](#112-trd-is-a-scalar-division-inside-the-kernel-where-soltab_horad_potsw-is-0-the-run-stops-with-zerodivisionerror-pywatershed-divides-arrays-and-continues)
    - [11.3 Snow does not publish `pkwater_ante` (7.3): canopy's test and pywatershed's agree except when `pkwater_equiv` is a remainder above 2.23e-16 while `pk_ice + freeh2o` is below it](#113-snow-does-not-publish-pkwater_ante-73-canopys-test-and-pywatersheds-agree-except-when-pkwater_equiv-is-a-remainder-above-223e-16-while-pk_ice--freeh2o-is-below-it)
    - [11.4 `ai` is declared `restart=True`; the kernel zeroes it every step, and pywatershed leaves it out of its restart list for that reason](#114-ai-is-declared-restarttrue-the-kernel-zeroes-it-every-step-and-pywatershed-leaves-it-out-of-its-restart-list-for-that-reason)
    - [11.5 Three parameters index arrays inside the kernel and their values and sizes are taken on trust; numba does not check bounds](#115-three-parameters-index-arrays-inside-the-kernel-and-their-values-and-sizes-are-taken-on-trust-numba-does-not-check-bounds)
    - [11.6 Three pywatershed quirks are kept with no comment](#116-three-pywatershed-quirks-are-kept-with-no-comment)
    - [11.7 Six statements in the module docstring do not match the code](#117-six-statements-in-the-module-docstring-do-not-match-the-code)
    - [11.8 Cleanup: constants repeated across modules, conditions kept in pywatershed's layout with no comment](#118-cleanup-constants-repeated-across-modules-conditions-kept-in-pywatersheds-layout-with-no-comment)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Questions handed to this slice](#questions-handed-to-this-slice)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 11: snow (PRMSSnow)

- pws_phoenix commit: 9d370f3 (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Path: `pws_phoenix/hydrology/prms_snow.py`
- Checklist: B. Reviewed 2026-11-18.
- Method: `/code-review high` on the file (9 candidates, all kept
  and merged into 11.1, 11.2 and 11.4-11.8; 11.3 is from the read).
  Then the file beside its pywatershed source
  (`pywatershed/hydrology/prms_snow.py`): the nine helpers
  (`calc_sca_deplcrv` ... `calc_snowevap`) were compared as text with
  comments and whitespace removed, the declarations, `initialize`,
  `advance`, the kernel and `calculate` by reading (`:77-668`,
  `:744-1224` upstream). Outside the slice, through CodeGraph and
  direct reads: the declarers of the snow variables
  (`prms_canopy.py:155-168`, `:418-490`; `prms_runoff.py:1142-1178`;
  `prms_soilzone.py:599-613`), `prms_atmosphere.py:596-612`,
  `prms_solar_geometry.py:60-108`, `:140-165`, `process.py:310-320`,
  `prms_translate/parameters.py:119`, `docs/MAINTENANCE.md:8-41`,
  `docs/PORTS.md` (the snow entries), findings 3.1, 3.7, 3.9 and 7.3,
  pywatershed's `autotest/test_prms_snow.py`, the headers of
  `tests/test_prms_snow*.py`, `coverage_serial.txt:92`.
- Nothing was run. Every consequence below is from reading.

## Findings

### 11.1 PRMSSnow declares no state group, though a pack is only consistent as a set; and a nonzero `snowpack_init` raises, so supplied state is the only way to start with snow

- Where: `pws_phoenix/hydrology/prms_snow.py:1928-1989`, `:160-180`,
  `:1276-1283`; `pws_phoenix/process.py:314`
- Code:

      budgets = {"mass": _MASS_BUDGET}
      # (no state_groups, no check_initial_state)
      ...
      if (refs["snowpack_init"] > 0.0).any():
          raise NotImplementedError(
              "PRMSSnow: nonzero snowpack_init is not ported (upstream "
              "initialization block is faulty; nhm domains use zero)"
          )

  the kernel, on the day the pack first gains heat (`calc_calin`):

      if snowcov_area > 0.0:
          apk_ice = pk_ice / snowcov_area  # [inches]
      ...
      if pmlt > apk_ice:
          snowmelt = snowmelt + pkwater_equiv  # [inches]
          pkwater_equiv = 0.0  # [inches]

  and on a day with sublimation (`calc_snowevap`):

      pk_ice = pk_ice - ez
      if pk_ice < 0.0:
          freeh2o = freeh2o + pk_ice

- Consequence: supply `pkwater_equiv` alone (the natural reading of
  "initial snowpack"). `pk_ice`, `freeh2o`, `pk_den`, `pk_depth`,
  `pss`, `pst` stay 0. On the first day with net energy `apk_ice` is
  0, `pmlt > apk_ice` is true, and the whole supplied pack leaves as
  `snowmelt`; the budget is off by the supplied amount on that day
  (`pk_ice_change + freeh2o_change` is 0 or positive). Before that
  day, each sublimation makes `freeh2o` more negative, and those days
  balance (`snow_evap = ez`, `freeh2o_change = -ez`). With
  `control["budget"]` unset neither gives a message. Canopy
  (`prms_canopy.py:464-466`) reads the pack as `pk_ice_prev +
  freeh2o_prev`, so it sees no pack at all (7.3, 11.3). pywatershed's
  init block (`:408-464`) shows what a start needs: `pk_depth`,
  `pk_den`, `pk_ice`, `freeh2o`, `snowcov_area`, `pss`, `pst` from
  `pkwater_equiv` (it also sets `pk_ice = pkwater_equiv` and then
  `freeh2o = pk_ice * freeh2o_cap`, so its own start has `pk_ice +
  freeh2o > pkwater_equiv`). No document in the repository lists the
  variables a user must supply together; `docs/PORTS.md:698` lists
  nonzero `snowpack_init` only as not supported.
  The guard itself: the module docstring (`:51`) says it raises "if
  any value is nonzero"; a negative or nan `snowpack_init` passes.
  The message does not name supplied state as the way to start with
  a pack. Under MPI it raises on one rank alone (3.7 lists it).
  Neither guard at `:1980` and `:1985` is reached by a test
  (`coverage_serial.txt:92`).
- Source: /code-review, checklist B (the read)
- Outcome: open

### 11.2 `trd` is a scalar division inside the kernel: where `soltab_horad_potsw` is 0 the run stops with `ZeroDivisionError`; pywatershed divides arrays and continues

- Where: `pws_phoenix/hydrology/prms_snow.py:2233-2235`;
  `pws_phoenix/atmosphere/prms_solar_geometry.py:74-75`, `:99-106`;
  `pws_phoenix/atmosphere/prms_atmosphere.py:610`
- Code, the port:

      if pkwater_equiv[jj] > 0.0:
          # STEP 4: radiation fluxes and energy balance
          trd = orad_hru[jj] / soltab_horad_potsw[current_doy - 1, jj]

  pywatershed (`prms_snow.py:785`, before the loop):

      trd = orad_hru / soltab_horad_potsw

  the table (`compute_t`, then `func3` with `x == y == 0`):

      result[np.where(tx > 1.0)] = 0.0
      ...
      np.sin(dd) * np.sin(ww) * (x - y)
      + np.cos(dd) * np.cos(ww) * (np.sin(x + vv) - np.sin(y + vv))

- Consequence: numba's default error model raises
  `ZeroDivisionError` on a float division by zero (as in 6.3, 8.6,
  9.4). On a day when the sun does not rise on an HRU (north of the
  Arctic Circle in winter) the horizontal table is exactly 0, the
  atmosphere gives `orad_hru = radadj * 0`, and an HRU with a
  snowpack reaches `0.0 / 0.0`. The error names no HRU and no day.
  pywatershed gets nan there (numpy array division) and reads `trd`
  only in `calc_snowbal` when `hru_ppt > 0.0` and `tstorm_mo == 1`,
  where a nan fails both `trd > _ONETHIRD` tests. Under MPI one rank
  raises alone (3.7). The drb and ucb test domains have no such HRU.
- Source: /code-review
- Outcome: open

### 11.3 Snow does not publish `pkwater_ante` (7.3): canopy's test and pywatershed's agree except when `pkwater_equiv` is a remainder above 2.23e-16 while `pk_ice + freeh2o` is below it

- Where: `pws_phoenix/hydrology/prms_snow.py:1995-1998`, `:2337-2352`,
  `:184-196`, `:1276-1290`; `pws_phoenix/hydrology/prms_canopy.py:464-466`
- Code, the port:

      def advance(self) -> None:
          refs = self._refs
          refs["freeh2o_prev"][:] = refs["freeh2o"]
          refs["pk_ice_prev"][:] = refs["pk_ice"]

  pywatershed (`prms_snow.py:523-530`):

      self.freeh2o_prev[:] = self.freeh2o
      self.pk_ice_prev[:] = self.pk_ice
      ...
      self.pkwater_ante[:] = self.pkwater_equiv

  the end of the port's first loop:

      if pkwater_equiv[jj] <= _DNEARZERO:
          pkwater_equiv[jj] = 0.0
          ...
          pk_ice[jj] = 0.0
          freeh2o[jj] = 0.0

- Consequence: this is the snow half of 7.3. From the code:
  - When `pkwater_equiv <= 2.23e-16` at the end of a step, all three
    are set to 0, so both tests say "no pack". The two paths that
    skip this block (`LAKE`; no pack and no new snow) leave the three
    as the block left them on an earlier step.
  - The tests can differ only when `pkwater_equiv > 2.23e-16` and
    `pk_ice + freeh2o < 2.23e-16`: pywatershed says "pack", the port
    says "no pack" and lets grass intercept rain. The kernel moves
    `pkwater_equiv` apart from `pk_ice + freeh2o` in three places:
    `calc_calin` 3.2 (`pkwater_equiv = pkwater_equiv - dif_water`
    beside `freeh2o = pwcap`), `calc_snowevap` (`pkwater_equiv =
    pkwater_equiv - ez` beside `pk_ice = pk_ice - ez`), and the two
    corrections at the end of `calc_snowevap`
    (`docs/MAINTENANCE.md:8-41`). The first two differ by roundoff
    (under 1e-14 inches on drb, `:1917`), so a difference needs a
    pack that ends a step within about 1e-14 inches of zero.
  - Supplied state without the full set (11.1) puts every HRU with a
    pack in the differing case.
  So the equivalence does not follow from the code; it holds on drb
  to the extent no pack ends a step that close to zero, which only a
  run can show (the check that slice 7 lists for the fix phase).
  Publishing `pkwater_ante` would be one declared variable and one line in
  `advance`; pywatershed does not put it in its restart list.
  JLM's deep dive on 7.3 is still open.
- Source: checklist B (the read; handed over by slice 7)
- Outcome: open

### 11.4 `ai` is declared `restart=True`; the kernel zeroes it every step, and pywatershed leaves it out of its restart list for that reason

- Where: `pws_phoenix/hydrology/prms_snow.py:1635-1642`, `:2099`,
  `:566-570`
- Code, the port:

      ai = DataArrayMeta(
          kind="variable",
          restart=True,
      ...
      ai[jj] = 0.0
      ...
      ai = pst  # [inches]       (calc_snowcov, before any read)

  pywatershed (`prms_snow.py:295-298`):

      # ... ai is not saved: _calculate
      # zeroes it every step and snowcov() recomputes it as
      # min(pst, snarea_thresh) before any read.

- Consequence: a supplied or restart-read `ai` is accepted, checked
  for units, written to every restart file, and discarded on the
  first step. The model contract lists it as initial state. The
  other 23 `restart=True` variables are pywatershed's list exactly,
  and none is missing.
- Source: /code-review
- Outcome: open

### 11.5 Three parameters index arrays inside the kernel and their values and sizes are taken on trust; numba does not check bounds

- Where: `pws_phoenix/hydrology/prms_snow.py:1978-1980`, `:2181`,
  `:2201`, `:2235`, `:2173`, `:2255`, `:2286`, `:560`, `:113-114`
- Code:

      if refs["snarea_curve"].shape[0] % 11 != 0:
          raise ValueError("snarea_curve length must be a multiple of 11")
      ...
      crv0 = (hru_deplcrv[jj] - 1) * 11
      ...
      snarea_curve=snarea_curve[crv0 : crv0 + 11],
      ...
      trd = orad_hru[jj] / soltab_horad_potsw[current_doy - 1, jj]
      ...
      (tmax_allsnow[month_ind, jj] - 32.0) / 1.8

- Consequence: `hru_deplcrv` of 0 (a 0-based index) or above the
  number of curves gives an empty or short slice, and
  `snarea_curve[11 - 1]` and `snarea_curve[idx - 1]` read memory
  outside it: `snowcov_area` is then arbitrary, with no error, and
  feeds melt, sublimation, runoff and the soil zone. pywatershed
  with 0 takes the last curve (a negative numpy index), also without
  a message. A `soltab_horad_potsw` with 365 rows is read past its
  end on day 366 of a leap year; nothing compares the size of `ndoy`
  with 366 or of `nmonth` with 12 (3.1: a parameter's dims are never
  compared with its declaration). `prms_translate` validates none of
  `hru_deplcrv`, `snarea_curve`.
- Source: /code-review, checklist B (the read)
- Outcome: open

### 11.6 Three pywatershed quirks are kept with no comment

- Where: `pws_phoenix/hydrology/prms_snow.py:2104-2118`, `:2365-2386`
- Code:

      if current_dowy == 1:
          pss[jj] = 0.0  # [inches]
          iso[jj] = 1  # [flag]
          mso[jj] = 1  # [flag]
          lso[jj] = 0  # [counter]
      ...
      if current_doy == melt_force[jj]:
          iso[jj] = 2  # [flag]
      if current_doy == melt_look[jj]:
          mso[jj] = 2  # [flag]
      ...
      if cond1 and cond3 and cond4 and cond5:
          through_rain[jj] = net_ppt[jj]

- Consequence:
  - The season flags are set by equality with the day of year. A
    cold start after `melt_force` (a run that begins June 1 with
    `melt_force = 140`) never sets `iso = 2` in that water year, so
    the spring pack keeps the accumulation-season albedo rules; a
    start between `melt_look` and `melt_force` leaves `mso = 1`.
    pywatershed's comment says "beyond the forced melt day"; its
    code and the port test equality. The remedy (supply `iso` and
    `mso` as initial state) is written nowhere.
  - The reset on day 1 of the water year (October 1) applies to
    every HRU. pywatershed carries a TODO on the southern hemisphere
    there. `docs/PORTS.md:697` lists the southern hemisphere as not
    supported for the atmosphere only.
  - `through_rain`: an all-snow day with `0 < net_snow < 1e-6`, no
    pack, and enough energy to melt it the same day gives `snowmelt
    = net_snow` and, because `snowmelt < _NEARZERO` and `snow_evap <
    _NEARZERO`, also `through_rain = net_ppt = net_snow`: the snow
    leaves twice and the budget is off by `net_snow`. Canopy moves a
    trace `netsnow` to rain only for `cov_type > 1` with `cov > 0`
    (`prms_canopy.py:478-489`), so a bare or grass HRU can deliver
    it. The chain is pywatershed's (`:1160-1185`). Traced by reading
    only; the fix phase should probe it with `control["budget"] =
    "error"`.
- Source: /code-review, checklist B (the read)
- Outcome: open

### 11.7 Six statements in the module docstring do not match the code

- Where: `pws_phoenix/hydrology/prms_snow.py:8`, `:15-23`, `:32-39`,
  `:43-52`
- Code:

      Sixth REAL process port (July 2026)
      ...
      - **(nmonth, space) parameters** (``tmax_allsnow`` -> derived
        ``tmax_allsnow_c``, ``cecn_coef``, ``tstorm_mo``)
      ...
        computed at call time) -- per-HRU variants of den_max/settle_const
        (upstream broadcasts) would surface as a dims mismatch, loudly.
      ...
      Deliberately NOT ported: Budget/ConservativeProcess; adapters;
      restart; calc_method; verbose
      ...
      ``initialize()`` raises NotImplementedError if any value is nonzero

- Consequence:
  1. "NOT ported: Budget ...; restart": the class declares `budgets =
     {"mass": _MASS_BUDGET}` (`:1928`) and 24 `restart=True`
     variables (the same pattern as 7.10).
  2. `tmax_allsnow_c` does not exist; the comment at `:1526-1531`
     says why, and the kernel converts per element (`:2173`).
  3. A per-HRU `den_max`, `den_init` or `settle_const` does not
     raise: the Model compares no parameter dims (3.1) and
     `calculate` takes element 0 (`refs["den_max"][0]`, `:2393-2394`,
     `:2470`), so every HRU gets HRU 0's value. Only
     `prms_translate/parameters.py:119` raises on non-uniform values,
     for parameter sets that pass through it.
  4. The guard tests `> 0.0`, not "nonzero" (11.1).
  5. The numerics are presented as pywatershed's; the one departure
     (`freeh2o = freeh2o + net_rain - pndz`, `:401-412`) is not in
     the docstring, though "what each port kept, dropped or guarded
     is in its module docstring" (`CLAUDE.md`). It is in the comment
     at the line, `docs/PORTS.md` and `docs/MAINTENANCE.md:37-41`.
  6. Two things pywatershed has at ba3974b are absent and not
     listed: `pkwater_ante` (11.3) and the active-HRU loop
     (`ActiveHruMixin`, `for aa in prange(nactive_hrus)`; see the
     notes).
- Source: /code-review, checklist B (the read)
- Outcome: open

### 11.8 Cleanup: constants repeated across modules, conditions kept in pywatershed's layout with no comment

- Where: `pws_phoenix/hydrology/prms_snow.py:68-73`, `:2179`,
  `:2233`, `:2293-2295`
- Code:

      _NEARZERO = 1.0e-6  # nearzero
      _DNEARZERO = 2.23e-16
      ...
      if pkwater_equiv[jj] > 0.0:      # STEP 2, 3
      ...
      if pkwater_equiv[jj] > 0.0:      # STEP 4 (nothing between writes pkwater_equiv)
      ...
      if (transp_on[jj] == 0.0) or (
          transp_on[jj] != 0.0 and cov_type[jj] < 2
      ):

- Consequence: `_NEARZERO` is defined in six modules, `_DNEARZERO` in
  four (`prms_canopy.py:57-60`, `prms_runoff.py:80-83`,
  `prms_solar_geometry.py:54`, here), `_LAKE` in three, `_INCH2CM` in
  two. Snow, canopy and runoff compare the same buffers
  (`pkwater_equiv`, `net_snow`) with these thresholds, so the copies
  must stay equal and nothing ties them. The two consecutive `if`
  blocks and the long condition (equal to `transp_on[jj] == 0.0 or
  cov_type[jj] < 2`) are pywatershed's layout (`:910`, `:981`,
  `:1052-1054`); no comment says they are kept for that reason.
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations: the nine helpers are pywatershed's line for
  line once comments, whitespace and the names of constants are
  removed. The differences: `freeh2o = freeh2o + net_rain - pndz`
  (the logged departure); `not (...)` for `~(...)`; `pptmix_nopack =
  1.0` for `True`; three initial assignments numba needs (`train`,
  `swn`, `cst`); arguments dropped (`calc_calin` / `calc_caloss` /
  `calc_snowbal` passed as functions, `hru_deplcrv` in
  `calc_snowcov`, `acum_init` / `amlt_init`, `verbose`). The kernel
  body follows `_calculate_numpy` step for step; what was array
  staging upstream is per element (`newsnow`, the resets,
  `canopy_covden`, `trd` (11.2), the `tmax_allsnow` conversion, the
  `through_rain` chain in a second loop). The loop covers every HRU
  where pywatershed's covers the active ones (notes).
- Allocation, return, fastmath: the kernel returns nothing and
  allocates nothing; `snarea_curve[crv0 : crv0 + 11]` is a view; the
  helpers return scalars, as the port rules ask. No fastmath.
  `_ACUM_INIT` / `_AMLT_INIT` are module arrays read by
  `calc_snalbedo`.
- Frozen parameter edited: none. pywatershed's edits (`del
  self.tmax_allsnow`, broadcasting `den_max` / `settle_const`,
  `hru_deplcrv.astype`) are replaced by reads (11.7 item 3 on the
  scalars).
- Units: every float field declares one. Budget: pywatershed's terms
  verbatim (`:326-335`); the cases where it does not close are the
  `calc_snowevap` corrections (`docs/MAINTENANCE.md`), a partial
  supplied state (11.1) and the trace-snow case (11.6).
- Quirks and departures: the departure is logged; 11.6 lists the
  quirks with no comment. `calc_snalbedo` and `calc_snowbal` mark
  their unreachable branches as "kept verbatim".
- Restart: no variable is missing; `ai` is extra (11.4).

## Questions handed to this slice

- 3.9 (under MPI the bool and int64 state is float64 with a nan
  fill): every use is correct for 0.0 / 1.0. `initialize` sets all
  36 variables, so no nan reaches the kernel. `newsnow[jj] =
  net_snow[jj] > 0.0` (`:2093`) is the first statement of the loop,
  before the first read (`:2120`). The flags are only tested for
  truth or compared with 1 and 2; `lso = lso + 1` and `lso > 4` are
  exact in float64. What remains is 3.9 itself: the MPI kernel is
  compiled for a different signature than the serial tests run, and
  the MPI restart files hold these seven variables as float64.
- 7.3: see 11.3.

## Notes for later slices

- Inactive HRUs (JLM, 2026-11-17: one feature, not one finding per
  process). Snow needs: the loop at `:2090` covers every HRU, and
  the kernel tests `hru_type` only for `_LAKE` (`:2101`), so an
  inactive HRU builds a pack as land does; pywatershed loops over
  `wh_active_hrus` and masks the inactive ones at init
  (`_mask_inactive_hrus`). The description at `:1346` already lists
  `INACTIVE=0`.
- Slice 12 (atmosphere): `orad_hru` and `swrad` on a day with a zero
  solar table (11.2); `swrad` divides by `hru_cossl`
  (`prms_atmosphere.py:607-609`). `soltab_horad_potsw` is declared
  by both processes as a `(ndoy, space)` parameter, one shared
  buffer of 366 values per HRU.
- Slice 14 (translation): nothing validates `hru_deplcrv` against
  the number of curves in `snarea_curve` (11.5); the uniform-value
  collapse of the three density parameters is the only thing between
  a per-HRU `den_max` and `[0]` (11.7 item 3).
- Slice 15: no kernel units test for snow (known before the start).
  The kernel assigns bare `0.0` and `1.0` into dimensional outputs
  throughout (`:2094-2099`, `:2122`, `:2126`, `:2339-2357`), and
  `calc_snalbedo` assigns bare albedo values; the named-zero rule
  applies before a test can be written.
- Slice 17: `tests/test_restart_processes.py::test_snow` exists; no
  test supplies a partial pack (11.1) or shows `ai` is discarded
  (11.4).
- Slice 19: `coverage_serial.txt:92` (92%). Never reached: the
  `LAKE` skip (`:2102`: no test has a lake HRU in snow), both guards
  in `initialize` (`:1980`, `:1985`), the `calc_snowevap`
  corrections (`:1303-1306`, `:1310-1317`), rain that exactly meets
  the heat deficit (`:385-387`), `snowcov_area <= 0` in `calc_calin`
  (`:163`), the `den_max` fallbacks (`:202-203`, `:2325-2326`), a
  negative `pkwater_equiv` (`:277-278`, `:351`, `:2318`), `:100` and
  `:107` in `calc_sca_deplcrv` (not reachable: `frac_swe` is capped
  at 1.0 and the curve is not called when it equals 1.0).
  `tests/test_prms_snow.py` compares `iso`, `pkwater_equiv`,
  `snow_evap` at 1e-3 and `tcal`, `through_rain` by outlier
  fraction: pywatershed's list, whose own test
  (`autotest/test_prms_snow.py:45`) is marked `xfail`. Of the other
  31 variables, `tests/test_prms_snow_ab_numpy.py` compares 14 state
  variables over 120 days with pywatershed's numpy path
  (`docs/PORTS.md:143-144`; the test file itself was not read here).
  Why no test caught 11.1, 11.2, 11.5: one domain (drb), cold start
  with no pack, parameters from the pywatershed files.
- Fix phase, runs that confirm before fixing: a one-process snow
  Model with `pkwater_equiv` supplied alone and `control["budget"] =
  "error"` (11.1); one HRU with a zero row in `soltab_horad_potsw`
  and a pack (11.2); `pkwater_equiv` against `pk_ice + freeh2o` at
  every step end on drb, counting steps where they fall on opposite
  sides of 2.23e-16 (11.3, with the check listed under 7.3); a trace
  all-snow day on an empty HRU (11.6).
