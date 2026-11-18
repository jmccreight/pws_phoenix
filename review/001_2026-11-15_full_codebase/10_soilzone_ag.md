<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 10: soilzone Ag (PRMSSoilzoneAg, PRMSSoilzoneAgObsET)](#slice-10-soilzone-ag-prmssoilzoneag-prmssoilzoneagobset)
  - [Findings](#findings)
    - [10.1 `ag_area` is one buffer shared with PRMSRunoffAg, which overwrites it before the soil zone reads it as "the previous step's area": in a coupled model the ag storage is never redistributed when `ag_frac` changes](#101-ag_area-is-one-buffer-shared-with-prmsrunoffag-which-overwrites-it-before-the-soil-zone-reads-it-as-the-previous-steps-area-in-a-coupled-model-the-ag-storage-is-never-redistributed-when-ag_frac-changes)
    - [10.2 With `_pref_flow_flag` True, an HRU whose `pref_flow_max` is 0 loses `topfr`: it leaves `slow_stor` and reaches no store and no flux](#102-with-_pref_flow_flag-true-an-hru-whose-pref_flow_max-is-0-loses-topfr-it-leaves-slow_stor-and-reaches-no-store-and-no-flux)
    - [10.3 In the ObsET iteration `sroff` receives `dunnian_flow` once per pass of the kernel](#103-in-the-obset-iteration-sroff-receives-dunnian_flow-once-per-pass-of-the-kernel)
    - [10.4 The capped branch of `_adjust_irrigation` adds to `ag_irrigation_add` without asking for another pass: the loop can end with irrigation in the budget that the kernel never received](#104-the-capped-branch-of-_adjust_irrigation-adds-to-ag_irrigation_add-without-asking-for-another-pass-the-loop-can-end-with-irrigation-in-the-budget-that-the-kernel-never-received)
    - [10.5 Under MPI the iteration ends per rank, so `iter_count` (and, through 10.3 and 10.4, `sroff` and a capped HRU's state) depends on the decomposition](#105-under-mpi-the-iteration-ends-per-rank-so-iter_count-and-through-103-and-104-sroff-and-a-capped-hrus-state-depends-on-the-decomposition)
    - [10.6 `_post` counts infiltration on LAKE and INACTIVE HRUs as a budget input; the kernel leaves those HRUs before storing any water. The only guard is a bare `assert` at step 0](#106-_post-counts-infiltration-on-lake-and-inactive-hrus-as-a-budget-input-the-kernel-leaves-those-hrus-before-storing-any-water-the-only-guard-is-a-bare-assert-at-step-0)
    - [10.7 On a day when `aet_observed` is the missing value -1.0, the ag area gets no soil evapotranspiration and the HRU is reported as converged](#107-on-a-day-when-aet_observed-is-the-missing-value--10-the-ag-area-gets-no-soil-evapotranspiration-and-the-hru-is-reported-as-converged)
    - [10.8 Supplied state: an inconsistent `soil_lower` is booked as a redistribution and so hidden from the budget; the ag pair has no state group; `ag_soil_lower` is restart state that the kernel zeroes before any read](#108-supplied-state-an-inconsistent-soil_lower-is-booked-as-a-redistribution-and-so-hidden-from-the-budget-the-ag-pair-has-no-state-group-ag_soil_lower-is-restart-state-that-the-kernel-zeroes-before-any-read)
    - [10.9 Four statements in the docstrings do not match the code or pywatershed](#109-four-statements-in-the-docstrings-do-not-match-the-code-or-pywatershed)
    - [10.10 Cleanup: an argument list written out by hand, the area arithmetic written twice, values computed or passed and never read](#1010-cleanup-an-argument-list-written-out-by-hand-the-area-arithmetic-written-twice-values-computed-or-passed-and-never-read)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 10: soilzone Ag (PRMSSoilzoneAg, PRMSSoilzoneAgObsET)

- pws_phoenix commit: d855ada (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Path: `pws_phoenix/hydrology/prms_soilzone_ag.py`
- Checklist: B. Reviewed 2026-11-18.
- Method: `/code-review high` on the file (10 candidates; all 10 kept,
  10.5 narrowed after the read). Then a read of the whole file beside
  its pywatershed source
  (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:150-2263`), with
  the code outside the slice through CodeGraph and direct reads:
  `PRMSRunoffAg` (`pws_phoenix/hydrology/prms_runoff.py:2409-2615`,
  `:2894-3071`), `Model._add_process_fields`
  (`pws_phoenix/model.py:680-806`), `Model.step`, `ModelMPI.run`
  (`:1674-1730`), `Time.set_index`, the four helpers and the plain
  kernel of `pws_phoenix/hydrology/prms_soilzone.py` (`:127-320`,
  `:1321-1383`), `docs/MAINTENANCE.md`, `docs/PORTS.md:298-349` and
  `:401-453`, `tests/test_prms_runoff_soilzone_ag.py`,
  `tests/test_prms_soilzone_ag_obs_et.py`,
  `tests/test_prms_translate_ag.py`,
  `tests/test_prms_translate_ag_mpi.py`, `coverage_serial.txt:94`.
- Nothing was run. Every consequence below is from reading.

## Findings

### 10.1 `ag_area` is one buffer shared with PRMSRunoffAg, which overwrites it before the soil zone reads it as "the previous step's area": in a coupled model the ag storage is never redistributed when `ag_frac` changes

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:430-435`, `:1084`,
  `:1094-1117`, `:1133`; `pws_phoenix/hydrology/prms_runoff.py:2518-2526`,
  `:2928-2933`; `pws_phoenix/model.py:785-787`; `docs/PORTS.md:443-447`
- Code, the soil zone:

      ag_area = _meta(
          "variable",
          "Agricultural HRU area [acres] (per-step)",
          restart=True,
          units="acres",
      )
      ...
      old_ag_frac = ag_area[hh] / harea
      ...
      if not (
          abs(ag_frac[hh] - old_ag_frac)
          <= 1.0e-8 + 1.0e-5 * abs(old_ag_frac)
      ):

  runoff (`_post_areas`, after its kernel, same step):

      new_ag_area = ag_frac[ii] * hru_area[ii]
      ...
      ag_area[ii] = new_ag_area

  the Model:

      for name, meta in allocated.items():
          if name in grid_ds:
              continue

- Consequence: both classes declare `ag_area` as a variable, so the
  grid dataset holds one array. Runoff precedes the soil zone in
  `process_dict` (the soil zone reads this step's `infil`). At a step
  where `ag_frac` changes, runoff has already written `ag_area =
  ag_frac * hru_area`; the soil zone then computes `old_ag_frac ==
  ag_frac` and skips the block. Example: `ag_soil_moist = 3`, `ag_frac`
  0.2 -> 0.5. pywatershed scales `ag_soil_moist` to 1.2 (volume kept).
  The coupled port keeps the depth 3 over the larger area: 0.9 inch
  over the HRU appears. For 0.5 -> 0.2 pywatershed moves `3 * 0.3 =
  0.9` inch to `slow_stor`; the coupled port drops it. For `ag_frac`
  -> 0 the ag water stays in `ag_soil_moist` with no area.
  `ag_soil_moist_redistribution` is 0, so the budget shows nothing.
  The pervious rescale still runs: it reads `hru_area_perv`, a name
  only the soil zone declares.
  - pywatershed keeps `self.ag_area` per process
    (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:472`,
    `:909-911`, `:1010`).
  - `docs/PORTS.md:443-447` says both processes "read the PREVIOUS
    step's areas at step start"; for `ag_area` in the soil zone of a
    coupled model that is not so.
  - No test runs the coupled pair across a change date. The chain test
    tiles a static `ag_frac` (`tests/test_prms_runoff_soilzone_ag.py:199-205`);
    the restart chain is spinup (`docs/PORTS.md:451-453`);
    `test_fgr_one_liner_runs` runs 3 days; the MPI test and its serial
    reference run 200 days from Jan 1
    (`tests/test_prms_translate_ag_mpi.py:45-47`), and the first change
    is at step 366 (`docs/PORTS.md:324-325`). The one test that crosses
    it, `tests/test_prms_soilzone_ag_obs_et.py`, is a one-process
    Model, where `ag_area` is private.
  - Second effect of the same sharing, `hru_frac_perv`: runoff computes
    the pervious area as `((A - imperv) - ag) - dprst`, the soil zone
    as `((A - imperv) - dprst) - ag`. Runoff's kernel at step t+1 reads
    the value the soil zone wrote at step t; pywatershed's runoff reads
    its own. The two can differ in the last bit.
  - An instance of the open design topic "A buffer-ownership rule (one
    owner, the others view)" (`CLAUDE.md`), as 9.6.
- Source: /code-review, confirmed by the read
- Outcome: open

### 10.2 With `_pref_flow_flag` True, an HRU whose `pref_flow_max` is 0 loses `topfr`: it leaves `slow_stor` and reaches no store and no flux

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:1426-1430`,
  `:1449-1476`; compare `pws_phoenix/hydrology/prms_soilzone.py:1325-1330`,
  `:1359`, `:1382-1383`
- Code:

      if hru_type[hh] != _SWALE:
          if pref_flow_flag:
              topfr = max(_ZERO_INCHES, availh2o - pref_flow_thrsh[hh])
          ssresin = soil_to_ssr[hh] - topfr
          slow_stor[hh] = availh2o - topfr
      ...
      # dunnian from PFR (upstream's exact branch structure)
      dunnianflw_gvr = _ZERO_INCHES
      if pref_flow_flag:
          if pref_flow_max[hh] > 0.0:
              ...
      elif not (pref_flow_max[hh] > 0.0):
          if hru_type[hh] != _SWALE:
              dunnianflw_gvr = topfr

  the plain soil zone:

      if pref_flow_den[hh] > 0.0:
          ...
      elif hru_type[hh] == _LAND:
          dunnianflw_gvr = topfr

- Consequence: the flag is one scalar, True when any HRU has
  `pref_flow_den > 0`. On a LAND HRU with `pref_flow_den == 0` in such
  a domain, `pref_flow_thrsh = sat_threshold` and `pref_flow_max = 0`.
  On a day when `slow_stor + soil_to_ssr > sat_threshold`: `topfr > 0`
  is taken out of `slow_stor`; the `if pref_flow_flag:` block does
  nothing; the `elif` belongs to `if pref_flow_flag` and is not
  evaluated. Residual of the mass budget: `topfr`. With
  `control["budget"] = "error"` the run raises; otherwise the water is
  gone without a message. The plain soil zone sends this water to
  `dunnian_flow`.
  - With the flag False (fgr_ag_2yr), `topfr` stays 0: `slow_stor` is
    not limited by `sat_threshold` and there is never dunnian flow.
    The plain soil zone, on the same parameters, limits `slow_stor`
    and produces dunnian flow.
  - The lines are pywatershed's
    (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1904-1909`,
    `:1934-1957`). The port's comment says only "upstream's exact
    branch structure"; `docs/MAINTENANCE.md` has no entry.
  - Under MPI the flag is `any()` over the rank's rows (the comment at
    `:934-941`), so which HRUs lose `topfr` depends on the
    decomposition.
- Source: /code-review, confirmed by the read; checklist B (an
  upstream quirk kept without a comment on its consequence)
- Outcome: open

### 10.3 In the ObsET iteration `sroff` receives `dunnian_flow` once per pass of the kernel

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:139-147`, `:1582`,
  `:2138-2147`
- Code:

      _IT0_NAMES = (
          "soil_moist",
          "soil_rechr",
          "ssres_stor",
          "slow_stor",
          "ag_soil_moist",
          "ag_soil_rechr",
          "pref_flow_stor",
      )
      ...
      sroff[hh] = sroff[hh] + dunnian_flow[hh]

- Consequence: `sroff` is runoff's array, edited in place, and is not
  among the state restored before each pass. A day with `dunnian_flow
  = 0.1` in each of 4 passes leaves `sroff` 0.3 inch too high; `_post`
  writes `sroff_vol` from it and the channel receives it.
  `dunnian_flow` holds the last pass only, so the soil zone budget
  closes and runoff's budget was checked before the edit: nothing
  reports it. Dunnian flow needs `_pref_flow_flag` True (10.2), so
  fgr_ag_2yr does not reach it. Under MPI the number of passes is per
  rank (10.5), so `sroff` would also depend on the decomposition.
  pywatershed has the same lines
  (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1134-1142`,
  `:2113`). No comment, no ledger entry.
- Source: /code-review, confirmed by the read; checklist B (an
  upstream quirk kept without a comment)
- Outcome: open

### 10.4 The capped branch of `_adjust_irrigation` adds to `ag_irrigation_add` without asking for another pass: the loop can end with irrigation in the budget that the kernel never received

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:2060-2078`,
  `:2119`, `:2167-2168`
- Code:

      if soil_iter < max_soilzone_ag_iter:
          unsatisfied_max = unsatisfied_ag_et
          if unsatisfied_ag_et > ag_soil_moist_max[hh]:
              unsatisfied_max = ag_soil_moist_max[hh]
          else:
              ...
              # upstream marks needs-irrigation ONLY
              # in this (uncapped) branch
              add_estimated_irrigation = True
              num_hrus_ag_iter += 1
          ag_irrigation_add[hh] = (
              ag_irrigation_add[hh] + unsatisfied_max
          )
      ...
      ag_irrigation_hru_source[hh] = ag_irrigation_add[hh] * ag_frac[hh]

- Consequence: an HRU whose unsatisfied ag ET exceeds its
  `ag_soil_moist_max` (a small maximum, down to the guard's 1e-5, and
  a larger daily deficit) gets `ag_soil_moist_max` added to
  `ag_irrigation_add` and does not set the flag. If no other HRU sets
  it, `keep_iterating` becomes False and the last pass of the kernel
  ran without that water. `ag_irrigation_add`, `ag_irrigation_add_vol`
  and the budget input `ag_irrigation_hru_source` include it: residual
  `ag_soil_moist_max * ag_frac`, a raise under `budget = "error"`, and
  `iter_end_status = -9`. Whether the HRU's water is applied depends
  on some other HRU asking for a pass. pywatershed has the same lines
  (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:2191-2209`). The
  port's comment records the marking and not this consequence.
  `coverage_serial.txt:94` lists `:2063` as run; by which test is for
  slice 20.
- Source: /code-review, confirmed by the read; checklist B
- Outcome: open

### 10.5 Under MPI the iteration ends per rank, so `iter_count` (and, through 10.3 and 10.4, `sroff` and a capped HRU's state) depends on the decomposition

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:2116`,
  `:2131-2175`; `tests/test_prms_translate_ag_mpi.py:9-17`, `:52-58`,
  `:118`
- Code:

      n_transp = self._prep_aet(*self._prep_aet_args)
      ...
      if not add_estimated_irrigation:
          keep_iterating = False
      ...
      iter_count[hh] = soil_iter

- Consequence: `add_estimated_irrigation` and `n_transp` are computed
  from the rank's own HRUs. pywatershed writes one count for the whole
  domain (`self.iter_count[:] = soil_iter`,
  `pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1314`). On a day
  when the HRU that needs the most passes (say 6) is on one rank and
  the others stop at 2, `iter_count` is 6 there and 2 elsewhere;
  serial gives 6 everywhere. The argument in the MPI test's docstring
  (a converged HRU recomputes identically on an extra pass) holds for
  the seven restored storages. It does not hold for `iter_count`, for
  `sroff` (10.3), or for an HRU in the capped branch (10.4), whose
  water is applied only if an HRU of its own rank asks for a pass.
  The test compares `iter_count` bit for bit, but only the values of
  the last step (day 200, `model._ds_mpi_stream[nn].values` after the
  run). The module comment records the same limit for
  `_pref_flow_flag` and not for the loop.
- Source: /code-review, narrowed by the read
- Outcome: open

### 10.6 `_post` counts infiltration on LAKE and INACTIVE HRUs as a budget input; the kernel leaves those HRUs before storing any water. The only guard is a bare `assert` at step 0

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:1305-1306`,
  `:1314-1317`, `:1723-1724`, `:1847-1850`
- Code:

      if hru_type[hh] == _INACTIVE:
          continue
      ...
      if hru_type[hh] == _LAKE:
          unused_potet[hh] = potet[hh] - hruactet
          hru_actet[hh] = hruactet
          continue
      ...
      perv_infil_hru[hh] = infil[hh] * hru_frac_perv[hh]
      ag_infil_hru[hh] = infil_ag[hh] * ag_frac[hh]
      ...
      assert (refs["ag_frac"][wh_no_ag] == 0.0).all()

- Consequence:
  - `hru_frac_perv` is not zero on a LAKE HRU, and in a coupled model
    runoff writes it on INACTIVE HRUs too
    (`pws_phoenix/hydrology/prms_runoff.py:2933`). Runoff sends the
    water of an HRU that is not LAND to `infil` (8.7). A wet day on
    such an HRU: budget input > 0, no storage change, no output. With
    `budget = "error"` the run raises; otherwise the water is gone.
    The plain soil zone runs its reservoirs on a LAKE HRU (9.3). The
    lines are pywatershed's
    (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1750-1763`,
    `:1357-1358`). Latent: every HRU of the test domains is LAND. Part
    of what the inactive-HRU feature needs (note of slice 7).
  - The `assert` runs at step 0 only, though `ag_frac` varies in time;
    it has no message; `python -O` removes it; under MPI it raises on
    one rank alone, inside the run loop (the class of 3.7).
    pywatershed's is also an `assert` at step 0 (`:561-565`).
- Source: /code-review, confirmed by the read
- Outcome: open

### 10.7 On a day when `aet_observed` is the missing value -1.0, the ag area gets no soil evapotranspiration and the HRU is reported as converged

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:2017-2023`,
  `:1485-1494`, `:1525-1526`, `:2108-2109`
- Code:

      AET_external[hh] = aet_observed[hh]
      if (
          (AET_external[hh] < 0.0)
          and (AET_external[hh] != _AET_MISSING_INCHES)
          and (ag_frac[hh] > 0.0)
      ):
          AET_external[hh] = _ZERO_INCHES
      ...
      ag_AETtarget = aet_external[hh]
      ...
      ag_avail_targetAET = ag_AETtarget - hru_intcpevap[hh]
      if ag_avail_targetAET < 0.0:
          ag_avail_targetAET = _ZERO_INCHES

- Consequence: -1.0 is kept as the target; `ag_avail_targetAET`
  becomes 0; `compute_szactet` is not called; `agactet = 0`,
  `unused_ag_et = 0`, and `_iter_diag` gives `iter_end_status = 0`
  (converged). A gap in the observation file turns ag ET off for that
  day (it does not fall back to `potet`), and the status hides it.
  pywatershed has the same lines
  (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1109-1121`,
  `:1968-1981`). The field description says "-1.0 = missing" and
  nothing about what follows.
- Source: /code-review, confirmed by the read; checklist B (an
  upstream quirk kept without a comment)
- Outcome: open

### 10.8 Supplied state: an inconsistent `soil_lower` is booked as a redistribution and so hidden from the budget; the ag pair has no state group; `ag_soil_lower` is restart state that the kernel zeroes before any read

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:601-606`, `:707`,
  `:1053`, `:1162`, `:1172-1174`, `:1296`, `:1678-1682`;
  `pws_phoenix/hydrology/prms_runoff.py:1432`
- Code:

      state_groups = (("soil_moist", "soil_rechr", "soil_lower"),)
      ...
      soil_lower[hh] = soil_moist[hh] - soil_rechr[hh]            # _pre
      ...
      soil_lower_redistribution[hh] = (
          soil_lower[hh] - soil_lower_prev[hh]
      )
      ...
      ag_soil_lower[:] = _ZERO_INCHES                             # kernel

- Consequence:
  - Supply `soil_moist = 2`, `soil_rechr = 1`, `soil_lower = 0` (the
    group checks presence only). `advance()` sets `soil_lower_prev =
    0`; PRMSRunoffAg computes `soil_moist_prev = soil_lower_prev +
    soil_rechr_prev = 1` and infiltrates against a soil 1 inch too
    dry. `_pre` then sets `soil_lower = 1` and
    `soil_lower_redistribution = 1`, which `_post` subtracts from
    `soil_lower_change`. In 9.2 the budget showed the inconsistency;
    here it shows nothing.
  - `ag_soil_moist` and `ag_soil_rechr` are both restart variables and
    are in no group. Supplying `ag_soil_moist = 0.5` alone leaves the
    default `ag_soil_rechr`, clamped against the default
    `ag_soil_moist`, possibly above it.
  - `ag_soil_lower` is `restart=True` (`docs/PORTS.md:449` says "for
    symmetry"). The kernel zeroes it and rewrites it before anything
    reads it, and no other process declares it: a supplied or restart
    value is discarded. pywatershed's restart list has `soil_lower`
    and not `ag_soil_lower`
    (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:414-422`).
  - The comment above `state_groups` says `initialize()` clamps "from
    the DEFAULTS"; no clamp is applied to supplied state, and the
    class has no `check_initial_state()` (as 9.1, 9.2).
- Source: /code-review, confirmed by the read
- Outcome: open

### 10.9 Four statements in the docstrings do not match the code or pywatershed

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:41-44`, `:83-84`,
  `:1158-1160`, `:1835`
- Code:

      (``compute_soilmoist`` / ``compute_interflow`` / ``compute_gwflow``
      / ``compute_szactet`` -- upstream's ag kernel calls the same four
      from prms_soilzone.py)
      ...
      ``soil_zone_max`` (computed upstream, never read), the Budget
      machinery, restart, calc_method, verbose, ``adjust_parameters``
      ...
      redistribution = current - prev (nonzero only when ag_frac
      changed this step)."""
      ...
      """istep0 areas (once) + per-step area update + pre lines."""

- Consequence, text only except the first:
  - pywatershed's ag class has its own four static methods
    (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1371-1564`).
    Three compute the same as the port's shared helpers
    (`_compute_soilmoist` tests `soil2gw_max > 0.0` where the helper
    takes a flag; the port passes `soil2gw_max[hh] > 0.0`). The
    fourth differs: pywatershed's ag `_compute_interflow` (`:1414-1446`)
    has no `if c3 == 0.0: raise RuntimeError`, which the shared helper
    has (`pws_phoenix/hydrology/prms_soilzone.py:192-196`). A departure
    for one exact value of `ssres_in`, in no document.
  - "the Budget machinery, restart" as not ported: the file declares
    `budgets` (`:702`, `:1954`) and nine `restart=True` fields.
  - `_pre`: after the kernel's cap `soil_lower[hh] =
    soil_lower_max[hh]` (`:1605`), the next `_pre` recomputes
    `soil_moist - soil_rechr` and books the difference (below 1e-4 of
    `soil_lower_max`) as `soil_lower_redistribution` on a day with no
    change of `ag_frac`. Also true of supplied state (10.8).
  - `_pre_kernel` does not call `_istep0_areas`; the two `calculate`
    methods do.
- Source: /code-review, confirmed by the read
- Outcome: open

### 10.10 Cleanup: an argument list written out by hand, the area arithmetic written twice, values computed or passed and never read

- Where: `pws_phoenix/hydrology/prms_soilzone_ag.py:1736-1832`;
  `:1852-1865` and `:1087-1090`, `:1133-1136`; `:1254`; `:848-850`;
  `:2049`, `:2079`, `:2149-2152`; `:818`, `:914`, `:926`; `:965`
- Code:

      return (
          iter_aet_flag,
          self._pref_flow_flag,
          refs["soil_moist"],
          ...                                   # 85 lines
      ...
      pref_flow_den: np.ndarray,                # kernel argument
      ...
      refs["ag_soil_lower_stor_max"][:] = (
          refs["ag_soil_moist_max"] - ag_soil_rechr_max
      )
      ...
      pref_flow_den[np.where(hru_type != 1)] = 0.0  # non-LAND
      ...
      flags = (False, True) if "AET_external" in refs else (False,)

- Consequence, none on results today:
  - `_build_family_args` lists 85 `refs[...]` by position though
    `_hold(*names)` exists; two float arrays swapped there compile and
    run. The present order matches the signature (checked line by
    line).
  - `_istep0_areas` repeats the area formula of `_update_areas`, which
    runs immediately after it and recomputes the same three arrays.
  - The kernel argument `pref_flow_den` is never read (pywatershed
    passes it unused too). `ag_soil_lower_stor_max` is computed and
    read nowhere in the file, while the docstring drops
    `soil_zone_max` for that reason (the question of 8.9 and 9.5).
  - `_adjust_irrigation` returns `num_hrus_ag_iter` and
    `unsatisfied_big`; both are discarded.
  - `hru_type != 1` / `== 1` in three places, beside `_INACTIVE`,
    `_LAKE`, `_SWALE` imported from `prms_soilzone`, which also has
    `_LAND`.
  - The base class tests `"AET_external" in refs`, a field of its
    subclass; the docstring says "The core never knows iteration
    exists" (`:74-75`).
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations per element:
  - The kernel (`:1272-1623`) matches `_calculate_numpy`
    (`pywatershed/hydrology/prms_soilzone_ag_obs_et.py:1702-2177`)
    line for line, with the irrigation block moved to
    `_adjust_irrigation` and the storage-change lines to `_post`.
    `transp_on[ihru] and ...` is written `(transp_on[hh] != 0.0) and
    ...`. `ag_soilwater_deficit[:] = 0.0` moves from the kernel to
    `_adjust_irrigation`, which runs once per pass as the kernel
    does.
  - `_update_areas` matches `_update_ag_areas` (`:892-1026`) per
    element, `np.isclose` written out with the same second argument
    in each of its three uses. pywatershed's `soil_zone_max` lines
    are not ported (the docstring says so).
  - `_pre`, `_post`, `_prep_aet`, `_iter_diag` and the loop of
    `PRMSSoilzoneAgObsET.calculate` match `_calculate` (`:1028-1368`)
    in order, the three exit tests of the loop included.
  - `initialize()` matches `_initialize_soilzone_ag_data`
    (`:459-821`) in order, with the area block moved to step 0 of
    `calculate` (`ag_frac` is an input). `ag_soil_lower_stor_max` is
    computed before the clamps, as upstream.
  - The helpers: 10.9.
- A kernel that allocates per step, returns a value, or uses
  fastmath: no allocation (pywatershed copies seven arrays per step
  and allocates two per pass; the port holds seven buffers from
  `initialize()`). No fastmath. `_prep_aet` returns a count and
  `_adjust_irrigation` three scalars; the loop needs the first value
  of each (10.10 for the two discarded).
- A frozen parameter that is edited: none. `soil_rechr_max`,
  `ag_soil_rechr_max`, `_sat_threshold`, `_pref_flow_den` are derived
  fields. pywatershed's two edits of true parameters
  (`soil_moist_max`, `ag_soil_moist_max` below 1e-5) are
  `NotImplementedError` (`:854-863`).
- A float field without units: none. `ssr2gw_exp` is `fraction` (as
  in slice 9). `max_soilzone_ag_iter` is an int.
- A budget missing a term: none against pywatershed's list
  (`:425-450`), plus `swale_actet`. 10.2, 10.4 and 10.6 are about
  what the list does with cases the test domain does not have.
- An upstream quirk kept without a comment: 10.3, 10.7; with a
  comment that omits the consequence: 10.2, 10.4. Kept with a
  comment: `soil2gw_max` in the ag `compute_soilmoist` call,
  `ag_hortonian` never written, the 1e-4 cap of `soil_lower_ratio`.
- A departure not logged in `docs/MAINTENANCE.md`: the `c3 == 0.0`
  raise (10.9). The ledger has no entry for the Ag soil zone; its
  soil zone entry says "The Ag family shares the helper".
- A restart variable missing from the declarations: none.
  pywatershed's six are declared; the port adds `pref_flow_stor`,
  `hru_area_perv`, `ag_area` (needed, with a comment) and
  `ag_soil_lower` (10.8). `time.current_index == 0` is the index on
  the full time axis in both paths (`pws_phoenix/model.py:976`,
  `:1698-1700`), so `_istep0_areas` does not run again at a warm
  start.
- Findings of slice 9 in this file:
  - 9.1 applies as written at `:1452-1462` (a start state with
    `pref_flow_stor > pref_flow_max`, flag True).
  - 9.3 path 1 applies on a SWALE HRU with the flag True
    (`dunnianflw_pfr` is read only under `hru_type != _SWALE`,
    `:1571-1572`). Path 2 does not (LAKE and INACTIVE leave the loop
    early); 10.6 is what happens instead.
  - 9.4 does not apply: `infil` arrives as a depth over the pervious
    area, and `compute_soilmoist` is called only under `perv_area >
    0` / `ag_area > 0`.
  - 9.5 does not apply: the test is `> 0.0`, as upstream.
  - 9.6: `hru_frac_perv` is a variable in both Ag classes, with the
    same value up to the last bit (10.1). PRMSRunoff with
    PRMSSoilzoneAg leaves `infil_ag` unresolved, so that mixed pair
    raises at assembly.
- Already recorded elsewhere: `_pref_flow_flag` is `any()` over a
  rank's rows (the comment at `:934-941`, `docs/PORTS.md:403-405`);
  `_istep0_areas` overwrites supplied `hru_area_perv` / `ag_area` at
  a cold start (`CLAUDE.md`, "KNOWN GAPS"); the raises of
  `initialize()` (`:822`, `:855`, `:860`) depend on the data and
  happen on one rank alone (3.7).
- Inactive HRUs (not a finding, per the note of slice 7): the kernel
  skips them, as upstream. `initialize()`, `_update_areas`,
  `_istep0_areas` treat `hru_type == 0` in five places, following
  upstream. The feature will need one rule for `hru_frac_perv` on
  these HRUs shared with PRMSRunoffAg (10.1, 10.6).

## Notes for later slices

- Slice 14: `resolve_classes` pairs PRMSRunoffAg with the Ag soil
  zone classes (`tests/test_prms_translate_ag.py:114-137`), so every
  model built from an fgr control has the shared `ag_area` of 10.1.
  Whether the translation of the OpenET file can deliver -1.0 on an
  HRU with `ag_frac > 0` decides whether 10.7 is reachable from the
  fgr files.
- Slice 15: `coverage_serial.txt:94` lists the preferential-flow
  lines (`:1341-1366`, `:1452-1473`) as run. fgr_ag_2yr has
  `pref_flow_den` zero everywhere (the comment at `:937`), so the
  units test is the one that can reach them, and it compares the
  kernel with itself. Check its data for: an HRU with `pref_flow_max
  == 0` under the flag True (10.2); a swale with
  `pref_flow_infil_frac > 0` (9.3 path 1).
- Slice 17: the restart chain test is spinup with a static `ag_frac`
  (`docs/PORTS.md:451-453`); none restarts the coupled pair across a
  change date (10.1). No supplied-state test gives an inconsistent
  `soil_lower` or `ag_soil_moist` alone (10.8).
- Slice 20: no test compares the coupled Ag pair with the analysis
  answers on or after step 366 (10.1); the docstring of
  `tests/test_prms_translate_ag.py:17-22` gives the snow reason for
  having no full-chain comparison. `coverage_serial.txt:94`: the
  raises at `:822`, `:855`, `:860` are never reached; `:2173` and the
  branch `2060->2051` never (no test reaches
  `max_soilzone_ag_iter`); `1475->1479`, `1589->1596`, `1591->1596`
  never (a swale without the flag; a swale below `sat_threshold` or
  with no unsatisfied ET). Which test runs `:2063` (10.4).
  `tests/test_prms_translate_ag_mpi.py` compares the last step only
  (10.5).
- Fix phase, runs that confirm before fixing: (a) the coupled
  analysis model from `analysis.control`, serial, 370 days: compare
  `ag_soil_moist` and `slow_stor` at step 366 with the one-process
  ObsET Model fed from disk, and print
  `ag_soil_moist_redistribution` there (10.1); (b) a two-HRU Ag soil
  zone Model, `budget = "error"`, `pref_flow_den = (0.1, 0)`, a wet
  day (10.2); the same under ObsET with several passes, reading
  `sroff` (10.3); (c) one ag HRU with `ag_soil_moist_max = 0.01` and
  an observed AET of 0.3 (10.4); (d) `hru_type = 2` with `infil > 0`
  (10.6); (e) `aet_observed = -1.0` on an ag HRU (10.7).
