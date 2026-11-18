<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 9: soilzone (PRMSSoilzoneNoDprst, PRMSSoilzone)](#slice-9-soilzone-prmssoilzonenodprst-prmssoilzone)
  - [Findings](#findings)
    - [9.1 A start state with `pref_flow_stor > pref_flow_max` creates water as dunnian flow; `docs/MAINTENANCE.md` says the budget stays closed](#91-a-start-state-with-pref_flow_stor--pref_flow_max-creates-water-as-dunnian-flow-docsmaintenancemd-says-the-budget-stays-closed)
    - [9.2 `soil_lower` is a restart variable that must be supplied with `soil_moist` and `soil_rechr`, and nothing checks that the three agree](#92-soil_lower-is-a-restart-variable-that-must-be-supplied-with-soil_moist-and-soil_rechr-and-nothing-checks-that-the-three-agree)
    - [9.3 On an HRU that is not LAND, two paths take water out of the capillary reservoir and put it in no store and no flux](#93-on-an-hru-that-is-not-land-two-paths-take-water-out-of-the-capillary-reservoir-and-put-it-in-no-store-and-no-flux)
    - [9.4 `infil_hru / hru_frac_perv` divides by zero on an HRU with no pervious area](#94-infil_hru--hru_frac_perv-divides-by-zero-on-an-hru-with-no-pervious-area)
    - [9.5 The test on `pref_flow_infil_frac` is `!= 0.0` where pywatershed has `> zero`; `_pref_flow_flag` is computed and passed and never read](#95-the-test-on-pref_flow_infil_frac-is--00-where-pywatershed-has--zero-_pref_flow_flag-is-computed-and-passed-and-never-read)
    - [9.6 `hru_frac_perv` is written by runoff's `initialize()` and again by the soil zone's; the later one wins, and a mixed pair of variants gives no error](#96-hru_frac_perv-is-written-by-runoffs-initialize-and-again-by-the-soil-zones-the-later-one-wins-and-a-mixed-pair-of-variants-gives-no-error)
    - [9.7 The module docstring lists as "Deliberately NOT ported" things the file and its sibling have](#97-the-module-docstring-lists-as-deliberately-not-ported-things-the-file-and-its-sibling-have)
    - [9.8 The PRMSSoilzone kernel and its `calculate` repeat the NoDprst ones to change two expressions](#98-the-prmssoilzone-kernel-and-its-calculate-repeat-the-nodprst-ones-to-change-two-expressions)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 9: soilzone (PRMSSoilzoneNoDprst, PRMSSoilzone)

- pws_phoenix commit: 7d49929 (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Path: `pws_phoenix/hydrology/prms_soilzone.py`
- Checklist: B. Reviewed 2026-11-17.
- Method: `/code-review high` on the file (10 candidates; 9 kept as 8
  findings, two of them merged into 9.3; 1 dropped, see the checklist
  section). The command read pywatershed's `prms_soilzone.py`. Then a
  read of the file beside its pywatershed sources
  (`pywatershed/hydrology/prms_soilzone.py`,
  `prms_soilzone_no_dprst.py`), with the code outside the slice
  through CodeGraph and direct reads: `Model._run_initialize_hooks`,
  `Model._raise_if_partial_state_group`, `Model._apply_supplied_state`,
  the two `initialize` methods of `pws_phoenix/hydrology/prms_runoff.py`
  (`:1357-1365`, `:1970-1991`) and its kernel line `:1432`,
  `prms_translate/control.py::resolve_classes`,
  `prms_translate/readers.py::_ensure_f64_parse`, `units.VOCABULARY`,
  `docs/MAINTENANCE.md`, `docs/PORTS.md:220-224`,
  `tests/test_kernel_units.py:560-661`, `coverage_serial.txt:93`.
- Nothing was run. Every consequence below is from reading.

## Findings

### 9.1 A start state with `pref_flow_stor > pref_flow_max` creates water as dunnian flow; `docs/MAINTENANCE.md` says the budget stays closed

- Where: `pws_phoenix/hydrology/prms_soilzone.py:1359-1370` (the same
  lines at `:1816-1827`); `docs/MAINTENANCE.md:89-90`, `:92-97`
- Code:

      if pref_flow_den[hh] > 0.0:
          # PRMSIV eqn 1-135
          availh2o = pref_flow_stor[hh] + topfr
          dunnianflw_gvr = max(
              _ZERO_INCHES, availh2o - pref_flow_max[hh]
          )
          if dunnianflw_gvr > 0.0:
              # PRMSIV eqn 1-136
              topfr = max(_ZERO_INCHES, topfr - dunnianflw_gvr)

          pref_flow_in[hh] = pref_flow_infil[hh] + topfr
          pref_flow_stor[hh] = pref_flow_stor[hh] + topfr

  the ledger:

      message. For any real `c3` and `storage > 0`, `1 + c1*c2 > 0`, so the
      `else` is reached ONLY through the nan. (The `c3 == 0.0` raise needs
      `ssres_in` exactly `-coef_lin**2 / (4 coef_sq)`.) The mass budget
      stays closed (the storage keeps the water).

- Consequence: on a LAND HRU with `_pref_flow_den > 0`, a supplied or
  restart `pref_flow_stor = 2` with `pref_flow_max = 1` (state carried
  across a change of `sat_threshold` or `pref_flow_den`, for example),
  on a day with no preferential infiltration: `availh2o = 2`,
  `dunnianflw_gvr = 1`, `topfr = max(0, 0 - 1) = 0`, and
  `pref_flow_stor` stays 2 less the interflow. `dunnian_flow` (and
  `sroff`) gain 1 inch that left no store. The mass budget is off by
  `pref_flow_stor - pref_flow_max` on every such day until interflow
  drains the store below `pref_flow_max`. On a day with preferential
  infiltration the block at `:1285-1290` moves the excess to
  `dunnianflw_pfr` and resets the store, which conserves. The kernel
  never produces `pref_flow_stor > pref_flow_max` itself (both blocks
  leave it at or below the maximum) and neither does `initialize()`,
  so the one way in is supplied state, as the ledger says of the same
  start state. The ledger's sentence about the budget is true of the
  interflow lines it quotes and false of these lines. The class has no
  `check_initial_state()`. The lines are pywatershed's
  (`pywatershed/hydrology/prms_soilzone.py:1130-1144`).
- Source: /code-review, confirmed by the read
- Outcome: open

### 9.2 `soil_lower` is a restart variable that must be supplied with `soil_moist` and `soil_rechr`, and nothing checks that the three agree

- Where: `pws_phoenix/hydrology/prms_soilzone.py:775-782`, `:947-950`,
  `:1076-1086`, `:1130`, `:1156`, `:1412`, `:1456`;
  `pws_phoenix/hydrology/prms_runoff.py:1432`
- Code:

      # initialize() computes soil_lower = soil_moist - soil_rechr and
      # clamps soil_rechr <= soil_moist from the DEFAULTS (see
      # Process.state_groups)
      state_groups = (("soil_moist", "soil_rechr", "soil_lower"),)
      ...
      refs["soil_lower_prev"][:] = refs["soil_lower"]       # advance()
      ...
      soil_lower[hh] = soil_moist[hh] - soil_rechr[hh]       # kernel
      ...
      soil_lower_change[hh] = soil_lower[hh] - soil_lower_prev[hh]

  runoff:

      soil_moist_prev = soil_lower_prev[ii] + soil_rechr_prev[ii]

- Consequence: `soil_lower` is exactly `soil_moist - soil_rechr`. The
  state group makes the user supply it, and the Model checks only
  that all three are present (`Model._raise_if_partial_state_group`).
  With `soil_moist = 2`, `soil_rechr = 1`, `soil_lower = 0` supplied:
  `advance()` copies `soil_lower_prev = 0`; runoff computes
  `soil_moist_prev = 1` instead of 2, so step 0 infiltrates against a
  soil 1 inch too dry; the soil zone's kernel then sets `soil_lower =
  1` and reports `soil_lower_change = +1`, a budget residual of
  `1 * hru_frac_perv`. The clamps of `initialize()` (`soil_rechr <=
  soil_rechr_max`, `soil_moist <= soil_moist_max`, `soil_rechr <=
  soil_moist`) are applied to the defaults and not to supplied state,
  as the comment says. pywatershed does not treat `soil_lower` as
  state: its restart list is `soil_moist`, `soil_rechr`, `slow_stor`,
  `pref_flow_stor` (`pywatershed/hydrology/prms_soilzone.py:299-307`)
  and it derives `soil_lower` after the state is in place (`:587`).
  A derivation of `soil_lower` after supplied state and restart land
  would remove the third member of the group and the restart field.
  The same exposure as 7.4 and 8.4, with the difference that here a
  group exists and still admits the inconsistent case.
- Source: /code-review, confirmed by the read
- Outcome: open

### 9.3 On an HRU that is not LAND, two paths take water out of the capillary reservoir and put it in no store and no flux

- Where: `pws_phoenix/hydrology/prms_soilzone.py:1271-1291`,
  `:1323-1346`, `:1414-1441` (the same at `:1728-1748`, `:1780-1803`,
  `:1871-1898`); the budget comment at `:923-926`;
  `docs/PORTS.md:220-224`
- Code, path 1 (the preferential share of infiltration):

      if pref_flow_infil_frac[hh] != 0.0:
          ...
              capwater_maxin = capwater_maxin - pref_flow_maxin
              ...
              pref_flow_stor[hh] = pref_flow_stor[hh] + pref_flow_maxin
              dunnianflw_pfr = np.maximum(
                  _ZERO_INCHES, pref_flow_stor[hh] - pref_flow_max[hh]
              )
              if dunnianflw_pfr > 0.0:
                  pref_flow_stor[hh] = pref_flow_max[hh]
      ...
      if hru_type[hh] == _LAND:
          dunnianflw = dunnianflw_gvr + dunnianflw_pfr

  path 2 (the capillary excess):

      availh2o = slow_stor[hh] + soil_to_ssr[hh]

      if hru_type[hh] == _LAND:
          ...
          slow_stor[hh] = max(_ZERO_INCHES, availh2o - topfr)
          ...
      elif hru_type[hh] == _SWALE:
          slow_stor[hh] = availh2o

- Consequence:
  - Path 1, a SWALE, LAKE or INACTIVE HRU with `pref_flow_infil_frac >
    0`: `initialize()` sets `pref_flow_max` only where `hru_type ==
    _LAND`, so it is 0 here. The preferential share goes into
    `pref_flow_stor`, all of it becomes `dunnianflw_pfr`, the store
    is reset to 0, and `dunnianflw_pfr` is read only under `hru_type
    == _LAND`. Residual of the mass budget: `infil_hru *
    pref_flow_infil_frac` on every wet day. `initialize()` zeroes
    `_pref_flow_den` on these HRUs and leaves `pref_flow_infil_frac`
    as supplied; its error message tells the user to copy
    `pref_flow_den` (the raw parameter, not zeroed) into
    `pref_flow_infil_frac`.
  - Path 2, a LAKE or INACTIVE HRU: `soil_to_ssr` leaves `soil_moist`
    and is added to `slow_stor` in neither branch. It is reported in
    `ssres_in` and is in no store. Residual: `soil_to_ssr`, on every
    day the capillary reservoir is full and receives water. Runoff
    sends all water of an HRU that is not LAND to `infil` (8.7), so
    the reservoir does fill.
  - With `control["budget"] = "error"` both raise; otherwise the loss
    has no message. The budget comment and `docs/PORTS.md` present
    `swale_actet` as what closes the list on swales; with path 1 it
    does not.
  - The lines are pywatershed's
    (`pywatershed/hydrology/prms_soilzone.py:964-994`, `:1046-1110`,
    `:1192-1195`), and pywatershed runs them on LAKE HRUs too (it
    skips only INACTIVE ones). The port has no comment on either
    path, and `docs/MAINTENANCE.md` has no entry. The comment `# For
    swales` (`:1430`) heads a branch that every HRU that is not LAND
    takes. Latent: every HRU of every domain pws_phoenix uses is LAND
    (the table in slice 7).
- Source: /code-review, confirmed by the read; checklist B (an
  upstream quirk kept without a comment)
- Outcome: open

### 9.4 `infil_hru / hru_frac_perv` divides by zero on an HRU with no pervious area

- Where: `pws_phoenix/hydrology/prms_soilzone.py:1268`, `:160` (the
  same at `:1725`)
- Code:

      capwater_maxin = infil_hru[hh] / hru_frac_perv[hh]
      ...
      infil = infil - (excess / perv_frac)

- Consequence: an HRU where `hru_percent_imperv + dprst_frac == 1`
  (`hru_percent_imperv == 1` in the NoDprst class) has `hru_frac_perv
  == 0`. numba's default error model raises `ZeroDivisionError` on a
  float division by zero, `0.0 / 0.0` included, so the run stops at
  step 0 whether or not it rains, with an error that names no HRU and
  no parameter. `initialize()` has no check. Under MPI the raise is
  on the rank that owns the HRU (3.7). pywatershed has the line, with
  the comment "hru_frac_perv[hh] has to be > zero"
  (`pywatershed/hydrology/prms_soilzone.py:955-961`); the port has no
  comment. The same class as 8.6.
- Source: /code-review; checklist B (an upstream quirk kept without a
  comment)
- Outcome: open

### 9.5 The test on `pref_flow_infil_frac` is `!= 0.0` where pywatershed has `> zero`; `_pref_flow_flag` is computed and passed and never read

- Where: `pws_phoenix/hydrology/prms_soilzone.py:1271`, `:1728`;
  `:557-562`, `:1108-1110`, `:1235`, `:1541`, `:1693`; `:1411`,
  `:1868`
- Code:

      if pref_flow_infil_frac[hh] != 0.0:

  pywatershed (`pywatershed/hydrology/prms_soilzone.py:964`):

      if pref_flow_infil_frac[hh] > zero:

  and:

      refs["_pref_flow_flag"][:] = (hru_type == _LAND) & (
          pref_flow_den > 0.0
      )
      ...
      pref_flow_flag: np.ndarray,
      ...
      avail_potet = potet[hh] - hru_actet[hh]  # upstream (unused)

- Consequence: the two tests differ for a negative value and for nan.
  `initialize()` raises on a negative value. It lets nan through
  (`pfif.min() < 0.0` and `pfif.max() > 1.0` are both False when the
  array holds nan): the port then enters the block and nan spreads to
  `pref_flow_stor`, `soil_moist` and every flux of that HRU;
  pywatershed skips the block. No comment marks the difference.
  `_pref_flow_flag` costs an `(nhru,)` bool array and a kernel
  argument that change no output; pywatershed also passes it unused
  (`:696`, `:808`), and the port's rule "Not ported: ... unused
  declared parameters" would drop it (the same question as 8.9). The
  `avail_potet` line is a dead store in both kernels, marked as
  upstream's.
- Source: /code-review, extended by the read; checklist B (order of
  operations differs from pywatershed)
- Outcome: open

### 9.6 `hru_frac_perv` is written by runoff's `initialize()` and again by the soil zone's; the later one wins, and a mixed pair of variants gives no error

- Where: `pws_phoenix/hydrology/prms_soilzone.py:503-510`, `:956-970`,
  `:1592-1609`; `pws_phoenix/hydrology/prms_runoff.py:1365`, `:1991`;
  `pws_phoenix/model.py:542-553`
- Code, the soil zone (NoDprst):

      refs["hru_frac_perv"][:] = 1.0 - refs["hru_percent_imperv"]
      refs["hru_frac_perv"][wh_active] = (
          hru_area_perv[wh_active] / hru_area[wh_active]
      )

  runoff (full class):

      perv_area = perv_area - dprst_area_max[ii]
      ...
      hru_frac_perv[ii] = perv_area / harea

  the Model:

      """Call each process's initialize() (author order), then freeze
      derived parameters -- read-only once ALL hooks have run (a shared
      derived name is written by its owner before the freeze). ..."""

- Consequence: the field is one shared buffer on the HRU grid, and
  both processes are its writer. For the pairs that `resolve_classes`
  builds (both full or both NoDprst) the two write the same values on
  every HRU that is not INACTIVE, by the same operations. Two cases
  differ:
  - A `process_dict` built by hand with PRMSRunoff and
    PRMSSoilzoneNoDprst. Every input of the NoDprst soil zone
    resolves, so the build passes. The soil zone runs second and
    overwrites `hru_frac_perv` with the value that keeps the
    depression area in the pervious part. Runoff's kernel then reads
    a pervious fraction that disagrees with its own `hru_perv`, and
    its fluxes over the HRU (`infil_hru`, `hru_sroffp`) are scaled by
    the wrong fraction. No check compares the variants of the two
    classes.
  - INACTIVE HRUs: the soil zone leaves `1.0 - hru_percent_imperv`
    there (depression area not removed), runoff writes `perv_area /
    harea`. Which one the kernels read depends on the order of
    `process_dict`. Part of the inactive-HRU feature (see the
    checklist section).
  The eight lines of the hook are written twice (`:956-970`,
  `:1592-1609`). `CLAUDE.md` lists "A buffer-ownership rule (one
  owner, the others view)" as an open design topic; this is an
  instance.
- Source: /code-review, confirmed by the read
- Outcome: open

### 9.7 The module docstring lists as "Deliberately NOT ported" things the file and its sibling have

- Where: `pws_phoenix/hydrology/prms_soilzone.py:9-10`, `:58-61`,
  `:193-194`
- Code:

      Fourth REAL process port (July 2026) -- the process that produces
      ``ssres_flow_vol`` (replacing the submodel carrier) plus groundwater's
      ...
      Deliberately NOT ported (conventions in pws_phoenix/CLAUDE.md):

      - Budget / ConservativeProcess; adapters; restart; calc_method;
        verbose; ``imbalance_behavior``; the Ag/ObsET variants (iter_aet).

- Consequence, text only (the wording of 7.10 and 8.8):
  - "Budget ... restart": the file declares `budgets = {"mass":
    _MASS_BUDGET}` twice (`:945`, `:1586`), `restart=True` on five
    fields, and `state_groups`.
  - "the Ag/ObsET variants": they are in
    `pws_phoenix/hydrology/prms_soilzone_ag.py`.
  - "replacing the submodel carrier" is history of the port, not a
    description of the code.
  - The marker at `:194` (`# units-test: unreachable`) has no
    `(<reason>)`; the one at `:193` has. The harness matches the
    prefix only, so the tests are not affected (as in slice 8).
- Source: /code-review
- Outcome: open

### 9.8 The PRMSSoilzone kernel and its `calculate` repeat the NoDprst ones to change two expressions

- Where: `pws_phoenix/hydrology/prms_soilzone.py:1245-1468` and
  `:1699-1925`; `:1472-1543` and `:1929-2002`; the two differing
  expressions at `:1710-1715` and `:1909`
- Code:

      hru_actet[hh] = (
          hru_impervevap[hh]
          + hru_intcpevap[hh]
          + snow_evap[hh]
          + dprst_evap_hru[hh]
      )
      ...
      recharge[hh] = soil_to_gw[hh] + ssr_to_gw[hh] + dprst_seep_hru[hh]

- Consequence: about 380 lines are written twice. A fix to the kernel
  (9.1, 9.3, 9.4, 9.5) has to be made twice, identically. Each
  `calculate` passes about 70 arrays by position, and two float
  arrays swapped in one of the lists compile and run. One njit
  function per element that takes the evaporation seed as a scalar,
  called from two thin kernels, with `recharge` finished by the
  caller, would keep the order of operations per element and need no
  per-step zero arrays. Cleanup only (the same as 7.11 and 8.10).
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations per element:
  - `compute_soilmoist`, `compute_interflow`, `compute_gwflow`,
    `compute_szactet`: match line for line. `elif not transp_on:` is
    written `elif transp_on == 0.0:` and `et_type in [...]` as two
    `==` tests (the same result for every value, nan included).
  - The two kernels: match `_calculate_numpy` with `dprst_flag` False
    / True and the cascade blocks removed (`ncascade_hru is None`),
    except the test of 9.5. Upstream's array lines before and after
    its loop are folded per element in the same order: the
    `hru_actet` seed is `((imperv + intcp) + snow) + dprst`, as
    upstream's two statements; `recharge` is `(soil_to_gw +
    ssr_to_gw) + dprst_seep_hru`. Upstream's `cap_waterin[:] = zero`
    is not written; `cap_waterin[hh] = capwater_maxin` covers every
    element. Upstream's unused `interflow` local is dropped.
  - `initialize()`: matches `_initialize_soilzone_data` in order,
    including `soil_rechr` seeded from `soil_rechr_max` before its
    clamps. Upstream's `soil_zone_max` and `_swale_limit` are not
    computed; the docstring says so.
- A kernel that allocates per step, returns a value, or uses
  fastmath: none. Upstream compiles with `fastmath=True` and its
  NoDprst class allocates two zero arrays per step; the port has
  neither. The helpers return scalars (pywatershed's signatures).
- A frozen parameter that is edited: none. `_sat_threshold` and
  `_pref_flow_den` are derived copies. Upstream's one edit of a true
  parameter (`soil_moist_max < 1e-5` set to `1e-5`) is a
  `NotImplementedError` (`:1063-1067`), documented in the docstring.
  The `/code-review` candidate on this guard is dropped: it argued
  that a float32 `0.00001` read as float64 falls below `1.0e-5`, but
  `prms_translate/readers.py:244-247` parses PRMS reals as float64;
  and the guard scans every HRU, as upstream's mask does (`:428`).
- A float field without units: none. `ssr2gw_exp` is declared
  `units="fraction"`; it is an exponent, not a proportion (`none` and
  `fraction` are the same to pint, so no check is affected).
- A budget missing a term: none against pywatershed's list, plus
  `swale_actet`, which the port adds with a comment. 9.1 and 9.3 are
  about what the list does with a start state and with HRUs that are
  not LAND.
- An upstream quirk kept without a comment: 9.3, 9.4.
- A departure not logged in `docs/MAINTENANCE.md`: the `!= 0.0` of
  9.5; `soil_lower` as a restart variable (9.2), which is in no
  document as a departure. The ledger's soil zone entry is wrong in
  one sentence (9.1).
- A restart variable missing from the declarations: none. Upstream's
  four are declared; the port adds `soil_lower` (9.2). `ssres_stor`,
  `soil_moist_tot` and `soil_lower_ratio` hold `initialize()`'s
  values after supplied state or a restart lands, until the first
  `calculate` rewrites them; nothing reads them in between.
- Already recorded: the raises in `initialize()` (`:1041-1046`,
  `:1063-1067`) and in `compute_interflow` (`:192-196`) depend on the
  data and happen on one rank alone under MPI. 3.7 covers this class.
- Inactive HRUs (not a finding, per the note of slice 7): upstream
  loops over the active HRUs (`for ii in prange(nactive_hrus): hh =
  hru_route_order[ii] - 1`,
  `pywatershed/hydrology/prms_soilzone.py:934-935`) after
  `_mask_inactive_hrus()` (`:186`); its array lines before and after
  the loop cover every HRU. The port computes every HRU. An HRU with
  `hru_type == 0` runs the capillary reservoir and its
  evapotranspiration, then loses its capillary excess (9.3, path 2).
  `initialize()` already treats `hru_type == 0` in three places
  (`hru_frac_perv`, `_sat_threshold`, `ssres_stor`), following
  upstream. The feature will need the mask in both kernels and one
  rule for `hru_frac_perv` on these HRUs shared with runoff (9.6).
- The notes of slice 8 for this slice: on a canopy changeover day the
  soil zone has no code of its own; it reads `infil_hru` only, so 8.1
  does not reach it in the classes of this slice. The `ag_frac`
  question belongs to slice 10.

## Notes for later slices

- Slice 10: `prms_soilzone_ag.py` shares the four helpers and, if it
  repeats the kernel, 9.1, 9.3, 9.4 and 9.5 as well; check its
  `state_groups` against 9.2 and its `hru_frac_perv` (a variable in
  PRMSRunoffAg) against 9.6.
- Slice 14: `resolve_classes` pairs the variants consistently
  (`prms_translate/control.py:227-228`), so 9.6's mixed pair needs a
  `process_dict` built by hand. Whether the translation layer fills
  `pref_flow_infil_frac` from `pref_flow_den` on HRUs that are not
  LAND decides whether 9.3's path 1 is reachable from a PRMS
  parameter file.
- Slice 15: in the soil zone kernel units data
  (`tests/test_kernel_units.py:560-661`) one HRU has `pref_flow_den >
  0` (h3), the swale (h5) has `pref_flow_infil_frac = 0` (9.3, path
  1), no HRU is a lake or inactive (9.3, path 2), `hru_frac_perv` is
  0.8 everywhere (9.4), and `pref_flow_flag` is all False, which no
  result can show (9.5).
- Slice 17: no supplied-state test gives the soil zone an
  inconsistent `soil_lower` (9.2) or `pref_flow_stor >
  pref_flow_max` (9.1).
- Slice 20: `coverage_serial.txt:93` shows the raises at `:193`,
  `:1042` and `:1064` never reached, and these branches never taken:
  `274->285` (a `soil_type` that is not 1, 2 or 3), `1289->1291`,
  `1365->1369`, `1371->1385` (the preferential reservoir below its
  maximum, or empty), `1345->1348` (an HRU that is neither LAND nor
  SWALE), `1433->1441`, `1436->1441`, and the same six in the full
  kernel. This is what a regression domain with `pref_flow_den == 0`
  on every HRU gives (upstream's comment at
  `pywatershed/hydrology/prms_soilzone.py:304` says drb is such a
  domain): the preferential reservoir is run by the units test alone,
  which compares the kernel with itself, and by no comparison with
  pywatershed's answers.
- Fix phase, runs that confirm before fixing: a one-HRU soil zone
  Model with `control["budget"] = "error"` and (a) supplied
  `pref_flow_stor` above `pref_flow_max` and no infiltration (9.1);
  (b) supplied `soil_moist`, `soil_rechr` and a `soil_lower` that is
  not their difference (9.2); (c) `hru_type = 3` with
  `pref_flow_infil_frac > 0`, and `hru_type = 2` with a full
  capillary reservoir (9.3); (d) `hru_percent_imperv + dprst_frac ==
  1` (9.4).
