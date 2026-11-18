<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 8: runoff (PRMSRunoffNoDprst, PRMSRunoff, PRMSRunoffAg)](#slice-8-runoff-prmsrunoffnodprst-prmsrunoff-prmsrunoffag)
  - [Findings](#findings)
    - [8.1 A live PRMSCanopy -> PRMSRunoffAg model drops the canopy changeover water](#81-a-live-prmscanopy---prmsrunoffag-model-drops-the-canopy-changeover-water)
    - [8.2 `dprst_comp_ag` lacks the changeover inflow that pywatershed at ba3974b has; its docstring calls the old form verbatim](#82-dprst_comp_ag-lacks-the-changeover-inflow-that-pywatershed-at-ba3974b-has-its-docstring-calls-the-old-form-verbatim)
    - [8.3 PRMSRunoffAg's first-step block is applied on top of supplied `hru_perv`, `hru_frac_perv` and `ag_area`](#83-prmsrunoffags-first-step-block-is-applied-on-top-of-supplied-hru_perv-hru_frac_perv-and-ag_area)
    - [8.4 `imperv_stor` and `hru_impervstor` are both restart variables with no state group](#84-imperv_stor-and-hru_impervstor-are-both-restart-variables-with-no-state-group)
    - [8.5 PRMSRunoffAg's mass budget cannot close on a day `ag_frac` changes](#85-prmsrunoffags-mass-budget-cannot-close-on-a-day-ag_frac-changes)
    - [8.6 A zero pervious or agricultural fraction divides by zero inside the njit helpers](#86-a-zero-pervious-or-agricultural-fraction-divides-by-zero-inside-the-njit-helpers)
    - [8.7 On an HRU that is not LAND, the spill of the open depressions leaves storage and enters no flux](#87-on-an-hru-that-is-not-land-the-spill-of-the-open-depressions-leaves-storage-and-enters-no-flux)
    - [8.8 The module docstring and several comments contradict the code, or describe an earlier pywatershed](#88-the-module-docstring-and-several-comments-contradict-the-code-or-describe-an-earlier-pywatershed)
    - [8.9 PRMSRunoffAg declares a parameter, a derived parameter and an input that no computation uses](#89-prmsrunoffag-declares-a-parameter-a-derived-parameter-and-an-input-that-no-computation-uses)
    - [8.10 `PRMSRunoff.initialize` and the three kernels repeat code that the class structure could share](#810-prmsrunoffinitialize-and-the-three-kernels-repeat-code-that-the-class-structure-could-share)
    - [8.11 `initialize()` loops over HRUs in python](#811-initialize-loops-over-hrus-in-python)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 8: runoff (PRMSRunoffNoDprst, PRMSRunoff, PRMSRunoffAg)

- pws_phoenix commit: 9e9948e (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Path: `pws_phoenix/hydrology/prms_runoff.py`
- Checklist: B. Reviewed 2026-11-17.
- Method: `/code-review high` on the file (10 candidates; 9 kept, 1
  already recorded as 3.7, see the checklist section). The command
  did not open pywatershed. Then a read of the file beside its
  pywatershed sources (`pywatershed/hydrology/prms_runoff.py`,
  `prms_runoff_no_dprst.py`, `prms_runoff_ag.py`), with the code
  outside the slice through CodeGraph and text scans:
  `PRMSCanopy._calculate`, `prms_translate/control.py::resolve_classes`,
  `Model._run_initialize_hooks`, `Model._raise_if_partial_state_group`,
  `Model._apply_supplied_state`, the time index under a restart
  (`pws_phoenix/model.py:1342`, `:1692`), and the Ag tests.
- Nothing was run. Every consequence below is from reading.

## Findings

### 8.1 A live PRMSCanopy -> PRMSRunoffAg model drops the canopy changeover water

- Where: `pws_phoenix/hydrology/prms_runoff.py:2432-2435`;
  `pws_phoenix/hydrology/prms_canopy.py:527`, `:532`;
  `pws_phoenix/prms_translate/control.py:223-232`
- Code, runoff:

      # GSFLOW accounting: intcp_changeover IS in net_rain (upstream
      # derives this from the control's executable_desc; the fgr_ag_2yr
      # answers are GSFLOW). A PRMS-mode ag domain would flip this.
      _intcp_changeover_in_net_rain = True

  canopy:

      net_rain[ii] = netrain
      ...
      intcp_changeover[ii] = changeover + extra_water

  the translation layer:

      if ag:
          runoff: type = PRMSRunoffAg
      ...
          "prms_canopy": PRMSCanopy,

- Consequence: with the flag True, `compute_infil`, `compute_infil_ag`
  and `dprst_comp_ag` all skip `intcp_changeover`: they expect it
  inside `net_rain` / `through_rain`. The port's canopy never puts it
  there. So in a model where the canopy is computed (every model
  built by `model_from_control` from a `soilzone_ag` control), the
  changeover water of a season-change day leaves the canopy and
  enters nothing. Neither budget reports it: the canopy counts
  `intcp_changeover` as an output, and runoff sets
  `intcp_changeover_budget[:] = 0.0`. Size: the canopy store times
  the drop in cover density, per HRU, on the days `transp_on`
  changes (plus the store of a bare-soil HRU).
  Why no test catches it: the Ag regression tests feed `net_rain` and
  `intcp_changeover` from the GSFLOW answer files
  (`tests/test_prms_runoff_soilzone_ag.py:35`, `:46`), where the
  changeover is already inside `net_rain`. The only tests with a
  computed canopy and PRMSRunoffAg are
  `tests/test_prms_translate_ag.py:323` (3 days, checks finiteness)
  and the budget, which cannot see a loss between two processes.
  pywatershed: its canopy does not add the changeover to `net_rain`
  either (`pywatershed/hydrology/prms_canopy.py:606` is the only use
  after it is computed), but its flag is an argument and a control
  option that defaults to False
  (`pywatershed/hydrology/prms_runoff_ag.py:149`, `:187-193`); its
  autotests set it True when the answers are GSFLOW. The port's
  comment "upstream derives this from the control's executable_desc"
  describes the autotests, not the package. The flag is a property of
  the canopy and runoff together: either the canopy adds the
  changeover to `net_rain` when it is True, or the flag is False
  whenever the canopy is computed.
- Source: /code-review, confirmed by the read
- Outcome: open

### 8.2 `dprst_comp_ag` lacks the changeover inflow that pywatershed at ba3974b has; its docstring calls the old form verbatim

- Where: `pws_phoenix/hydrology/prms_runoff.py:735-784` (the
  signature, the docstring at `:777-778`, the line at `:784`)
- Code, the port:

      - inflow = through_rain + snowmelt UNCONDITIONALLY (no
        pptmix_nopack / no-snowpack rain logic, no net_rain/availh2o);
      ...
      inflow = through_rain + snowmelt

  pywatershed (`pywatershed/hydrology/prms_runoff_ag.py:1018-1019`,
  `:1052-1057`):

      intcp_changeover,
      intcp_changeover_in_net_rain,
      ...
      inflow = through_rain + snowmelt
      # Canopy changeover water reaches depression storage as in the
      # parent's dprst_comp (via availh2o = intcp_changeover + net_rain);
      # when changeover is in net_rain it is already part of through_rain.
      if not intcp_changeover_in_net_rain:
          inflow = inflow + intcp_changeover

- Consequence: none while the flag is True (8.1): the two give the
  same numbers. With the flag False, which is the fix 8.1 points to
  and what the class comment promises ("A PRMS-mode ag domain would
  flip this"), the port leaves the changeover out of the depression
  inflow. The pervious, agricultural and impervious parts receive it
  (`compute_infil`, `compute_infil_ag`), the budget counts it as an
  input (`intcp_changeover_budget`), so the mass budget is off by
  `intcp_changeover * dprst_frac` on every changeover day.
  `tests/test_kernel_units.py:965-974` runs the Ag kernel with the
  flag False, but compares the kernel only with itself.
- Source: checklist B (order of operations differs from pywatershed)
- Outcome: open

### 8.3 PRMSRunoffAg's first-step block is applied on top of supplied `hru_perv`, `hru_frac_perv` and `ag_area`

- Where: `pws_phoenix/hydrology/prms_runoff.py:2445-2462`,
  `:2518-2526` (the three `restart=True` fields, no `state_groups`
  entry), `:2950-2964`; `CLAUDE.md:518-521`
- Code:

      if time.current_index == 0:
          ...
          np.multiply(
              refs["ag_frac"],
              refs["hru_area"],
              out=refs["ag_area"],
          )
          refs["hru_perv"][:] -= refs["ag_area"]
          np.divide(
              refs["hru_perv"],
              refs["hru_area"],
              out=refs["hru_frac_perv"],
          )

- Consequence: the Model accepts every restart variable as supplied
  initial state. On a cold start, a supplied `hru_perv` that already
  has the agricultural area removed (the values of a restart file,
  for example) has it removed a second time; a supplied `ag_area` and
  `hru_frac_perv` are overwritten. Step 0's kernel then runs with a
  pervious area that is too small by `ag_area`. A restart is not
  affected: the time index stays global (`pws_phoenix/model.py:1342`,
  `:1692`), so the block does not run again. `CLAUDE.md` lists three
  first-step blocks that overwrite supplied state (PRMSAtmosphere,
  PRMSSoilzoneAg, StarfitDailyFlowNode); this is a fourth, and the
  only one that applies a change on top of the supplied value instead
  of replacing it. `docs/design_description.md:565-567` says the
  soil zone's block is closed by the init-hook split on branch
  feat_spotpy; whether that branch also moves this block was not
  checked (it is outside this commit).
- Source: /code-review
- Outcome: open

### 8.4 `imperv_stor` and `hru_impervstor` are both restart variables with no state group

- Where: `pws_phoenix/hydrology/prms_runoff.py:1250-1279`, `:1373`,
  `:1494`, `:1515`, `:1523-1525`; the dprst group at `:1873-1881`
- Code:

      refs["hru_impervstor_old"][:] = refs["hru_impervstor"]   # advance()
      ...
      if imperv_stor[ii] > 0.0:
          ...
          hru_impervstor[ii] = imperv_stor[ii] * imperv_frac
      ...
      hru_impervstor_change[ii] = (
          hru_impervstor[ii] - hru_impervstor_old[ii]
      )

- Consequence: the same exposure as 7.4. With `imperv_stor` supplied
  alone, `hru_impervstor` stays at `initialize()`'s 0, `advance()`
  copies that to `hru_impervstor_old`, and step 1 reports the whole
  supplied store as a storage gain: the mass budget is off by
  `imperv_stor * hru_percent_imperv` (a raise under
  `control["budget"] = "error"`, a wrong `hru_impervstor_change`
  otherwise). With `hru_impervstor` supplied alone, the value is
  kept for as long as `imperv_stor[ii] > 0.0` is False, because the
  kernel writes `hru_impervstor` only inside that test.
  `PRMSRunoff.state_groups` covers the five depression variables for
  this reason; the impervious pair is in no group, in any of the
  three classes. Here one is exactly the other times a parameter
  (`hru_impervstor = imperv_stor * hru_percent_imperv`), so a group
  or a derivation in the Model after supplied state would both work.
- Source: /code-review
- Outcome: open

### 8.5 PRMSRunoffAg's mass budget cannot close on a day `ag_frac` changes

- Where: `pws_phoenix/hydrology/prms_runoff.py:2855-2858`,
  `:2928-2936`; the comment at `:2921-2926`
- Code, the kernel (areas of the previous step):

      runoff = runoff + sroff_ag * ag_area[ii]
      ...
      hru_sroffp[ii] = srp * perv_frac
      hru_sroff_ag[ii] = sroff_ag * ag_frac[ii]

  `_post_areas` (areas of this step):

      hru_frac_perv[ii] = hru_perv[ii] / hru_area[ii]
      infil_perv_hru[ii] = infil[ii] * hru_frac_perv[ii]
      infil_ag_hru[ii] = infil_ag[ii] * ag_frac[ii]

- Consequence: `srp + infil` is the water on the pervious part, as a
  depth over that part. The budget multiplies `srp` by the old
  fraction and `infil` by the new one, so on an HRU whose `ag_frac`
  changed the residual is `srp * (old hru_frac_perv - new
  hru_frac_perv)`. Where `ag_area` was 0 and the new `ag_frac` is
  positive, the kernel skips `compute_infil_ag` (`:2764`) and the
  whole agricultural share of the input is missing as well. `sroff`
  is built from the old `ag_area` and `hru_sroff_ag` from the new
  `ag_frac`, so `sroff` is also not the sum of its four parts on that
  day. With `control["budget"] = "error"`, a run with dynamic
  `ag_frac` raises on the first change date that has rain or melt on
  a changed HRU. The lines are pywatershed's
  (`pywatershed/hydrology/prms_runoff_ag.py:826-829`, `:544-550`) and
  the port's comment names the ordering as a kept quirk; it does not
  say what the quirk does to the budget, and
  `docs/design_description.md:540-545` tells users of dynamic
  `ag_frac` that "the budget is the check". No test runs
  PRMSRunoffAg across a change date: the regression and restart
  tests use the static configuration
  (`tests/test_prms_runoff_ag.py:134`), and the dynamic one runs 3
  days (`tests/test_prms_translate_ag.py:327`).
- Source: /code-review, confirmed by the read
- Outcome: open

### 8.6 A zero pervious or agricultural fraction divides by zero inside the njit helpers

- Where: `pws_phoenix/hydrology/prms_runoff.py:404`, `:812`, `:844`;
  also `:426`, `:851`
- Code:

      srp = srp - dprst_srp / perv_frac
      ...
      sroff_ag = sroff_ag - dprst_sra / ag_frac
      ...
      open_vol_r = dprst_vol_open / dprst_vol_open_max

- Consequence: numba's default error model raises
  `ZeroDivisionError` on a float division by zero. Three inputs reach
  one:
  - `hru_frac_perv == 0` (an HRU where `hru_percent_imperv +
    dprst_frac`, plus `ag_frac` in the Ag class, equals 1) with
    depressions: `perv_comp` still returns `srp > 0` on a rain day,
    `dprst_srp` is 0, and `:404` is `0.0 / 0.0`.
  - Dynamic `ag_frac`: `ag_area[ii] > 0.0` from the previous step and
    the current `ag_frac[ii] == 0.0` (the HRU leaves agriculture),
    with `sroff_ag > 0`: `:844`.
  - A supplied `dprst_vol_open > 0` on an HRU whose
    `dprst_vol_open_max` is 0 (all depressions closed): `:426`.
    Runoff has no `check_initial_state()`.
  The run stops with an error that names no HRU and no parameter.
  `initialize()` checks `dprst_depth_avg == 0` only. Under MPI the
  raise is on the rank that owns the HRU (3.7). pywatershed has the
  same lines compiled the same way
  (`pywatershed/hydrology/prms_runoff.py:1330`, `:1355`;
  `prms_runoff_ag.py:1085`, `:1118`), so this is its behavior too;
  the port has no comment on it.
- Source: /code-review; checklist B (an upstream quirk kept without a
  comment)
- Outcome: open

### 8.7 On an HRU that is not LAND, the spill of the open depressions leaves storage and enters no flux

- Where: `pws_phoenix/hydrology/prms_runoff.py:2278-2285`,
  `:2848-2856`; the comments at `:1854`, `:2574-2575`
- Code:

      runoff = runoff + dprst_sroff_hru[ii] * hruarea

      # runoff for pervious, impervious, and depression areas
      srunoff = _ZERO_INCHES
      if hru_type[ii] == _LAND:
          runoff = runoff + srp * perv_area + sri * hruarea_imperv
          srunoff = runoff / hruarea

- Consequence: `dprst_comp` runs for every HRU with
  `dprst_area_max > 0`, whatever its type, and removes
  `dprst_sroff_hru` from `dprst_vol_open`. For `hru_type` 2 (lake), 3
  (swale) or 0, `srunoff` stays 0, so that water is in no output of
  the model. The budget closes, because `dprst_sroff_hru` is listed
  as an output; the comment `# sroff = hru_sroffi + hru_sroffp +
  dprst_sroff_hru` is false on those HRUs. The lines are pywatershed's
  (`pywatershed/hydrology/prms_runoff.py:916`, `:922-925`). The
  budget comment at `:1296-1302` records the swale case for the core
  class ("the list closes there too"); nothing records this one.
  Latent: every HRU of every domain pws_phoenix uses is LAND (the
  table in slice 7).
- Source: checklist B (an upstream quirk kept without a comment)
- Outcome: open

### 8.8 The module docstring and several comments contradict the code, or describe an earlier pywatershed

- Where: `pws_phoenix/hydrology/prms_runoff.py:4`, `:46-57`,
  `:105-109`, `:560-562`, `:1007`, `:2432-2434`
- Code:

      PRMSRunoffNoDprst + PRMSRunoff: PRMS surface runoff, ported from

      - Budget / ConservativeProcess (backlogged); adapters; restart;

      - Hardwired-off upstream paths, kept hardwired: cascades, frozen
        ground, glaciers, water use.
      - ``intcp_changeover_in_net_rain`` (GSFLOW/PRMS-6 accounting): fixed
        False (pywatershed default; module constant, compile-time).

      # staticmethods verbatim (prms_runoff_ag.py; scalar in/out; function
      # args and the intcp flag dropped per port convention)

      description="HRU type (HruType enum; LAND = 1, LAKE = 2)",

- Consequence, text only:
  - The title omits PRMSRunoffAg.
  - "Budget ... (backlogged); ... restart" as not ported: the file
    declares `budgets = {"mass": ...}` three times and `restart=True`
    on ten fields (the same wording as 7.10).
  - The flag "fixed False ... compile-time": `:105-109` calls it a
    runtime argument, and PRMSRunoffAg passes True.
  - "the intcp flag dropped": `compute_infil_ag` takes the flag and
    tests it at `:645`.
  - "cascades" as hardwired off upstream: at ba3974b the cascade path
    of the kernel is live (`ncascade_hru_active`,
    `pywatershed/hydrology/prms_runoff.py:777-782`, `:927-952`,
    `:1058-1077`), used by
    `pywatershed/hydrology/prms_runoff_cascades_no_dprst.py`. The
    port has no cascade code, which is the known backlog; the
    sentence gives the wrong reason.
  - `hru_type`: the enum also has INACTIVE = 0 and SWALE = 3
    (`pywatershed/constants.py:133-136`), and the kernel's `== _LAND`
    test treats all three others alike.
  - The comment at `:2432-2434` (see 8.1) and the docstring of
    `dprst_comp_ag` (see 8.2).
- Source: /code-review, extended by the read
- Outcome: open

### 8.9 PRMSRunoffAg declares a parameter, a derived parameter and an input that no computation uses

- Where: `pws_phoenix/hydrology/prms_runoff.py:2472-2489`,
  `:2500-2507`, `:566-580`, `:2426-2427`
- Code:

      ag_soil_rechr_max = DataArrayMeta(
          kind="parameter_internal",
          ...
          description="Ag recharge-zone maximum storage [inches] "
          "(in upstream kernel signatures but unused)",

      ag_soil_rechr_prev is in upstream's signature but unused."""

      Deliberately NOT ported: ``sat_threshold`` (declared upstream,
      never used by its kernel); ...

- Consequence: two porting rules meet here. "Not ported: ... unused
  declared parameters" drops `sat_threshold`; "scalar helper
  functions stay separate njit functions with verbatim signatures"
  keeps `ag_soil_rechr_max_frac`, `ag_soil_rechr_max` and the input
  `ag_soil_rechr_prev`. The class applies each rule to part of the
  same case. A user must supply a parameter and wire an input that
  change no output, and each costs an `(nhru,)` array. No rule says
  which of the two wins.
- Source: /code-review
- Outcome: open

### 8.10 `PRMSRunoff.initialize` and the three kernels repeat code that the class structure could share

- Where: `pws_phoenix/hydrology/prms_runoff.py:1327-1365` and
  `:1895-1991`; `:1373` and `:2077`; `:1493-1517`, `:2288-2312`,
  `:2861-2885`; `:1969`, `:1987`, `:2002`
- Code:

      refs["hru_impervstor_old"][:] = refs["hru_impervstor"]
      refs["dprst_stor_hru_old"][:] = refs["dprst_stor_hru"]

      dprst_clos_flag = False
      ...
              if dprst_area_clos_max[ii] > 0.0:
                  dprst_clos_flag = True
      ...
      del dprst_clos_flag

- Consequence: `PRMSRunoff.initialize` copies the 13-name zero list
  and the impervious half of the basin loop from the core class
  instead of calling `super().initialize()` (PRMSRunoffAg does call
  it); `advance` repeats the core's line instead of calling
  `super().advance()`. The 25-line impervious-evaporation block is
  written three times, with the same order of operations, and could
  be one njit helper beside `imperv_et`. `dprst_clos_flag` is set and
  deleted without being read. A fix to the impervious block (8.4, for
  example) has to be made three times. Cleanup only.
- Source: /code-review
- Outcome: open

### 8.11 `initialize()` loops over HRUs in python

- Where: `pws_phoenix/hydrology/prms_runoff.py:1357`, `:1970`,
  `:2004`; the docstring at `:1888-1889`
- Code:

      """basin_init + dprst_init verbatim (init-time loops; perf
      irrelevant). ...
      for ii in range(nhru):
          harea = hru_area[ii]

- Consequence: each iteration reads and writes a few dozen numpy
  scalars. That is nothing on 765 HRUs. At the 350,000-HRU goal the
  two loops of `PRMSRunoff.initialize` are of the order of seconds
  per model build (not measured). The loops have no dependence
  between HRUs, so one njit function would keep the order of
  operations per element. Not a correctness finding.
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations per element:
  - `perv_comp`, `check_capacity`, `imperv_et`, `ag_comp`,
    `check_capacity_ag`: match line for line (the hardwired
    `smidx_module = True` branch is folded).
  - `compute_infil`: matches with the cascade block removed
    (`avail_water` starts at 0, as upstream's `else`).
  - `dprst_comp`: matches. `if pptmix_nopack:` is written
    `!= 0`. The explicit zeros for `dprst_vol_open_frac` and
    `dprst_vol_frac` (`:524-529`) replace locals that upstream leaves
    undefined on one path; commented.
  - `compute_infil_ag`: matches, including the rain-on-snow test that
    ignores the flag (hand-off file, C5).
  - `dprst_comp_ag`: differs, 8.2. The rest matches, including the
    duplicated `min()` and the three differences from `dprst_comp`
    that the docstring lists.
  - The core kernel and the full kernel: match `_calculate_numpy`
    with `dprst_flag` False / True, and the post-kernel array lines.
  - The Ag kernel and `_post_areas`: match `_calculate_numpy_ag`,
    `_update_ag_areas` and the lines after it. The port loops with
    `numba.prange`; upstream's Ag kernel uses a plain `range` (no
    HRU depends on another, so the result is the same).
  - `initialize()`: matches `basin_init` and `dprst_init` with the
    flag assignments folded. Upstream's Ag class runs both at step 0
    inside `_calculate`; the port runs them in `initialize()` and
    only the carve-out at step 0 (8.3).
- A kernel that allocates per step, returns a value, or uses
  fastmath: none. Upstream allocates `soil_moist_prev` and
  `hru_horton_cascflow` per step; the port has scalars. The helpers
  return scalars (pywatershed's signatures). The first-step block of
  the Ag `calculate` writes with `out=` and `[:] -=`.
- A frozen parameter that is edited: none. The unported zeroing of
  `dprst_seep_rate_clos` and `va_clos_exp` is argued in a comment
  (`:1994-2001`); checked: every use of the two in the three kernels
  is under `dprst_area_clos_max > 0.0`, and the one in `initialize()`
  under `dprst_vol_clos[ii] > 0.0`, which is `dprst_frac_init *
  dprst_area_clos_max * dprst_depth_avg`. In PRMSRunoffAg the
  redeclared `hru_perv` / `hru_frac_perv` are variables, not frozen.
- A float field without units: none.
- A budget missing a term: none against pywatershed's three lists.
  8.5 and 8.7 are about what the lists do with upstream's ordering
  and with HRUs that are not LAND.
- An upstream quirk kept without a comment: 8.6, 8.7; the budget
  half of 8.5.
- A departure not logged in `docs/MAINTENANCE.md`: 8.1 (the flag
  hardwired True where upstream defaults to False) and 8.2. The
  ledger has no runoff entry.
- A restart variable missing from the declarations: none. Upstream's
  lists are `imperv_stor`, `hru_impervstor`, and for the full class
  `dprst_stor_hru`, `dprst_area_open`, `dprst_area_clos`,
  `dprst_vol_open`, `dprst_vol_clos`, `dprst_vol_thres_open`. The
  port has all but `dprst_vol_thres_open`, which it derives in
  `initialize()` from parameters. PRMSRunoffAg adds `hru_perv`,
  `hru_frac_perv` and `ag_area`, which upstream recomputes at step 0
  of every run; the port's reason (`:2441-2444`) holds for dynamic
  `ag_frac`.
- Already recorded, from /code-review: the raise in
  `PRMSRunoff.initialize` (`:2005-2009`) is on one rank alone under
  MPI. 3.7 lists these lines.
- Inactive HRUs (not a finding, per the note of slice 7): upstream's
  core and full kernels loop over the active HRUs
  (`for k in prange(nactive_hrus): i = hru_route_order[k] - 1`,
  `pywatershed/hydrology/prms_runoff.py:789-794`) after
  `_mask_inactive_hrus()`; upstream's Ag kernel loops over all HRUs
  (`prms_runoff_ag.py:667`). The port computes every HRU. An HRU with
  `hru_type == 0` is handled like a swale: no runoff, all water to
  `infil`, depression and impervious storage active. The feature
  will need the mask in all three kernels and in `_post_areas`, and
  a decision for the Ag kernel, where upstream itself does not skip.
- The marker `# units-test: unreachable` at `:470`, `:483`, `:895`,
  `:908`, `:965` has its reason in the comment lines above it, not in
  the `(<reason>)` form that `CLAUDE.md` gives. The harness matches
  the prefix only (`tests/kernel_units.py:102`), so the tests are not
  affected. The reason at `:963-964` ("initialize() rejects
  dprst_depth_avg == 0") does not cover a negative or nan
  `dprst_depth_avg`, which also reaches the line.

## Notes for later slices

- Slice 9, 10: the soil zones read `infil_hru` or `infil` /
  `infil_ag`, `sroff`, `hru_impervevap`, `dprst_evap_hru`; check what
  they do on a changeover day (8.1) and on an `ag_frac` change day
  (8.5: PRMSSoilzoneAg has its own area update, and the two classes
  must agree on which step's areas a depth is over).
- Slice 12 (or the fix of 8.1): the flag belongs to the canopy and
  runoff together. GSFLOW's canopy adds the changeover to the
  throughfall; the port's canopy has no such option.
- Slice 14: `resolve_classes` selects PRMSRunoffAg for every
  `soilzone_ag` control and has no way to pass the flag (8.1); it
  does not read `executable_desc`.
- Slice 15: the Ag kernel units test runs the flag False
  (`tests/test_kernel_units.py:965-974`) on a `dprst_comp_ag` that
  is not upstream's for that case (8.2). The runoff test data has a
  lake HRU (`:914`) but no check on where the depression spill goes
  (8.7).
- Slice 17: no supplied-state test gives runoff `imperv_stor` alone
  (8.4) or PRMSRunoffAg its areas on a cold start (8.3).
- Slice 20: `coverage_serial.txt:91` shows the closed-depression half
  of `PRMSRunoff.initialize` run by no serial test (`:1987`, `:2036`,
  `:2038`, `:2045`, `:2048-2059`), the raise at `:2007` never
  reached, and in the helpers the branches `210->224` (changeover
  with the flag False on a non-LAND HRU), `396->400`, `410->414`,
  `804->808`, `818->822`, `836->840` (runoff into depressions with no
  open area). No test has `hru_frac_perv == 0` (8.6), an HRU that is
  not LAND in a regression (8.7), or PRMSRunoffAg across an `ag_frac`
  change date (8.5). The Ag regressions take `net_rain` from GSFLOW
  files (8.1).
- Fix phase, runs that confirm before fixing: the full fgr_ag_2yr
  model from its control, comparing the sum of the canopy's
  `intcp_changeover` with what runoff receives (8.1); PRMSRunoffAg on
  the analysis configuration across Jan 1 with `control["budget"] =
  "error"` (8.5); a one-HRU runoff Model with `hru_percent_imperv +
  dprst_frac == 1` and rain (8.6); a runoff Model with `imperv_stor`
  supplied alone and `control["budget"] = "error"` (8.4).
