<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 7: groundwater, canopy, channel, hydraulic geometry](#slice-7-groundwater-canopy-channel-hydraulic-geometry)
  - [Findings](#findings)
    - [7.1 PRMSCanopy treats every HRU as LAND; pywatershed at ba3974b reads `hru_type`, and three documents say it does not](#71-prmscanopy-treats-every-hru-as-land-pywatershed-at-ba3974b-reads-hru_type-and-three-documents-say-it-does-not)
    - [7.2 Groundwater and canopy compute inactive HRUs; pywatershed masks them and skips them](#72-groundwater-and-canopy-compute-inactive-hrus-pywatershed-masks-them-and-skips-them)
    - [7.3 Canopy tests the snowpack with `pk_ice_prev + freeh2o_prev`; pywatershed's input is `pkwater_ante`](#73-canopy-tests-the-snowpack-with-pk_ice_prev--freeh2o_prev-pywatersheds-input-is-pkwater_ante)
    - [7.4 PRMSCanopy declares no state group, so `intcp_stor` can be supplied without `hru_intcpstor` and `intcp_transp_on`](#74-prmscanopy-declares-no-state-group-so-intcp_stor-can-be-supplied-without-hru_intcpstor-and-intcp_transp_on)
    - [7.5 PRMSChannel takes `segment_order` and `tosegment` on trust; an order that is not upstream-first drops flow without a message](#75-prmschannel-takes-segment_order-and-tosegment-on-trust-an-order-that-is-not-upstream-first-drops-flow-without-a-message)
    - [7.6 Nothing stops PRMSChannel on the distributed grid](#76-nothing-stops-prmschannel-on-the-distributed-grid)
    - [7.7 A `seg_slope` below 1e-7, or a nan `mann_n`, gives 24-hour routing without a message; pywatershed warns on the slope](#77-a-seg_slope-below-1e-7-or-a-nan-mann_n-gives-24-hour-routing-without-a-message-pywatershed-warns-on-the-slope)
    - [7.8 Intercepted snow evaporates at the rain rate unless snow falls the same day; the quirk is pywatershed's and has no comment](#78-intercepted-snow-evaporates-at-the-rain-rate-unless-snow-falls-the-same-day-the-quirk-is-pywatersheds-and-has-no-comment)
    - [7.9 A nan or negative `seg_outflow` becomes zero geometry; the quirk is pywatershed's and has no comment](#79-a-nan-or-negative-seg_outflow-becomes-zero-geometry-the-quirk-is-pywatersheds-and-has-no-comment)
    - [7.10 The "Deliberately NOT ported" lists say budgets and restart are not ported; the classes declare both](#710-the-deliberately-not-ported-lists-say-budgets-and-restart-are-not-ported-the-classes-declare-both)
    - [7.11 The two groundwater kernels are one kernel written twice](#711-the-two-groundwater-kernels-are-one-kernel-written-twice)
    - [7.12 Single-letter temporaries `z` and `d` in the canopy kernel](#712-single-letter-temporaries-z-and-d-in-the-canopy-kernel)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 7: groundwater, canopy, channel, hydraulic geometry

- pws_phoenix commit: c2de500 (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Paths: `pws_phoenix/hydrology/prms_groundwater.py`,
  `pws_phoenix/hydrology/prms_canopy.py`,
  `pws_phoenix/hydrology/prms_channel.py`,
  `pws_phoenix/hydrology/prms_hydraulic_geometry.py`
- Checklist: B. Reviewed 2026-11-17.
- Method: `/code-review high` on the four files (10 candidates; 8
  kept, 1 already recorded as 3.22, 1 dropped, see the checklist
  section). The command did not open pywatershed. Then a read of each
  file beside its pywatershed source
  (`pywatershed/hydrology/prms_groundwater.py`,
  `prms_groundwater_no_dprst.py`, `prms_canopy.py`, `prms_channel.py`,
  `prms_hydraulic_geometry.py`, `pywatershed/base/active_hru_mixin.py`),
  with the callers outside the slice (`Model._raise_if_partial_state_group`,
  `Model._run_initialize_hooks`, `Discretization.topological_order`)
  through CodeGraph, and the open questions that slices 2 and 5 left
  for this one.
- One thing was run, with JLM's permission (2026-11-17): a count of
  the `hru_type` values in every `parameters_dis_hru.nc` under the
  test data root (`PWS_TEST_DATA` unset, so
  `mpix_meta/pywatershed/test_data`):

  | domain | 0 inactive | 1 land | 2 lake |
  |---|---|---|---|
  | drb_2yr | 0 | 765 | 0 |
  | ucb_2yr | 0 | 3851 | 0 |
  | fgr_ag_2yr | 0 | 612 | 0 |
  | hru_1 | 0 | 1 | 0 |
  | sagehen_5yr | 0 | 128 | 0 |
  | sagehen_gridded_5yr | 2241 | 3368 | 0 |

  No file in pws_phoenix names either sagehen domain. Every other
  consequence below is from reading.

## Findings

### 7.1 PRMSCanopy treats every HRU as LAND; pywatershed at ba3974b reads `hru_type`, and three documents say it does not

- Where: `pws_phoenix/hydrology/prms_canopy.py:22-25`, `:76-78`,
  `:418`, `:450`; `docs/PORTS.md:73`, `:250`;
  `docs/MAINTENANCE.md:49`, `:65`
- Code, the port:

      # upstream: self._hru_type = LAND for every HRU (NHM); the dis
      # hru_type is deliberately ignored. numba folds the dead LAKE branch.
      _HRU_TYPE = _LAND
      ...
      if _HRU_TYPE == _LAKE or cov_type[ii] == _BARESOIL:
      ...
      if _HRU_TYPE != _LAKE and cov_type[ii] != _BARESOIL:

  pywatershed (`pywatershed/hydrology/prms_canopy.py:147`, `:372`,
  `:465`, `:498`):

      "hru_type",                          # in get_parameters()
      hru_type=self.hru_type,              # the supplied parameter
      if hru_type[i] == LAKE or cov_type[i] == BARESOIL:
      if hru_type[i] != LAKE and cov_type[i] != BARESOIL:

- Consequence: for an HRU with `hru_type == 2` (LAKE), a cover type
  above bare soil and a cover density above zero, pywatershed sets
  the store to zero every step and skips rain interception. The port
  intercepts rain there, holds it, and evaporates it, so `net_rain`,
  `hru_intcpevap` and `hru_intcpstor` differ. The port's runoff does
  read `hru_type` (`pws_phoenix/hydrology/prms_runoff.py:1003`), so
  within one pws_phoenix model the same HRU is land to the canopy and
  a lake to runoff. The statement "upstream sets `self._hru_type =
  LAND` for every HRU" is false at ba3974b; the only trace of
  `_hru_type` left upstream is a commented signature line (`:294`).
  `docs/MAINTENANCE.md:65` repeats it ("hardwired LAND here, as
  upstream"), and the code that entry quotes (`:49`,
  `if hru_type[ii] == _LAKE or ...`) is not the port's line. The
  ledger's "reachable only if" condition for the lake branch is
  therefore wrong in the other direction too: in the port the branch
  can never run. Latent today: no domain in the test data has a lake
  HRU (the table at the top), so the port and pywatershed agree on
  all of them.
- Source: checklist B (order of operations differs; a departure not
  logged)
- Outcome: open

### 7.2 Groundwater and canopy compute inactive HRUs; pywatershed masks them and skips them

- Where: `pws_phoenix/hydrology/prms_groundwater.py:215`, `:295`;
  `pws_phoenix/hydrology/prms_canopy.py:396`
- Code, the port:

      for ii in numba.prange(gwres_stor.shape[0]):
      ...
      for ii in numba.prange(nhru):

  pywatershed (`pywatershed/hydrology/prms_groundwater.py:98-99`,
  `:298-300`; `pywatershed/hydrology/prms_canopy.py:123-124`,
  `:442`):

      self._set_active_hrus()
      self._mask_inactive_hrus()
      ...
      # Inactive HRUs need no re-masking: ActiveHruMixin._mask_inactive_hrus
      # sets gwres_stor to NaN at init and every output above is arithmetic
      # on it, so NaN propagates.
      ...
      for aa in prange(nactive_hrus):
          i = wh_active_hrus[aa]

- Consequence: for an HRU with `hru_type == 0`, pywatershed's
  outputs are the fill value (nan for float64, -9999 for integers:
  `pywatershed/constants.py:55`, `:69-74`). The port computes numbers
  there from whatever parameters the HRU carries, and
  `gwres_flow_vol` for that HRU reaches the channel through the Map
  if its weight column is not zero. The port's groundwater does not
  declare `hru_type` at all (pywatershed lists it,
  `prms_groundwater.py:132`). Neither module docstring lists the
  active-HRU masking under "Deliberately NOT ported", and it is not
  in `docs/MAINTENANCE.md`. The port is not consistent with itself
  either: `pws_phoenix/hydrology/prms_soilzone_ag.py:1305` has
  `if hru_type[hh] == _INACTIVE:`. Upstream uses the same mixin in
  runoff, soilzone, snow, atmosphere and solar geometry (slices
  8-12). Latent on every domain pws_phoenix uses (all HRUs are
  land); it would differ on sagehen_gridded_5yr, where 2241 of 5609
  HRUs are inactive. JLM, 2026-11-17: support for inactive HRUs has
  not been formally introduced into pws_phoenix yet, and will be
  before long. So this is a feature not yet built, to be done for
  every hru process at once, and until then the docstrings and
  `docs/PORTS.md` should say that domains with inactive HRUs are not
  supported.
- Source: checklist B (the read)
- Outcome: open

### 7.3 Canopy tests the snowpack with `pk_ice_prev + freeh2o_prev`; pywatershed's input is `pkwater_ante`

- Where: `pws_phoenix/hydrology/prms_canopy.py:155-168`, `:464-466`
- Code, the port:

      if (
          pk_ice_prev[ii] + freeh2o_prev[ii]
      ) < _DNEARZERO and netsnow < _NEARZERO:

  pywatershed (`pywatershed/hydrology/prms_canopy.py:525-528`, and
  `prms_snow.py:530` `self.pkwater_ante[:] = self.pkwater_equiv`):

      if (
          pkwater_ante[i] < dnearzero
          and netsnow < nearzero
      ):

- Consequence: the branch decides whether grass (`cov_type == 1`)
  intercepts rain. The two tests agree only if the snow process
  keeps `pkwater_equiv` and `pk_ice + freeh2o` on the same side of
  2.23e-16. `docs/MAINTENANCE.md:8-41` records that snow tracks
  `pkwater_equiv` separately from `pk_ice + freeh2o` and that the
  two have differed (by up to 1.2e-7 inches on drb, before the
  `freeh2o` departure). Where one is exactly zero and the other is a
  roundoff remainder, the port and pywatershed take different
  branches and `net_rain` differs by up to the interception
  capacity. The input names also break the names-verbatim rule: the
  port has two inputs where pywatershed has one, and no file in
  `pws_phoenix/` contains the name `pkwater_ante`. The module
  docstring does not mention the difference. Not run: the equivalence
  on drb can be checked from pywatershed's answer files.
  JLM, 2026-11-17: noted; a deep dive on this one comes later
  (with the snow slice or after it).
- Source: checklist B (the read)
- Outcome: open

### 7.4 PRMSCanopy declares no state group, so `intcp_stor` can be supplied without `hru_intcpstor` and `intcp_transp_on`

- Where: `pws_phoenix/hydrology/prms_canopy.py:258-287` (three
  `restart=True` fields, no `state_groups`), `:333-349`, `:357`,
  `:535-537`
- Code:

      refs[name][:] = 0.0                    # initialize(): hru_intcpstor
      refs["intcp_transp_on"][:] = _OFF
      ...
      refs["hru_intcpstor_old"][:] = refs["hru_intcpstor"]   # advance()
      ...
      hru_intcpstor_change[ii] = (
          hru_intcpstor[ii] - hru_intcpstor_old[ii]
      )

- Consequence: `Model._raise_if_partial_state_group`
  (`pws_phoenix/model.py:601-618`) checks only declared groups; the
  three users are runoff and the two soil zones. With `intcp_stor`
  supplied alone, `hru_intcpstor` stays 0, the first `advance()`
  copies that to `hru_intcpstor_old`, and step 1 reports the whole
  supplied store as a storage gain: the mass budget is off by
  `intcp_stor * cov` (a raise under `control["budget"] = "error"`,
  a wrong `hru_intcpstor_change` otherwise). If `transp_on` is 1 on
  day 1, the default `intcp_transp_on = 0` also runs the
  winter-to-summer branch, `changeover = intcpstor * (covden_win -
  cov)`, on water the caller meant to start in summer cover.
  `hru_intcpstor = intcp_stor * cov` and `cov` depends on that day's
  `transp_on`, so `initialize()` cannot derive one from the other: a
  group is the fix the framework offers. Not run.
- Source: /code-review
- Outcome: open

### 7.5 PRMSChannel takes `segment_order` and `tosegment` on trust; an order that is not upstream-first drops flow without a message

- Where: `pws_phoenix/hydrology/prms_channel.py:275-284`, `:472`,
  `:553-585`; the claim at `:71-76`
- Code:

      refs["tosegment0"][:] = refs["tosegment"] - 1
      ...
      for ihr in range(24):
          seg_upstream_inflow[:] = _ZERO_CFS
          for jseg in segment_order:
              ...
              to_seg = tosegment0[jseg]
              if to_seg >= 0:
                  seg_upstream_inflow[to_seg] += outflow_ts[jseg]

- Consequence: `segment_order` is a declared parameter. It is
  computed from `tosegment` only when the caller passes
  `Discretization(topo_order={"segment_order": "tosegment"})`. A
  grid whose parameter file already carries `segment_order` (a file
  written from another network, a subset, a renumbering) is accepted
  with no check that it is a permutation, that it is upstream-first
  for this `tosegment`, or that `tosegment` is in range.
  - Out of range (`tosegment` larger than the grid): the njit kernel
    writes past the end of `seg_upstream_inflow`; numba does not
    check bounds.
  - Not upstream-first: a segment routed before its upstream
    neighbour reads `seg_upstream_inflow` before the neighbour has
    added to it, and the addition is zeroed at the top of the next
    hour. That upstream flow is lost every hour of every day. The
    channel declares no budget, so nothing reports it.
  The module docstring argues that no budget is needed because
  "`Discretization.topological_order` rejects an index that does not
  land". Finding 2.7 shows it does not reject one cleanly, and this
  path does not call it at all.
  Answer to the question 2.7 left for this slice: pywatershed checks
  nothing about `tosegment` either
  (`pywatershed/hydrology/prms_channel.py:239-276`), but it always
  computes the order itself in the same method, so its order and its
  `tosegment` cannot disagree. Not run.
  Who supplies `segment_order` today (text scan, 2026-11-17): every
  test, example and benchmark uses `topo_order=`. One path does not:
  `pws_phoenix/prms_translate/assemble.py:234` takes the saved
  `segment_order` from the preprocessed artifacts, after
  `verify_preprocessed` has compared its digest stamps with the
  current `tosegment`. So the exposure is a model built by hand from
  a parameter file that already carries the variable; no existing
  path reaches it.
- Source: /code-review, extended by the read
- Outcome: open

### 7.6 Nothing stops PRMSChannel on the distributed grid

- Where: `pws_phoenix/hydrology/prms_channel.py:555-585`; no check
  in `pws_phoenix/model.py` (finding 5.6's scan)
- Code:

      for jseg in segment_order:
          ...
          to_seg = tosegment0[jseg]
          if to_seg >= 0:
              seg_upstream_inflow[to_seg] += outflow_ts[jseg]

- Consequence: the same exposure as 5.6, for the array channel. The
  class docstring says the segment grid "stays serial/replicated
  under MPI"; no code enforces it. With `control["mpi_grid"]` set to
  the segment grid, each rank's block of `segment_order` and
  `tosegment0` holds indices into the whole network, and the njit
  kernel reads and writes outside the rank's arrays. One rank gives
  the right answer. Not run.
- Source: slice 5 note (a serial / MPI rule that no code enforces)
- Outcome: open

### 7.7 A `seg_slope` below 1e-7, or a nan `mann_n`, gives 24-hour routing without a message; pywatershed warns on the slope

- Where: `pws_phoenix/hydrology/prms_channel.py:158-173`
- Code, the port:

      velocity = (
          ((1.0 / mann_n) * np.sqrt(seg_slope) * seg_depth ** (2.0 / 3.0))
          ...
      # pywatershed clamps too-flat seg_slope IN PLACE here -- AFTER
      # velocity, so the edit affects nothing downstream in channel
      # init. We deliberately do NOT edit the read-only dis variable
      ...
      kcoef = _KCOEF_MAX_HOURS * np.ones(n_seg, dtype=np.float64)
      wh_moving = velocity > 0.0
      kcoef[wh_moving] = seg_length[wh_moving] / velocity[wh_moving]

  pywatershed (`pywatershed/hydrology/prms_channel.py:291-304`):

      mask_too_flat = self.seg_slope < 1e-7
      if mask_too_flat.any() and self._adjust_parameters != "no":
          msg = (
              "seg_slope < 1.0e-7, set to 1.0e-4 at indices:"
              ...
          warn(msg, UserWarning)
          if self._adjust_parameters == "error":
              raise ValueError(

- Consequence: the numbers match pywatershed (the clamp comes after
  the velocity in both), and not editing the parameter is the rule.
  What the port drops is the message. A zero slope gives velocity 0,
  a negative slope gives nan from `np.sqrt` (numpy's RuntimeWarning
  only), a nan `mann_n` gives nan; in each case `velocity > 0.0` is
  False, Kcoef stays 24 hours, and the segment is routed as a lake
  with finite coefficients. pywatershed tells the user which
  segments, and can raise. The comment says pywatershed edits "IN
  PLACE"; it rebinds the attribute to a new array (`:304`). The
  function is shared with the FlowGraph channel node type, where 5.2
  reached the same lines through a missing parameter. Not run.
- Source: /code-review; slice 5 note
- Outcome: open

### 7.8 Intercepted snow evaporates at the rain rate unless snow falls the same day; the quirk is pywatershed's and has no comment

- Where: `pws_phoenix/hydrology/prms_canopy.py:407-409`, `:493-506`
- Code:

      intcp_form[ii] = _RAIN
      if hru_snow[ii] > 0.0:
          intcp_form[ii] = _SNOW
      ...
      if intcpstor > 0.0:
          if hru_ppt[ii] < _NEARZERO:
              ...
              if intcp_form[ii] == _SNOW:
                  z = intcpstor - evsn

- Consequence: `intcp_form` is set from today's `hru_snow`, and the
  evaporation block runs only when `hru_ppt < 1e-6`. The sublimation
  branch therefore needs `0 < hru_snow < 1e-6` on the same day. Snow
  intercepted yesterday leaves today at `evrn = potet`, not
  `evsn = potet * potet_sublim`, and `potet_sublim` has almost no
  effect on the canopy (a calibration of it moves nothing here). The
  lines are pywatershed's
  (`pywatershed/hydrology/prms_canopy.py:455-457`, `:560-579`), so
  the port is right to keep them. The module docstring's note on
  `intcp_form` ("recomputed from scratch each step") describes the
  allocation, not this effect. The porting rule asks for a comment
  on a kept quirk.
- Source: /code-review
- Outcome: open

### 7.9 A nan or negative `seg_outflow` becomes zero geometry; the quirk is pywatershed's and has no comment

- Where: `pws_phoenix/hydrology/prms_hydraulic_geometry.py:192-198`;
  the docstring at `:13`
- Code:

      flow_cms = seg_outflow[ss] * _CFS_TO_CMS
      seg_flow_width[ss] = 0.0
      ...
      seg_res_time[ss] = 0.0
      if flow_cms > 0.0:

- Consequence: `flow_cms > 0.0` is False for nan, so a nan outflow
  from the channel is written as width 0, depth 0, area 0, velocity
  0 and residence time 0. PRMSStreamTemp then reads a dry segment
  where the channel produced nan, and the nan no longer shows in the
  geometry output. pywatershed does the same
  (`has_flow = flow_cms > 0.0`,
  `pywatershed/hydrology/prms_hydraulic_geometry.py:126-133`). The
  docstring says "All zero where flow is zero" and does not mention
  nan or negative flow.
- Source: /code-review
- Outcome: open

### 7.10 The "Deliberately NOT ported" lists say budgets and restart are not ported; the classes declare both

- Where: `pws_phoenix/hydrology/prms_groundwater.py:28-35`;
  `pws_phoenix/hydrology/prms_canopy.py:38-39`;
  `pws_phoenix/hydrology/prms_channel.py:83-85`, `:92-95`
- Code:

      Deliberately NOT ported:
      - Budget / ConservativeProcess machinery -- FLAGGED for a later design
        pass; ...
      - restart read/write; verbose; the calc_method switch ...

      Deliberately NOT ported: Budget/ConservativeProcess; adapters;
      restart; calc_method; verbose; ``imbalance_behavior``;

      - Budget / ConservativeProcess (see above); adapters; restart;
      ...
        init -- we do NOT edit the read-only dis variable. A deliberate
        design pass on init-time parameter editing is pending (CLAUDE.md).

- Consequence: groundwater declares `budgets = {"mass": ...}`
  (`:176`) and `restart=True` on `gwres_stor` (`:129`). Canopy says
  "Mass budget declared (Oct 2026)" seven lines above its list and
  has three `restart=True` fields. The channel has `restart=True` on
  `seg_inflow` and `outflow_ts`. What was not ported is pywatershed's
  budget and restart CLASSES; the lists read as if the features were
  absent. The channel's pointer to `CLAUDE.md` for the pending design
  pass meets `CLAUDE.md` pointing back ("see `prms_channel.py`").
  Text only.
- Source: /code-review
- Outcome: open

### 7.11 The two groundwater kernels are one kernel written twice

- Where: `pws_phoenix/hydrology/prms_groundwater.py:199-248`,
  `:278-333`
- Code, the only differing lines:

      stor = stor + (soil_to_gw[ii] * area + ssr_to_gw[ii] * area)

      stor = stor + (
          soil_to_gw[ii] * area
          + ssr_to_gw[ii] * area
          + dprst_seep_hru[ii] * area
      )

- Consequence: about 55 lines are repeated. A change to the
  reservoir arithmetic (the sink clamp, the conversion back to
  inches, a guard for 7.2) is made twice. A scalar njit helper that
  takes the summed inflow volume, as canopy does with `intercept`,
  would hold the arithmetic once and leave each class its loop.
  Cleanup only.
- Source: /code-review
- Outcome: open

### 7.12 Single-letter temporaries `z` and `d` in the canopy kernel

- Where: `pws_phoenix/hydrology/prms_canopy.py:500-501`, `:508-509`
- Code:

      z = intcpstor - evsn
      if z > 0:
      ...
      d = intcpstor - evrn
      if d > 0.0:

- Consequence: `CLAUDE.md`, python conventions 3: "Avoid
  single-letter variable names." The names and the integer `0` in
  `if z > 0:` are pywatershed's (`prms_canopy.py:573-574`,
  `:581-582`); the porting rules fix field names and the order of
  operations, not the names of temporaries. Cleanup only.
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations per element:
  - Groundwater: matches `_calculate_numpy` line for line. The
    NoDprst kernel omits pywatershed's `+ 0.0 * area` term, which
    changes at most the sign of a zero.
  - Canopy: matches, except 7.1 (`hru_type`), 7.2 (inactive HRUs)
    and 7.3 (the snowpack test). `intercept` is verbatim.
  - Channel kernel: matches `_muskingum_mann_numpy`. `ts /
    _ROUTING_STEP_HOURS` is `ts / 1.0`. The lateral inflow is summed
    after mapping where pywatershed sums per HRU first; this is the
    Map design, commented at `prms_channel.py:533-536` and noted in
    slice 5.
  - `muskingum_mann_coefficients`: matches
    `_initialize_channel_data`, except the dropped message (7.7).
  - Hydraulic geometry: matches the masked array expressions element
    by element (`flow_cms ** width_m`, the `> NEARZERO` test on the
    area, the residence time).
- A kernel that allocates per step, returns a value, or uses
  fastmath: none. pywatershed compiles all three with
  `fastmath=True`; the port does not, by rule. `intercept` returns
  two scalars (a helper with pywatershed's signature).
- A frozen parameter that is edited: none. `seg_slope` is not edited
  (7.7 is about the lost message). `depth_alpha` / `depth_m` in the
  WidthOnly class are `parameter_internal`, filled in `initialize()`
  before the freeze.
- A float field without units: none. `width_m` and `depth_m`
  (exponents) and `seg_slope` declare `fraction`, which the
  vocabulary defines as a proportion; pint sees `dimensionless`
  either way.
- A budget missing a term: none. Groundwater adds `gwres_sink` to
  pywatershed's list, with a comment (`prms_groundwater.py:172-175`)
  and entries in `docs/PORTS.md:218` and
  `docs/design_description.md:760`; it is not in the "Report
  upstream" entries of `docs/MAINTENANCE.md`. Canopy's list is
  pywatershed's. The channel has none by decision. Hydraulic
  geometry has none and no storage.
- An upstream quirk kept without a comment: 7.8, 7.9.
- A departure not logged in `docs/MAINTENANCE.md`: 7.1, 7.2, 7.3.
- A restart variable missing from the declarations: none. The
  restart lists equal pywatershed's (`gwres_stor`; `hru_intcpstor`,
  `intcp_stor`, `intcp_transp_on`; `seg_inflow`, `outflow_ts`).
  `inflow_ts_prev` and the `*_old` fields are rewritten by
  `advance()` before use. 7.4 is about supplying them, not declaring
  them.
- Initial state: groundwater starts at 2.0 inches when `gwres_stor`
  is not supplied, where pywatershed requires `gwstor_init`. This is
  the recorded initial-state design (a default in `initialize()`,
  PRMS's own default value), not a finding.
- Slice 5 carry-overs: 5.6 is 7.6. 5.2 (nan parameters into
  `muskingum_mann_coefficients`) is 7.7.
- Slice 2 carry-over (2.7): answered in 7.5.
- Already recorded, from /code-review: the kernels assume a daily
  step and nothing checks `dt` (3.22). In this slice that is
  `for ihr in range(24)` and `/ 24.0` in the channel
  (`prms_channel.py:552`, `:588-590`), and `gwflow_coef` /
  `gwsink_coef` (declared `1/day`) applied once per step
  (`prms_groundwater.py:221-223`). pywatershed's kernels assume the
  same.
- Dropped from /code-review: `stor / area` with `hru_area == 0`
  gives nan (`prms_groundwater.py:228`). pywatershed divides the
  same way (`prms_groundwater.py:290`), and no checklist item covers
  it.

## Notes for later slices

- Slices 8-12: pywatershed at ba3974b uses `ActiveHruMixin` in
  runoff, runoff ag, soilzone, snow, atmosphere and solar geometry
  (`_set_active_hrus`, `_mask_inactive_hrus`, loops over
  `wh_active_hrus`). Inactive HRUs are a feature not yet built
  (JLM, see 7.2): do not write the gap again as a finding per
  process. Record only what the fix will need: where a port already
  handles `hru_type == 0` in part (as `prms_soilzone_ag.py:1305`
  does), and any docstring statement about `hru_type` that describes
  an earlier pywatershed (7.1).
- Slice 11: the port's snow does not declare `pkwater_ante`
  (pywatershed `prms_snow.py:276`, `:530`). Decide 7.3 there:
  either snow publishes it, or the equivalence of the two tests is
  shown.
- Slice 14: `verify_preprocessed`
  (`pws_phoenix/prms_translate/preprocess.py:165-200`) is what
  stands between a saved `segment_order` and the channel kernel
  (7.5). It checks a digest only `if own is not None`, and the
  `derived_from_*` stamps only if present: check whether an artifact
  with no stamps passes.
- Slice 13: PRMSStreamTemp reads `segment_order` in several njit
  loops (`pws_phoenix/hydrology/prms_stream_temp.py:516`, `:623`,
  `:748`, `:870`); 7.5 applies. It also reads the zero geometry of
  7.9.
- Slice 15: hydraulic geometry has no kernel units test. Its kernel
  assigns bare `0.0` into five dimensional outputs
  (`prms_hydraulic_geometry.py:193-197`); the named-zero rule
  applies before a test can be written.
- Slice 17: `tests/test_restart_processes.py` covers canopy; no
  restart or supplied-state test gives canopy a partial state (7.4).
- Slice 19: no test has a LAKE or an inactive HRU in groundwater or
  canopy (7.1, 7.2), a `segment_order` that does not come from
  `topo_order=` (7.5), a `seg_slope` at or below zero (7.7), or a
  nan `seg_outflow` (7.9). `coverage_serial.txt:86` shows two canopy
  branches not taken (`426->450`, `461->476`) and `:90` one in
  hydraulic geometry (`202->204`, area at or below 1e-6).
- Fix phase, runs that confirm before fixing (the `hru_type` counts
  for 7.1 and 7.2 are done, see the top): `pkwater_ante <
  2.23e-16` against `pk_ice_prev + freeh2o_prev < 2.23e-16` over the
  drb answer files, on grass HRUs with rain (7.3); a canopy Model
  with `intcp_stor` supplied alone and `control["budget"] =
  "error"` (7.4); a drb channel with a reversed `segment_order`
  supplied in the parameter file (7.5).
