<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 19: the port tests, part A](#slice-19-the-port-tests-part-a)
  - [Findings](#findings)
    - [19.1 PRMSChannel is tested only at the end of a process chain; `docs/PORTS.md` gives its standard as 1e-13 "upstream's own", where pywatershed's is 1e-7 on the channel alone](#191-prmschannel-is-tested-only-at-the-end-of-a-process-chain-docsportsmd-gives-its-standard-as-1e-13-upstreams-own-where-pywatersheds-is-1e-7-on-the-channel-alone)
    - [19.2 `test_segment_order_is_dis_derived` does not check that the order is a permutation](#192-test_segment_order_is_dis_derived-does-not-check-that-the-order-is-a-permutation)
    - [19.3 Under MPI the channel side is compared at (1e-2, 1e-2) with outliers; the "replication-consistency check" compares each rank with the answers, not the ranks with each other](#193-under-mpi-the-channel-side-is-compared-at-1e-2-1e-2-with-outliers-the-replication-consistency-check-compares-each-rank-with-the-answers-not-the-ranks-with-each-other)
    - [19.4 Snow: 22 of 36 variables, 14 of them restart variables, are compared nowhere; the bit-for-bit comparison covers the first 120 of 731 days](#194-snow-22-of-36-variables-14-of-them-restart-variables-are-compared-nowhere-the-bit-for-bit-comparison-covers-the-first-120-of-731-days)
    - [19.5 Canopy: `intcp_transp_on` is compared nowhere; `pptmix` only at the last step, on a domain where its edit does not fire, and not under MPI](#195-canopy-intcp_transp_on-is-compared-nowhere-pptmix-only-at-the-last-step-on-a-domain-where-its-edit-does-not-fire-and-not-under-mpi)
    - [19.6 Atmosphere: the tindex state is compared nowhere; under MPI nine of ten variables at the last step only, and `ccov_hru` and the frost leaf do not run](#196-atmosphere-the-tindex-state-is-compared-nowhere-under-mpi-nine-of-ten-variables-at-the-last-step-only-and-ccov_hru-and-the-frost-leaf-do-not-run)
    - [19.7 Groundwater: every test supplies `gwres_stor`, so the cold-start default reaches no answer](#197-groundwater-every-test-supplies-gwres_stor-so-the-cold-start-default-reaches-no-answer)
    - [19.8 Hydraulic geometry: `PRMSHydraulicGeometryFull` is tested only with WidthOnly's default values](#198-hydraulic-geometry-prmshydraulicgeometryfull-is-tested-only-with-widthonlys-default-values)
    - [19.9 In the five MPI files a raise on rank 0 while it writes the input file leaves ranks 1-3 in `comm.Barrier()`](#199-in-the-five-mpi-files-a-raise-on-rank-0-while-it-writes-the-input-file-leaves-ranks-1-3-in-commbarrier)
    - [19.10 Guards and branches with no test](#1910-guards-and-branches-with-no-test)
    - [19.11 Comments and docstrings that do not match the code, conventions, side effects](#1911-comments-and-docstrings-that-do-not-match-the-code-conventions-side-effects)
  - [Code findings of slices 7, 11 and 12: why did no test catch this?](#code-findings-of-slices-7-11-and-12-why-did-no-test-catch-this)
  - [Questions handed to this slice](#questions-handed-to-this-slice)
  - [Checklist T, item by item](#checklist-t-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 19: the port tests, part A

- pws_phoenix commit: a863420 (branch `review_001`; read from
  `.git/HEAD` and `.git/refs/heads/review_001` as text). The commits
  since e33e217 are review commits, so the code lines of
  `coverage_serial.txt` still apply. The 14 files total 2,873 lines,
  as in the plan.
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text. Checklist T does
  not require it; its autotests and the start dates of two control
  files were read for comparison.
- Paths (under `tests/`): `test_prms_groundwater.py` (166),
  `test_prms_groundwater_mpi.py` (202), `test_prms_canopy.py` (169),
  `test_prms_canopy_mpi.py` (184), `test_prms_channel.py` (400),
  `test_prms_channel_mpi.py` (374), `test_prms_hydraulic_geometry.py`
  (143), `test_prms_atmosphere.py` (185),
  `test_prms_atmosphere_mpi.py` (184),
  `test_prms_atmosphere_transp_frost.py` (188),
  `test_prms_solar_geometry.py` (54), `test_prms_snow.py` (208),
  `test_prms_snow_ab_numpy.py` (216), `test_prms_snow_mpi.py` (200)
- Checklist: T. Covers code slices 7, 11, 12. Reviewed 2026-11-19.
- Method: `/code-review high` on the 14 files (10 candidates, all
  kept: the permutation as 19.2, `pk_def_prev[0]` in 19.4, the other
  eight in 19.11). Added by the read: 19.1, 19.3-19.10, and the
  `docs/PORTS.md` row in 19.1. Then every file of the slice, whole,
  against checklist T.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/process.py:157-211`, `:429-479`;
    `pws_phoenix/hydrology/prms_groundwater.py:170-320`;
    `pws_phoenix/hydrology/prms_canopy.py:260-330`, `:415-480`;
    `pws_phoenix/hydrology/prms_hydraulic_geometry.py:150-170`,
    `:225-246`; `pws_phoenix/atmosphere/prms_atmosphere.py:770-850`;
    the variable declarations of `prms_snow.py`, `prms_channel.py`
    (text scans); `docs/PORTS.md:60-130`, `:285`;
    `tests/test_prms_translate_ag.py:240-342`;
    `tests/test_prms_stream_temp_mpi.py:60-80`.
  - pywatershed: `autotest/test_prms_channel.py:15`, `:46-82`, and
    the tolerance lines of `test_prms_canopy.py`,
    `test_prms_groundwater.py`, `test_prms_atmosphere.py`,
    `test_prms_snow.py`; `test_data/{drb_2yr,ucb_2yr}/nhm.control:6-8`
    (both start 1979-01-01).
  - Before the review: `coverage_serial.txt` (header and the rows of
    the slice's code); the findings files of slices 7, 11, 12
    (headings, header, the questions and notes); 1.6, 16.11, 16.13,
    16.16, 17.5, 17.10 and the notes for slice 19 at the end of the
    files of slices 15-18.
- Run: the nine serial files in env pwpx2, with `--error-for-skips`:
  28 passed in 12.8 s. The five MPI files were not run. No test was
  changed to see whether it then fails, and no data file was read
  beyond the two control-file dates: a finding that depends on a data
  value says so and names the run that would settle it (the last
  section).

Each of the slice's processes is compared with pywatershed's answers
at every step, at or tighter than pywatershed's own tolerance, except
the channel (19.1). Snow has a bit-for-bit comparison with
pywatershed's numpy path, with the one logged departure excused by a
rule computed from the data. The findings are ordered by
consequence: first the channel, which no test isolates (19.1-19.3);
then the variables and days each process's tests leave out
(19.4-19.8); then a hang (19.9), the untested guards (19.10) and the
text (19.11).

## Findings

### 19.1 PRMSChannel is tested only at the end of a process chain; `docs/PORTS.md` gives its standard as 1e-13 "upstream's own", where pywatershed's is 1e-7 on the channel alone

- Where: `docs/PORTS.md:69`, `:128`; `tests/test_prms_channel.py:99-114`;
  `tests/test_prms_channel_mpi.py:77-82`; pywatershed
  `autotest/test_prms_channel.py:15`, `:73-82`
- Code:

      | PRMSChannel | `hydrology/prms_channel.py` | 1e-13; `seg_stor_change` (1e-7, 1e-4) cancellation carve-out | ...

      MODE_TOL = {
          "snow_disk": (1.0e-10, 0.0),
          "snow_live": (1.0e-2, 1.0e-3),  # observed outliers: 0.014%
      }

      default_rtol = default_atol = 1.0e-7        # pywatershed

- Consequence:
  - pywatershed tests its channel alone: the three lateral volumes
    come from its answer files, and the outputs are compared at
    1e-7. No pws_phoenix test runs PRMSChannel alone. The strictest
    comparison is the `snow_disk` mode, where the lateral volumes
    come from live runoff, soil zone and groundwater, at the chain's
    1e-10. So a channel error cannot be told from an error upstream
    of it, and 1e-13 is held by no test.
  - The `docs/PORTS.md` row and `:128` ("1e-13 gw/channel") give a
    standard that is neither pywatershed's (1e-7) nor the test's
    (1e-10). `CLAUDE.md` says the tolerance is "pywatershed's OWN
    autotest standard for that process"; the test is tighter than
    that, the document is wrong.
  - Answer to 16.16: no test holds `segment_order` to pywatershed's
    exact order. Another valid order changes the sums in
    `seg_upstream_inflow` at the level of rounding, far below 1e-10;
    the one test of the order accepts any valid order (19.2).
- Source: checklist T (a tolerance against `docs/PORTS.md`); the
  pywatershed autotest read
- Outcome: open

### 19.2 `test_segment_order_is_dis_derived` does not check that the order is a permutation

- Where: `tests/test_prms_channel.py:389-400`
- Code:

      """... read-only, and is a valid ordering."""
      position = np.empty_like(order)
      position[order] = np.arange(order.shape[0])
      for iseg in range(order.shape[0]):
          if to_seg[iseg] >= 0:
              assert position[iseg] < position[to_seg[iseg]]

- Consequence: if `order` repeats one segment and leaves out
  another, `position` of the left-out segment is whatever
  `np.empty_like` left there, and the assertion compares that. The
  test can pass on an order that is not valid. In this file the
  `snow_disk` comparison would still fail, since the left-out
  segment is never routed (7.5), so the gap is in this test, not in
  the suite. `np.array_equal(np.sort(order), np.arange(n))` first
  would close it.
- Source: /code-review, confirmed by reading
- Outcome: open

### 19.3 Under MPI the channel side is compared at (1e-2, 1e-2) with outliers; the "replication-consistency check" compares each rank with the answers, not the ranks with each other

- Where: `tests/test_prms_channel_mpi.py:23-25`, `:77-82`, `:333-368`
- Code:

      Every rank builds the segment side identically (same files, same
      topo_order) -- that IS the replication; the final-state test asserting
      on EVERY rank doubles as a replication-consistency check.

      bad = ~np.isclose(
          vals, answers[nn].values[-1, :], rtol=RTOL, atol=ATOL
      )
      assert bad.mean() <= 0.01, f"'{nn}' final state differs"

- Consequence:
  - The MPI path differs from serial on the channel side in two
    places: `MapMPI` (each rank's partial product, then
    `Allreduce(SUM)`) and the replicated serial grid. The one test
    that runs them uses the full chain's fastmath ceiling: 0.1% of
    segment-days may fall outside rtol = atol = 1e-2 (333 of
    456 x 731, `docs/PORTS.md:285`). An error confined to one
    segment on fewer than 333 days, or one inside that tolerance
    everywhere, passes.
  - At the last step each rank may differ from the answers on 1% of
    the segments (4 of 456) by more than 1e-2. Two ranks that
    disagree with each other on those segments both pass, so the
    test does not check replication. An `allgather` of the final
    state and `assert_array_equal` across ranks would.
  - No test compares this MPI run with the serial `snow_live` run of
    `tests/test_prms_channel.py`, which differ only by the order of
    the `Allreduce` sums.
- Source: checklist T (a serial test with no MPI counterpart where
  the MPI code differs)
- Outcome: open

### 19.4 Snow: 22 of 36 variables, 14 of them restart variables, are compared nowhere; the bit-for-bit comparison covers the first 120 of 731 days

- Where: `tests/test_prms_snow.py:56-66`;
  `tests/test_prms_snow_ab_numpy.py:9-10`, `:51-69`, `:197-204`;
  `tests/test_prms_snow_mpi.py:52-53`
- Code:

      ANSWER_NAMES = ("iso", "pkwater_equiv", "snow_evap")
      FRACTION_ANSWER_NAMES = ("tcal", "through_rain")

      for the first N_STEPS days (a full snow season)
      N_STEPS = 120

      pk_def_prev = np.zeros_like(our_states["pk_def"])
      pk_def_prev[1:] = our_states["pk_def"][:-1]

- Consequence:
  - PRMSSnow declares 36 variables. The A/B test compares 14
    (`CHECK`, which holds all five of `tests/test_prms_snow.py`). The
    other 22 are compared by no test of the slice: the restart
    variables `ai`, `iasw`, `int_alb`, `lso`, `lst`, `mso`, `pksv`,
    `pss`, `pst`, `salb`, `scrv`, `slst`, `snowcov_areasv`, `snsv`;
    and `frac_swe`, `newsnow`, `pk_precip`, `pptmix_nopack`, the
    copies `freeh2o_prev`, `pk_ice_prev`, and the budget terms
    `freeh2o_change`, `pk_ice_change` (these two only as part of a
    residual). `pptmix_nopack` is read by PRMSRunoff; it reaches an
    answer only through the `snow_live` chain at 1e-2.
  - The bit-for-bit window is 1979-01-01 to 1979-04-30 from a cold
    start with no pack. After that, two autumns of pack onset and a
    pack carried across 1980-01-01 are compared for five variables:
    three at 1e-3, two by outlier fraction. The docstring calls the
    window "a full snow season".
  - Under MPI: `pkwater_equiv` at every step; `iso` and `snow_evap`
    at the last step (1980-12-31) only.
  - `pk_def_prev[0]` is zero, not the state before the first step.
    On a cold start `freeh2o_prev[0]` is also zero, so the trigger
    cannot fire on day 1 and the rule is right for this run; with a
    supplied pack holding free water and a heat deficit, rain on day
    1 would take the departure on an HRU the rule does not excuse,
    and the test would fail on an HRU it means to excuse.
- Source: checklist T (only some variables, time steps); the last
  item from /code-review
- Outcome: open

### 19.5 Canopy: `intcp_transp_on` is compared nowhere; `pptmix` only at the last step, on a domain where its edit does not fire, and not under MPI

- Where: `tests/test_prms_canopy.py:11-12`, `:43-55`, `:144`;
  `tests/test_prms_canopy_mpi.py:44-53`;
  `tests/test_prms_atmosphere.py:14-16`
- Code:

      intcp_transp_on has no generated answer file (excluded); pptmix is
      compared as an output (input + recomputed edits).

      for nn in ("intcp_stor", "hru_intcpstor", "net_rain", "pptmix"):

      compared as-is (the file may carry canopy's in-place edits, which do
      not fire on drb -- see test_prms_canopy.py).

- Consequence:
  - `intcp_transp_on` is a restart variable (int64; float64 under
    MPI, 3.9) that chooses which changeover branch runs. Only its
    effect on `intcp_changeover` and `intcp_stor` is compared.
  - `pptmix` is not in `ANSWER_NAMES`; it is compared at the last
    step only, though the docstring says "compared as an output". By
    the atmosphere test's docstring the edit does not fire on drb,
    so the `pptmix` that comes out equals the file that went in and
    the comparison cannot fail on the edit. Under MPI, where a
    mutable input lives in a per-step buffer refilled from
    `step.mpi.src`, `pptmix` is compared at no step.
  - The MPI final state leaves out `intcp_form`,
    `hru_intcpstor_change`, `hru_intcpstor_old`.
- Source: checklist T (only some variables, time steps; a serial
  test with no MPI counterpart where the MPI code differs)
- Outcome: open

### 19.6 Atmosphere: the tindex state is compared nowhere; under MPI nine of ten variables at the last step only, and `ccov_hru` and the frost leaf do not run

- Where: `tests/test_prms_atmosphere.py:14`, `:147-173`;
  `tests/test_prms_atmosphere_mpi.py:38-49`, `:149-154`;
  `pws_phoenix/atmosphere/prms_atmosphere.py:800-842`
- Code:

      STREAMED_NAME = "swrad"  # the one to_netcdf var (mpixarray limit)
      FINAL_STATE_NAMES = (
          "tmaxf", "tavgc", "prmx", "hru_ppt", "hru_rain", "hru_snow",
          "orad_hru", "potet", "transp_on",
      )

- Consequence:
  - `tmax_sum` and `transp_check` (restart variables) have no answer
    file. `transp_on` is compared at every step serially, on drb for
    the tindex leaf and on ucb_2yr for the frost leaf.
  - Under MPI, `swrad` is compared at every step and nine variables
    on 1980-12-31 only; `transp_check` (int, float64 under MPI) is
    seen only through `transp_on` on that day. Whether any HRU has
    `transp_on == 1` on Dec 31 was not read.
  - `ccov_hru` is pinned serially only. PRMSAtmosphereTranspFrost
    has no MPI test (`tests/test_prms_atmosphere_mpi.py` runs
    PRMSAtmosphere).
  - `test_ccov_hru_pin` restates the kernel's formula in numpy on the
    model's own `swrad`: an independent second form, the right check
    when no answer file exists.
- Source: checklist T (only some variables, time steps)
- Outcome: open

### 19.7 Groundwater: every test supplies `gwres_stor`, so the cold-start default reaches no answer

- Where: `pws_phoenix/hydrology/prms_groundwater.py:194`;
  `tests/test_prms_groundwater.py:84`, `:153-166`;
  `tests/test_prms_groundwater_mpi.py:135-139`;
  `tests/test_prms_channel.py:273`; `tests/test_prms_channel_mpi.py:244-246`
- Code:

      self._refs["gwres_stor"][:] = 2.0

      "gwres_stor": parameters["gwstor_init"],

- Consequence:
  - The line runs in every test (it counts as covered) and is
    overwritten by the supplied state before the first step. A
    wrong default passes every test of the slice; 17.10 lists the
    same for the restart tests.
  - `test_answers_close_with_sink` sums pywatershed's own answer
    files; no change to the port can fail it. Its docstring says so;
    it documents why `gwres_sink` is an output term.
  - PRMSGroundwaterNoDprst is not run in this slice; it is in
    `tests/test_prms_no_dprst*.py` (slice 20).
- Source: checklist T (a test that cannot fail); coverage
- Outcome: open

### 19.8 Hydraulic geometry: `PRMSHydraulicGeometryFull` is tested only with WidthOnly's default values

- Where: `tests/test_prms_hydraulic_geometry.py:119-143`;
  `pws_phoenix/hydrology/prms_hydraulic_geometry.py:164-168`, `:245-246`
- Code:

      parameters["depth_alpha"] = (("nsegment",), np.full(nseg, 0.27, ...))
      parameters["depth_m"] = (("nsegment",), np.full(nseg, 0.39, ...))
      ...
      np.testing.assert_array_equal(full_run[nn].values, width_only_run[nn].values, ...)

      def _set_depth_params(self) -> None:
          pass  # supplied parameters; nothing to fill

- Consequence: the one test of Full cannot tell supplied values from
  the defaults. Anything that ends with 0.27 / 0.39 in the arrays
  passes: for example Full's `_set_depth_params` override removed,
  if a supplied parameter is still writable when `initialize()` runs
  (not checked). A second run with other values, compared with the
  power law by hand on a few segments, would show that the supplied
  values are used. The WidthOnly comparison (five variables, every
  step, 1e-5) is tighter than pywatershed's own test, which checks
  `seg_res_time` only. No test feeds a nan or negative `seg_outflow`
  (7.9) or reaches an area at or below 1e-6 (`coverage_serial.txt`:
  `202->204`). Under MPI, hydraulic geometry runs only in
  `tests/test_prms_stream_temp_mpi.py` (`seg_flow_width`,
  `seg_res_time`; slice 21).
- Source: checklist T (a test that cannot fail)
- Outcome: open

### 19.9 In the five MPI files a raise on rank 0 while it writes the input file leaves ranks 1-3 in `comm.Barrier()`

- Where: `tests/test_prms_groundwater_mpi.py:86-112`,
  `tests/test_prms_canopy_mpi.py:90-105`,
  `tests/test_prms_channel_mpi.py:126-183`,
  `tests/test_prms_atmosphere_mpi.py:84-115`,
  `tests/test_prms_snow_mpi.py:93-117`
- Code:

      if comm.rank == 0:
          ...
          combined = xr.merge([...], compat="no_conflicts")
          ...
          combined.to_netcdf(input_file)
      comm.Barrier()

- Consequence: the same pattern as 17.5. If rank 0 raises inside the
  block (a merge conflict, a missing variable, a full disk), ranks
  1-3 wait in the `Barrier` and `mpirun -n 4 pytest` hangs with no
  report (15.13). The test methods themselves hold no collective
  calls; the rank-0-only methods return early on the other ranks,
  so a failure there reports normally. In the `mpi_run` fixtures the
  Model's own raises are collective (the ranks agree first); none of
  these files expects a raise (16.2).
- Source: checklist T, with the note of slice 15 ("what one rank
  failing alone would do")
- Outcome: open

### 19.10 Guards and branches with no test

- Where: the rows of `coverage_serial.txt` for the slice's code, and
  the notes of slices 7, 11 and 12
- Code:

      prms_atmosphere.py   92%   449, 703, 779, 806, 809-818, 825-827
      prms_solar_geometry.py 85% 141, 157->160, 162->165, 169-170, 178-182, 186-190, 204-205
      prms_canopy.py       99%   426->450, 461->476
      prms_hydraulic_geometry.py 98%  202->204
      prms_snow.py         92%   100, 107, 163, 202-203, ... 1980, 1985, ... 2318, 2325-2326

- Consequence:
  - Guards that no test shows raising: the `hru_lat` hemisphere
    guard (`prms_atmosphere.py:449`, the one raise in the package's
    atmosphere; per rank under MPI, 3.7) and the two guards of
    `PRMSSnow.initialize` (`:1980`, `:1985`). The negative-radiation
    warning of `compute_soltab` (`:204-205`) is not shown to warn.
  - Branches no test takes, as the notes of slices 7, 11, 12 list:
    the end of the tindex season (`:825-827`) and the `elif` /
    `else` of the first-step block (`:809-818`); the dead blocks of
    `compute_soltab` (12.4); a lake HRU in snow (`:2102`); the
    `calc_snowevap` corrections; a negative `pkwater_equiv`.
  - Canopy `426->450`: the winter-to-summer changeover never comes
    with an empty store on drb. `461->476` cannot be taken for a
    `cov_type` in 0-4: the `elif cov_type[ii] == _GRASSES` follows
    `> _GRASSES` inside `!= _BARESOIL`.
  - No test in the slice has a lake or inactive HRU (7.1, 7.2), a
    `segment_order` that does not come from `topo_order=` (7.5), a
    `seg_slope` at or below 1e-7 (7.7), a nan `seg_outflow` (7.9),
    a zero row in `soltab_horad_potsw` (11.2), a start date other
    than Jan 1 (12.1), or an HRU south of the equator (12.5).
  - Answer to 15.3: snow, atmosphere, solar geometry and hydraulic
    geometry have no kernel units test, so their rows of
    `coverage_serial.txt` are what the regression tests run. For
    groundwater the one line only the units test runs is known
    (`if sink > stor:`, PLAN). For canopy and channel the table
    cannot tell; the run of 15.3 (run 1 alone) settles it.
- Source: coverage; checklist T (a guard with no test)
- Outcome: open

### 19.11 Comments and docstrings that do not match the code, conventions, side effects

- Where and code, item by item:
  1. `dt` is 1.0 where `CLAUDE.md` says "`dt` is SECONDS (86400.0)
     for real models": `model.run(np.float64(1.0), ...)` at
     `tests/test_prms_atmosphere.py:126`,
     `test_prms_atmosphere_transp_frost.py:147`,
     `test_prms_canopy.py:119`, `test_prms_groundwater.py:103`,
     `test_prms_snow.py:143`, `test_prms_snow_ab_numpy.py:188`,
     `test_prms_hydraulic_geometry.py:89`, and
     `model.run(np.float64(1.0))` in the four MPI files other than
     the channel's. No process of these tests reads `dt` (the
     channel's docstring: "only the channel reads dt"), so no answer
     changes; a process that starts reading it would get 1 second per
     day here. `tests/test_prms_channel*.py` pass `S_PER_TIME`.
  2. `tests/test_prms_channel_mpi.py:112-115`: "runoff + soilzone
     params + gw params + dis_hru vars + IC + the 13 disk forcings".
     The file holds the parameters of all six hru processes and the
     solar tables, no initial state, and three CBH forcings.
     `tests/test_prms_groundwater_mpi.py:75-76`: "params + dis vars +
     IC + forcings"; `gwres_stor` is supplied in `process_dict`
     (`CLAUDE.md`: "Under MPI the initial state is NOT in the input
     file").
  3. `tests/test_prms_snow.py:151`: "Upstream's own five compared
     variables" over a loop of three; `:11-12`: "ONE deviation: tcal
     gets an outlier-FRACTION criterion" where
     `FRACTION_ANSWER_NAMES = ("tcal", "through_rain")`.
  4. `tests/test_prms_atmosphere_transp_frost.py:181-188`:
     `test_final_state` compares `transp_on` with `assert_allclose`
     at 1e-5, though `EXACT_NAMES = ("transp_on",)` and the module
     says it is "compared EXACTLY". Harmless for 0/1 values.
  5. `tests/test_prms_atmosphere_transp_frost.py:17-18`: "and the
     pywatershed repo importable at the mpix root"; the file imports
     no pywatershed and its `skipif` checks data files only.
  6. `tests/test_prms_atmosphere_mpi.py:38` (`# the one to_netcdf
     var (mpixarray limit)`) and `tests/test_prms_snow_mpi.py:52`
     (`# the one to_netcdf var`); `tests/test_prms_groundwater_mpi.py:43`
     streams two with the patched mpixarray. One is a choice, not the
     limit.
  7. `tests/test_prms_snow_ab_numpy.py:166`: `pl.Path(tempfile.mkdtemp())
     / "ab.zarr"` is never removed, so each run leaves a store in the
     system temporary directory; `:47` `sys.path.insert(0,
     str(PYWS_ROOT))` runs at import and stays for the session.
     `tmp_path_factory` and a fixture-scoped path change would not.
  8. `tests/test_prms_groundwater_mpi.py:201-202`: `< 731 *
     BUDGET_TOL` holds the record length as a literal; the canopy and
     snow MPI files use a flat `1.0e-9` for the same check.
  9. `tests/test_prms_canopy.py:11-12`: "pptmix is compared as an
     output" (19.5). `tests/test_prms_snow_ab_numpy.py:10`: "a full
     snow season" (19.4).
  10. `tests/test_prms_atmosphere_mpi.py:166` opens the output with
     `xr.open_dataset` inside `with`, where the other MPI files use
     `xr.load_dataset` (`CLAUDE.md`: tests load and close). Closed
     either way.
- Consequence: a reader is misled about what is compared, what the
  MPI input file holds, and what mpixarray allows; item 7 leaves
  files behind and changes `sys.path` for later modules.
- Source: /code-review (items 1-8); the read (9, 10)
- Outcome: open

## Code findings of slices 7, 11 and 12: why did no test catch this?

"No test" means no test in the 14 files, by reading.

- 7.1, 7.2: every HRU of drb_2yr and ucb_2yr is land (the table of
  slice 7).
- 7.3, 11.3: the canopy test reads `pk_ice_prev` and `freeh2o_prev`
  from pywatershed's answer files, and the two tests of the pack
  differ only around 2.23e-16. In the chains snow is live but
  compared through the channel only.
- 7.4, 11.1: no test supplies part of a pack or of the interception
  state (17.10).
- 7.5: every `segment_order` comes from `topo_order=` (19.2).
- 7.6: the MPI test puts the channel on a replicated serial grid.
- 7.7: no slope at or below 1e-7 and no nan `mann_n` on drb (values
  not read; the run completes at 1e-10).
- 7.8, 11.6, 12.4: the quirk is pywatershed's; a comparison with it
  cannot catch it.
- 7.9: the hydraulic-geometry test is fed pywatershed's
  `seg_outflow`.
- 7.10, 7.11, 7.12, 11.7, 11.8, 12.9, 12.10: text and style.
- 11.2: drb is at mid-latitudes; a zero row of the solar table needs
  polar night (slice 12).
- 11.4: no test reads `ai` across a restart (17.9).
- 11.5, 12.7: every parameter comes from pywatershed's files, with
  valid values and sizes.
- 12.1: both domains start on 1979-01-01, where day of month and day
  of year are both 1.
- 12.2, 12.3: no kernel units test for the atmosphere (15.8).
- 12.5: every HRU of both domains is north of the equator.
- 12.6: not reached in this slice; no restart test of the frost
  leaves (17.9).
- 12.8: a test of values cannot see it.

## Questions handed to this slice

- 15.3: 19.10, last item.
- 16.11 and 1.6 (two or more terms in one budget component, a mix
  of read-only and writable arrays): groundwater has three inputs
  and two outputs, canopy two inputs and four outputs, snow two
  inputs, three outputs and two storage changes. No test in the
  slice mixes: the single-process tests feed every input from files
  (all read-only), and the chains make every input live (all
  writable). So the `TypingError` of 1.6 is reached by no
  configuration here; it needs a partial chain, such as groundwater
  with runoff live and soil zone from files.
- 16.13 and 3.9 (uneven blocks, int and bool state under MPI): all
  five MPI files run 765 HRUs on 4 ranks. The int and bool state
  that is float64 under MPI (snow's flags, `transp_check`,
  `intcp_transp_on`) is compared only through float outputs:
  `pkwater_equiv` at every step, `iso` and `transp_on` at the last
  step, `intcp_transp_on` not at all. `tests/test_prms_channel_mpi.py:323`
  remains the one MPI check of a write flag (`c0`).
- 16.16: 19.1, last item.
- 17.10, 17.2, 17.9: the slice supplies `gwres_stor` and
  `seg_inflow`, each whole; no state group of a port. No test pins
  the restart flag set of a class.
- 18.4 item 1: 19.1 (all three compared at every step and segment,
  in both modes).

## Checklist T, item by item

- A raise or a branch that no test reaches: 19.10.
- A test that cannot fail: 19.7 (the default, the answers-only
  test), 19.8, 19.5 (`pptmix` on drb), 19.2 in part. Expected values
  from the code under test: none; `test_ccov_hru_pin` restates the
  formula independently (19.6).
- A tolerance looser than the standard in `docs/PORTS.md`: the
  channel (19.1), where the document is wrong. Otherwise the tests
  hold pywatershed's own standard or tighter: 1e-13 groundwater,
  1e-12 canopy, 1e-10 solar tables, 1e-5 atmosphere and hydraulic
  geometry, 1e-3 snow plus the bit-for-bit window, budgets at 1e-12.
- A comparison where nan on both sides passes: every
  `np.testing.assert_allclose` of the slice (its `equal_nan` is True
  by default). The channel and outlier-fraction checks use
  `np.isclose`, which counts nan as a difference; the residual checks
  use `<`, which fails on nan. No test asserts that a compared
  answer is finite; whether any answer file of the slice holds a nan
  was not read.
- A comparison of only some output variables, time steps or
  elements: 19.4, 19.5, 19.6; under MPI the last step for most
  variables of every file.
- A serial test with no MPI counterpart where the MPI code differs:
  19.3, 19.5 (`pptmix`), 19.6 (the frost leaf, `ccov_hru`).
- A guard with no test showing that it raises: 19.10.

## Notes for later slices

- Slice 20: 1.6 is reached by a partial chain only (above); do the
  runoff or soil-zone tests feed groundwater or soil zone a mix of
  file and live terms? `tests/test_prms_no_dprst*.py` is where
  PRMSGroundwaterNoDprst is run (19.7).
- Slice 21: hydraulic geometry under MPI is compared on two of five
  variables (`tests/test_prms_stream_temp_mpi.py:59`, 19.8).
- Slice 22: PRMSAtmosphereTranspFrostDyn is run only in
  `tests/test_prms_translate_ag.py:290-342` (atmosphere alone,
  `transp_on` against the oracle); no MPI run.
- Fix phase, runs that settle what this slice did not read:
  - Run `tests/test_prms_snow_ab_numpy.py` with `N_STEPS = 731` and
    see whether bit identity holds through both autumns (19.4).
  - Change `2.0` to `0.0` at `prms_groundwater.py:194` and run the
    slice; expect every test to pass (19.7).
  - Run `tests/test_prms_hydraulic_geometry.py` with Full's
    `_set_depth_params` override removed (19.8).
  - In `tests/test_prms_channel.py`, `snow_disk` mode, pass another
    valid topological order of drb's segments; expect a pass at
    1e-10 (19.1).
  - In `tests/test_prms_channel_mpi.py`, `allgather` the final
    `seg_outflow` and compare the ranks exactly, and compare the
    rank-0 zarr with the serial `snow_live` zarr (19.3).
  - `np.isfinite(...).all()` over the answer files the slice reads
    (checklist item on nan); whether any HRU has `transp_on == 1` on
    1980-12-31 in drb (19.6).
  - Add `assert rank != 0` inside the rank-0 block of one
    `mpi_paths`, under `timeout` (19.9, with 15.13).
