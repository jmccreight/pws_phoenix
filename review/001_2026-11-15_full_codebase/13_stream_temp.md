<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 13: stream temperature (PRMSStreamTemp family, stream shade)](#slice-13-stream-temperature-prmsstreamtemp-family-stream-shade)
  - [Findings](#findings)
    - [13.1 An explicit `seg_close` is used as a 0-based position, though the check beside it treats 0 as "not set"; the behaviour is pywatershed's and has no comment](#131-an-explicit-seg_close-is-used-as-a-0-based-position-though-the-check-beside-it-treats-0-as-not-set-the-behaviour-is-pywatersheds-and-has-no-comment)
    - [13.2 The never-has-flow walk follows only the first upstream segment it finds, so a segment fed by a tributary with HRUs can be marked as never having flow](#132-the-never-has-flow-walk-follows-only-the-first-upstream-segment-it-finds-so-a-segment-fed-by-a-tributary-with-hrus-can-be-marked-as-never-having-flow)
    - [13.3 nan in `seg_tave_water` is three things at once: the static never-has-flow mask, a value the kernel can compute, and initial state a caller may supply](#133-nan-in-seg_tave_water-is-three-things-at-once-the-static-never-has-flow-mask-a-value-the-kernel-can-compute-and-initial-state-a-caller-may-supply)
    - [13.4 `gw_tau` and `ss_tau` are never checked against the width of the silos (366); the njit kernel indexes the silos with them](#134-gw_tau-and-ss_tau-are-never-checked-against-the-width-of-the-silos-366-the-njit-kernel-indexes-the-silos-with-them)
    - [13.5 The silos and their positions are restart state held in python attributes, outside the declarations](#135-the-silos-and-their-positions-are-restart-state-held-in-python-attributes-outside-the-declarations)
    - [13.6 Four behaviours of pywatershed are kept verbatim with no comment that marks them as quirks](#136-four-behaviours-of-pywatershed-are-kept-verbatim-with-no-comment-that-marks-them-as-quirks)
    - [13.7 The rewritten `_rprnvg` is a deliberate departure from pywatershed's order of operations, and `docs/MAINTENANCE.md` has no entry for it](#137-the-rewritten-_rprnvg-is-a-deliberate-departure-from-pywatersheds-order-of-operations-and-docsmaintenancemd-has-no-entry-for-it)
    - [13.8 `derive_aggregation_weights` builds four dense `(nsegment, nhru)` matrices with `nhru + 1` passes of three kernels](#138-derive_aggregation_weights-builds-four-dense-nsegment-nhru-matrices-with-nhru--1-passes-of-three-kernels)
    - [13.9 `initialize()` compares every pair of segments in Python, and holds seven arrays of 366 x `nsegment` on every rank](#139-initialize-compares-every-pair-of-segments-in-python-and-holds-seven-arrays-of-366-x-nsegment-on-every-rank)
    - [13.10 Nothing stops a stream-temperature process on the distributed grid](#1310-nothing-stops-a-stream-temperature-process-on-the-distributed-grid)
    - [13.11 Every kernel in `prms_stream_temp.py` is compiled on every run, though only two of them call into another file](#1311-every-kernel-in-prms_stream_temppy-is-compiled-on-every-run-though-only-two-of-them-call-into-another-file)
    - [13.12 Seven statements in docstrings, descriptions and comments do not match the code](#1312-seven-statements-in-docstrings-descriptions-and-comments-do-not-match-the-code)
    - [13.13 Cleanup: a single-letter name, an argument order, two unused arrays, repeated code](#1313-cleanup-a-single-letter-name-an-argument-order-two-unused-arrays-repeated-code)
  - [Checklist B, item by item](#checklist-b-item-by-item)
  - [Questions handed to this slice](#questions-handed-to-this-slice)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 13: stream temperature (PRMSStreamTemp family, stream shade)

- pws_phoenix commit: 00bdd46 (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Paths: `pws_phoenix/hydrology/prms_stream_temp.py` (2,124 lines),
  `pws_phoenix/hydrology/prms_stream_shade.py` (878 lines)
- Checklist: B. Reviewed 2026-11-18.
- Method: `/code-review high` on the two files (10 candidates and 3
  left out by its cap). The command did not open the pywatershed
  clone, so it could not say which candidates are pywatershed's own
  behaviour; the read below settles that. Kept: all 13, as 13.1-13.4,
  13.6, 13.8, 13.9, 13.11-13.13. Merged: its two candidates about nan
  in `seg_tave_water` are one finding (13.3). Added by the read: 13.5,
  13.7, 13.10. Then both modules beside their pywatershed sources.
  - Compared as text with `diff` (no difference except decorators
    and indentation): `_teak1`, `_equilb`, `_twavg`, `_lat_inflow`,
    `_compute_mixed_inlet_temp` (pywatershed
    `prms_stream_temp.py:2533-2903`), the three loop kernels
    (`:1625-2008`), the three aggregation kernels (`:2201-2531`),
    `_shday_vectorized`, `_solalt`, `_snr_sst` and the quadrature
    tables (`prms_stream_shade.py:20-62`, `:405-482`, `:760-950`).
  - Read side by side: `_shday_statics` + `_shday_from_statics`
    against `_shday` (`prms_stream_shade.py:484-757`); `_rprnvg`
    (`:951-1102`); `initialize()` and `resolve_aggregation_topology`
    against `_set_initial_conditions`, `_initialize_stream_temp`,
    `_compute_upstream_info` (`prms_stream_temp.py:560-864`);
    `calculate()` against `_calculate` (`:900-1025`); the humidity
    leaf against `:2157-2198`; constant shade against
    `prms_stream_shade.py:193-259`; the declarations against
    `:251-391`.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/model.py:442-457`, `:735-790`, `:1791-1854`;
    `pws_phoenix/process.py:464-478`; `pws_phoenix/maps.py:65-90`;
    `pws_phoenix/prms_translate/assemble.py:306-329`,
    `preprocess.py:132-153`; the declarations of the shared segment
    parameters in `prms_channel.py` and
    `prms_hydraulic_geometry.py`; `DEVELOPER.md:149-197`;
    `docs/PORTS.md:83-89`, `:351-400`; the headings of
    `docs/MAINTENANCE.md`; findings 1.3, 2.8, 3.5, 3.16, 5.6, 7.5,
    7.6, 7.9, 7.10, 12.10; `coverage_serial.txt:95-96`.
- Not read: the PRMS Fortran (`stream_temp.f90`, `routing.f90`).
  Where a finding says what PRMS does, it says so from pywatershed's
  comments, and the fix phase must read the source.
- Nothing was run. Every consequence below is from reading.

## Findings

### 13.1 An explicit `seg_close` is used as a 0-based position, though the check beside it treats 0 as "not set"; the behaviour is pywatershed's and has no comment

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1942-1953`,
  `:1616`, `:1838-1842`, `:1891-1892`
- Code:

      if seg_close_param[iseg] == 0:
          errors.append(
              f"segment {iseg} does not have associated HRUs "
              "but seg_close is 0"
          )
      ...
      seg_close = np.asarray(seg_close_param, dtype=np.int32)

  and in the kernels:

      seg_humid[i] = seg_humid[seg_close[i]]

  pywatershed (`prms_stream_temp.py:711-724`) is the same: the
  parameter is validated and then used as it is.
- Consequence: the check reads the parameter as a 1-based segment
  number (0 = none). The kernels read it as a 0-based position. For
  a parameter file that sets `seg_close` (first value not -1), every
  segment without HRUs copies air temperature, cloud cover, melt,
  rain, potential ET and humidity from the segment one position
  after the one named. A value equal to `nsegment` reads past the
  end of the array inside njit code, where numba does not check
  bounds, and the value read becomes a row of the Map weights.
  The value reaches this code unchanged from a PRMS parameter file:
  `pws_phoenix/prms_translate/assemble.py:320` and
  `preprocess.py:145` pass `params["seg_close"].values`.
  The automatic branch (-1) builds 0-based values from `segment_up`
  and is correct. No test runs the explicit branch
  (`coverage_serial.txt:96` lists `1937-1953` as never executed), so
  drb uses -1. Whether PRMS's `seg_close` is 1-based: not read; the
  check says so.
- Source: /code-review, extended by the read
- Outcome: open

### 13.2 The never-has-flow walk follows only the first upstream segment it finds, so a segment fed by a tributary with HRUs can be marked as never having flow

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1224-1249`; the
  claims at `:53-54` and `:1224-1225`
- Code:

      for jseg in range(nsegment):
          if (
              tosegment[jseg] > 0
              and tosegment[jseg] == this_seg + 1
          ):
              if segment_hruarea[jseg] > NEARZERO:
                  has_upstream_hrus = True
                  break
              this_seg = jseg
              found_upstream = True
              break

  pywatershed (`prms_stream_temp.py:753-782`) is the same, line for
  line.
- Consequence: both `break` statements leave the scan at the first
  segment that drains to `this_seg`. Take segment C with no HRUs and
  two tributaries: A (lower position, no HRUs, a headwater) and B
  (has HRUs). The scan finds A, moves to A, finds nothing above A,
  and sets `seg_tave_water[C] = np.nan`. B is never examined, though
  it gives C flow every day. From then on:
  - C is skipped by `if np.isnan(seg_tave_water[jj]): continue`
    (`:750`) on every step; its temperature is nan in the output.
  - The segment below C drops C's outflow from its mixing, because
    `not np.isnan(seg_tave_water[up_idx])` (`:762`) is False. Its
    `seg_tave_upstream` and inlet temperature leave out all the
    water that came through C.
  The comments say "a segment without HRUs anywhere upstream" and
  "(upstream init, verbatim walk)": the first is not what the code
  does, and the second does not say that the walk is incomplete.
  Whether drb's 40 segments without HRUs include such a confluence
  was not determined (it needs a run); the tests compare against
  pywatershed, which has the same walk, so they cannot show it.
- Source: /code-review, confirmed against pywatershed
- Outcome: open

### 13.3 nan in `seg_tave_water` is three things at once: the static never-has-flow mask, a value the kernel can compute, and initial state a caller may supply

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1132-1138`,
  `:1249`, `:784-786`, `:870-872`, `:750`, `:762`;
  `pws_phoenix/model.py:455-456`
- Code:

      seg_tave_water = _meta(
          "variable",
          "Water temperature [degC] (-99.9 invalid; -98.9 NOFLOW; NaN "
          "never-has-flow)",
          restart=True,
      ...
      if seginc_swrad[jj] < -99.0:
          seg_tave_water[jj] = np.nan
          continue
      ...
      for jj in segment_order:
          if not np.isnan(seg_tave_water[jj]):
              seg_tave_water[jj] = -99.9

  and in the Model:

      self._run_initialize_hooks()
      self._apply_supplied_state()

- Consequence:
  - Supplied state. `restart=True` makes `seg_tave_water` a variable
    a caller may supply under its own name, and `Model.input_spec`
    lists it. The Model writes the supplied array after
    `initialize()`, which replaces the nan mask the walk just set.
    The supplied temperatures themselves are never read:
    `_reset_water_temp` sets every value that is not nan to -99.9
    before `_compute_water_temp` reads any of them. So supplying
    `seg_tave_water` has exactly one effect: the never-has-flow
    segments are now computed, and their outflow enters the mixing
    below them. A supplied array with nan in other rows switches
    those segments off for the run.
  - No state is carried. Apart from the nan, nothing in
    `seg_tave_water` passes from one step to the next (the same case
    as 12.6). `initialize()` rebuilds the mask on a restart, so the
    flag is needed only to carry a nan that appeared during the run.
  - A nan that appears during the run stays. `seginc_swrad` is a
    per-step input here; in pywatershed it is computed from static
    topology, so its -99.9 is the same every day. One step with a
    value below -99 sets nan (`:785`), and the segment is skipped on
    every later step. One nan in `seg_humid`, `seg_ccov`,
    `seg_potet` or `seg_shade` gives nan from `_equilb` / `_twavg`
    (`if tw < 0.0` does not catch nan), with the same result. No
    message; the output looks like a never-has-flow segment. With
    the aggregation Maps the -99.9 cannot occur
    (`derive_aggregation_weights` raises `NotImplementedError`,
    `:2102-2109`); a file-fed model has no such guard.
  A separate mask (a bool `parameter_internal`) removes all three.
  pywatershed has the latch too; it has no supplied state.
- Source: /code-review (two candidates, merged), extended by the read
- Outcome: open

### 13.4 `gw_tau` and `ss_tau` are never checked against the width of the silos (366); the njit kernel indexes the silos with them

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:523-548`,
  `:1258-1263`; no check in `initialize()` (`:1165-1270`)
- Code:

      idx_gw = gw_index[jj]
      tau_gw = int(gw_tau[jj])
      gw_silo[jj, idx_gw] = seg_tave_air[jj]
      ...
      for j in range(tau_gw):
          if gw_silo[jj, j] > -98.0:
      ...
      if idx_gw < tau_gw - 1:
          gw_index[jj] = idx_gw + 1

- Consequence: with `gw_tau = 400` on segment `jj`, `gw_index[jj]`
  reaches 399. The write and the loop then touch columns 366-399 of
  a `(nsegment, 366)` array: the next segment's row, or memory past
  the array for the last segment. numba does not check bounds. The
  result is a wrong `seg_tave_gw` on two segments, or corrupted
  memory, with no error. `gw_tau = 0` gives `seg_tave_gw = 0.0`
  every day. `initialize()` validates `seg_length` and nothing
  else. pywatershed has no check either
  (`prms_stream_temp.py:601-650`); with `calc_method="numpy"` numpy
  would raise there. PRMS's declared maximum for the two parameters:
  not read.
- Source: /code-review, confirmed against pywatershed
- Outcome: open

### 13.5 The silos and their positions are restart state held in python attributes, outside the declarations

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1257-1265`,
  `:1379-1391`; `pws_phoenix/model.py:1835-1848`
- Code:

      def get_restart_state(self) -> dict[str, np.ndarray]:
          return {
              "gw_silo": self._gw_silo,
              "ss_silo": self._ss_silo,
              "gw_index": self._gw_index,
              "ss_index": self._ss_index,
          }

- Consequence: checklist B asks for restart variables missing from
  the declarations; these four are the only ones in the package
  (`docs/PORTS.md:422-426`). What follows from holding them outside:
  - A caller cannot supply them. A cold start always begins with
    empty silos, so `seg_tave_gw` on day 1 is that day's air
    temperature, and the average takes `gw_tau` days to fill. A run
    that starts where another stopped can only continue through a
    restart file.
  - Answer to the question 3.5 left for this slice: when
    `_read_restart` finds no `{process}__*` variables (the process
    has another key in `process_dict`, or the file has none),
    `set_restart_state` is not called. The silos keep -99.9 and the
    positions 0. `seg_tave_gw` and `seg_tave_ss` then restart their
    fill, and differ from the continuous run for up to `gw_tau` /
    `ss_tau` days (up to a year), with no message.
  - `set_restart_state` raises `KeyError` on a missing key, ignores
    an extra key, and checks no shape (`self._gw_silo[:] = ...`
    broadcasts a `(1, 366)` array).
  - They have no units, no description, and are not in
    `Model.input_spec`.
  - On the distributed grid the hook state raises
    `NotImplementedError`, but only in `write_restart`
    (`model.py:1836-1842`); see 13.10.
- Source: checklist B; slice 3 note
- Outcome: open

### 13.6 Four behaviours of pywatershed are kept verbatim with no comment that marks them as quirks

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:309-322`,
  `:1928`, `:1933`; `pws_phoenix/hydrology/prms_stream_shade.py:236`
- Code:

      if ql <= NEARZERO:
          # Zero lateral flow
          ...
      elif ql < 0.0:
          # Losing stream (should not happen in PRMS)

      if seg_close_param[0] == -1:

      if segment_up[iseg] == 0:

      horizontal_hour_angle = np.arccos(-tanod)

- Consequence:
  1. `elif ql < 0.0:` follows `if ql <= NEARZERO:` and can never
     run; a negative lateral flow is treated as zero.
     `coverage_serial.txt:96` lists `318-322` as never executed. The
     rule for kernel units tests (every line executes, or carries
     `# units-test: unreachable (<reason>)`) will need the marker
     here.
  2. The first element of `seg_close` decides for every segment
     whether the parameter is automatic.
  3. `segment_up[iseg] == 0` means "no segment drains to `iseg`",
     and also "segment 0 drains to `iseg`". A segment without HRUs
     whose only tributary is segment 0 gets the previous segment in
     route order instead of segment 0. The aggregation kernel says
     so in a comment (`:1762-1765`);
     `resolve_aggregation_topology` does not.
  4. `-tanod` is not clamped to [-1, 1], though `temp` (`:207-212`)
     is. Above about 66.5 degrees of latitude it
     leaves the range on some days, `np.arccos` returns nan, and the
     nan goes into the tables that `initialize()` builds, then into
     `seg_shade`, then into `seg_tave_water`, where it stays (13.3).
     No domain in the test data is that far north.
  All four are in pywatershed (`prms_stream_temp.py:2743-2756`,
  `:702`, `:734`; `prms_stream_shade.py:580`). The port's rule is
  "upstream quirks are preserved and commented".
- Source: /code-review (item 1), checklist B (items 2-4)
- Outcome: open

### 13.7 The rewritten `_rprnvg` is a deliberate departure from pywatershed's order of operations, and `docs/MAINTENANCE.md` has no entry for it

- Where: `pws_phoenix/hydrology/prms_stream_shade.py:757-765`,
  `:788-812`, `:839-863`; `docs/MAINTENANCE.md` (no entry);
  `docs/PORTS.md:83`
- Code:

      sinals = sinod + (cosod * coshrs)
      if sinals > 1.0:
          sinals = 1.0
      cosals = math.sqrt((1.0 - sinals) * (1.0 + sinals))
      if sinals == 0.0:
          sinals = NEARZERO
      if cosals == 0.0:
          cosals = NEARZERO

  pywatershed (`prms_stream_shade.py:1019-1026`):

      temp = sinod + (cosod * coshrs)
      if temp > 1.0:
          temp = 1.0
      als = np.arcsin(temp)
      cosals = np.cos(als)
      sinals = np.sin(als)
      if sinals == 0.0:
          sinals = NEARZERO

- Consequence: none found for the answers. The algebra was checked:
  `cos(arcsin(x)) = sqrt((1-x)(1+x))` holds on [-1, 1];
  `arccos(temp)` is in [0, pi], so its sine is the non-negative
  root; negating `azs` negates only the sine. The one added guard
  (`cosals == 0.0`) changes the value only where `sinals` is exactly
  1.0, by at most `vhe * 1e-6` metres of shadow width.
  The departure is recorded in the function's docstring, the module
  docstring and `docs/PORTS.md:83`. The project rule is that a
  deliberate departure is logged in `docs/MAINTENANCE.md`
  ("two so far"); this is a third, and it is the only place where a
  ported helper does not keep pywatershed's order of operations.
  The split of `_shday` into `_shday_statics` and
  `_shday_from_statics` was compared line by line and is not a
  departure: every operation before the `_rprnvg` call is unchanged,
  and `hrrh` is the value pywatershed passes in each branch.
  `coverage_serial.txt:95` lists the three guards (`790`, `793`,
  `795`) and the level-bank branch of the split (`268-271`) as never
  executed (slice 21).
- Source: checklist B
- Outcome: open

### 13.8 `derive_aggregation_weights` builds four dense `(nsegment, nhru)` matrices with `nhru + 1` passes of three kernels

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:2111-2122`; the
  docstring at `:2015-2016`; callers
  `pws_phoenix/prms_translate/assemble.py:315`, `preprocess.py:140`
- Code:

      weights = {
          kk: np.zeros((nsegment, nhru), dtype=np.float64)
          for kk in ("flow", "swrad", "met", "humid")
      }
      for jj in range(nhru):
          probe[jj] = 1.0
          _probe_kernels(probe)
          weights["flow"][:, jj] = out_flow

- Consequence: on drb (765 HRUs, 456 segments) this is 11 MB and
  instant. On the NHM-scale domain now being translated (114,958
  HRUs; about 57,000 segments assumed, not checked) one matrix is
  114,958 x 57,000 x 8 bytes = 52 GB, and there are four. Each of
  the 114,959 probes runs three kernels that loop over every HRU and
  every segment, about 2e10 loop passes per kernel. The build cannot
  finish on that domain. Each HRU contributes to a few segments, and
  `Map` keeps only the nonzeros after construction
  (`maps.py:65-76`), so the dense form is needed by nothing but the
  construction path (2.8). `preprocess.py:149` writes the same dense
  matrices to the artifact file. The docstring calls the cost
  "one-time build allocations (memory directive: justified)"; that
  holds on drb only.
  The cloud-cover block inside `_compute_segment_aggregates_numba`
  (`:1723-1739`) runs in every probe on stand-in values and its
  output is discarded (12.10).
- Source: /code-review
- Outcome: open

### 13.9 `initialize()` compares every pair of segments in Python, and holds seven arrays of 366 x `nsegment` on every rank

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1216-1222`,
  `:1235-1245`, `:1258-1263`, `:1437-1448`
- Code:

      for jseg in range(nsegment):
          count = 0
          for iseg in range(nsegment):
              if tosegment[iseg] > 0 and tosegment[iseg] == jseg + 1:
                  self._upstream_idx[jseg, count] = iseg

      # ... 5 x (366, nsegment) float64 = 6.7 MB on drb, the
      # same order as the gw/ss silos above

- Consequence: none on drb (456 segments). At about 57,000 segments
  the double loop is 3.2e9 comparisons of numpy scalars in the
  interpreter, and the never-has-flow walk scans all of `tosegment`
  again at every hop instead of reading the `_upstream_idx` table
  just built. One pass over `tosegment` fills the table in the same
  order. The five shade tables are then 834 MB and the two silos
  334 MB. The segment grid is replicated under MPI, so every rank
  repeats the loops and holds its own 1.2 GB. The comment gives the
  drb figure only. pywatershed has the same loops
  (`prms_stream_temp.py:843-864`, `:753-782`); the tables are the
  port's (Oct 2026).
- Source: /code-review (two candidates, merged)
- Outcome: open

### 13.10 Nothing stops a stream-temperature process on the distributed grid

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1167`,
  `:1216-1222`, `:748-768`; `pws_phoenix/model.py:1836-1842`
- Code:

      nsegment = refs["seg_length"].shape[0]
      ...
      if tosegment[iseg] > 0 and tosegment[iseg] == jseg + 1:
      ...
      for jj in segment_order:
          ...
          up_idx = upstream_idx[jj, kk]

- Consequence: the same exposure as 5.6 and 7.6. With
  `control["mpi_grid"]` set to the segment grid, `nsegment` is one
  rank's block, while `tosegment` and `segment_order` hold positions
  in the whole network. `initialize()` builds a wrong upstream
  table, and the njit kernels read and write outside the rank's
  arrays. The only guard is the `NotImplementedError` for hook state
  on the distributed grid, which is raised in `write_restart`, after
  the run. Not run.
- Source: slice 5 note (a serial / MPI rule that no code enforces)
- Outcome: open

### 13.11 Every kernel in `prms_stream_temp.py` is compiled on every run, though only two of them call into another file

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:97-99`; the rule
  at `DEVELOPER.md:183-192`
- Code:

      # NOT cache=True: these call prms_stream_shade.py's jitted helpers; a
      # cached caller never notices an edited callee (DEVELOPER.md "Numba cache").
      @nb.jit(nopython=True)
      def _teak1(a_coef, b_coef, c_coef, d_coef, teq, maxiter_sntemp):

- Consequence: `DEVELOPER.md` allows `cache=NUMBA_CACHE` "only where
  every jitted callee and every captured constant lives in the
  kernel's own file". That holds for `_teak1`, `_equilb`, `_twavg`,
  `_lat_inflow`, `_compute_mixed_inlet_temp`,
  `_update_running_avg_temp`, `_compute_lateral_temp`,
  `_compute_water_temp`, `_reset_water_temp` and the three
  aggregation kernels. Only `_shade_statics_table` and `_shade_all`
  call `prms_stream_shade.py`. The twelve are compiled in every
  Python process and on every MPI rank ("seconds per process"). The
  comment sits above `_teak1`, which calls nothing. `DEVELOPER.md`
  lists the whole module as excluded, so the rule and the list
  disagree for this file.
- Source: /code-review
- Outcome: open

### 13.12 Seven statements in docstrings, descriptions and comments do not match the code

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:22-23`, `:33-35`,
  `:37-44`, `:51-52`, `:1001`, `:1157-1159`, `:1224-1225`
- Code:

      THE structural departure (design in memory/pws-phoenix-status; the
      chain stage will complete it)
      ...
      The hru->segment aggregation port (and the humidity leaves) is the
      chain stage.
      ...
      never read); seg_close / hru->seg aggregation (chain stage); Budget;
      restart; calc_method; verbose.

- Consequence, text only:
  1. `:22` points at "memory/pws-phoenix-status", which is not in
     the repository.
  2. `:33-35` and `:43` speak of the aggregation and the humidity
     leaf as future work ("the chain stage"). The file contains the
     three aggregation kernels, `resolve_aggregation_topology`,
     `derive_aggregation_weights` and `PRMSStreamTempSegHumidity`.
  3. `:43-44` lists "restart" as deliberately not ported. The class
     has `restart=True` on `seg_tave_water` and the two restart
     hooks (the same case as 7.10).
  4. `:51-52` says pywatershed edits both parameters in place. It
     edits `seg_slope`; `seg_length_km` is a separate attribute
     there too (`prms_stream_temp.py:630`).
  5. `:1001` names the hook `_compute_shade(declination,
     summer_flag)`; the signature is `_compute_shade(self, doy,
     declination, summer_flag)`.
  6. `:1158` describes `seg_shade` as "(dynamic shday)" on the base
     class; PRMSStreamTempConstantShade inherits the description.
  7. `:1224-1225` and `:53-54` say "without HRUs anywhere upstream"
     (13.2).
  A reader deciding whether restart or the aggregation is supported
  gets the wrong answer from the module docstring.
- Source: /code-review, extended by the read
- Outcome: open

### 13.13 Cleanup: a single-letter name, an argument order, two unused arrays, repeated code

- Where: `pws_phoenix/hydrology/prms_stream_temp.py:1252`,
  `:865-867`, `:1266-1267`, `:1202-1210` and `:1916-1920`;
  `pws_phoenix/hydrology/prms_stream_shade.py:781-825` and
  `:832-876`
- Code:

      k = np.arange(MAX_DAYS_PER_YEAR) + 1.0

      def _reset_water_temp(
          segment_order: np.ndarray, seg_tave_water: np.ndarray
      ) -> None:

      self._gw_sum = np.zeros(nsegment, dtype=np.float64)
      self._ss_sum = np.zeros(nsegment, dtype=np.float64)

- Consequence: none for the answers.
  - `k` is in `initialize()`, which is the port's code, not an
    extract; the rule is no single-letter names. The rewritten
    `_rprnvg` keeps pywatershed's `for n in range(15):`. The
    verbatim kernels use `j`, `i`, `b`, `c`, `d`, `r`; whether an
    extract is exempt is not written down (7.12, 3.24).
  - `_reset_water_temp` is the port's kernel and takes its output
    buffer second; the convention is output buffers first.
  - `_gw_sum` and `_ss_sum` are allocated and passed to
    `_update_running_avg_temp`, which never reads or writes them
    (pywatershed's signature, kept).
  - `segment_hruarea` is accumulated by the same five lines in
    `initialize()` and in `resolve_aggregation_topology`.
  - The east-bank and west-bank loops of `_rprnvg` are the same 45
    lines with other arguments; pywatershed's are too, but the port
    rewrote both.
- Source: /code-review
- Outcome: open

## Checklist B, item by item

- Order of operations per element: the kernels and helpers of
  `prms_stream_temp.py` are pywatershed's text. `calculate()` keeps
  pywatershed's order (running averages, lateral temperature, zero
  `seg_tave_upstream`, shade, reset, water temperature); the
  aggregation that pywatershed runs first is done by Maps before
  the process. The one change of order inside a helper is 13.7.
- A kernel that allocates per step, returns a value, or uses
  fastmath: none. The loop kernels write in place. The helpers
  return scalars (`_equilb` returns nine), as pywatershed's do.
  `calculate()` and the two `_compute_shade` leaves allocate
  nothing per step; `refs["seg_humidity"][time.month - 1, :]` is a
  view. No fastmath; `parallel=True` was dropped from
  `_shday_vectorized` and the reason is in the module docstring.
- A frozen parameter that is edited: none. `_seg_slope` and
  `seg_length_km` are derived copies.
- A float field without units: none; `_meta` requires `units`.
  `lat_temp_adj` is `delta_degC`, as it should be. The shared
  segment parameters agree with `prms_channel.py` and
  `prms_hydraulic_geometry.py` (`seg_length` m, `seg_slope`
  fraction, `seg_outflow` cfs, `seg_flow_width` m). `gw_tau` and
  `ss_tau` are float64 `days`, truncated by `int()` in the kernel.
- A budget missing a term: no budget is declared. pywatershed
  declares an energy budget over eleven flux variables
  (`prms_stream_temp.py:341-391`) and no mass budget. The module
  docstring says the flux variables are not ported;
  `docs/PORTS.md:84` says the same. `CLAUDE.md` "Budgets" names only
  PRMSChannel as declaring none.
- An upstream quirk kept without a comment: 13.1, 13.2, 13.6. Kept
  with a comment: the swrad fallback that can divide by the area
  twice (`:1991-1997`, `docs/PORTS.md:369-373`), the dependence of
  the automatic `seg_close` on the segment order (`:1905-1909`).
- A departure not logged in `docs/MAINTENANCE.md`: 13.7.
- A restart variable missing from the declarations: 13.5. One that
  is declared and carries no state: 13.3.

## Questions handed to this slice

- 3.5 (hook state that `_read_restart` can skip): answered in 13.5.
- 7.5 (`segment_order` taken on trust): it applies. Every value is
  reset to -99.9 before the loop, so a segment computed before its
  tributary reads -99.9 there, fails
  `seg_tave_water[up_idx] > NOFLOW_TEMP` (`:763`), and leaves that
  tributary's flow out of its inflow and its mixing, with no
  message. The docstring's "any valid upstream-before-downstream
  order is equivalent for the mixing" is right; nothing checks that
  the order is one. No separate finding: the fix belongs with 7.5.
- 7.9 (zero geometry from a nan `seg_outflow`): `seg_outflow <= 0.0`
  (`:789`) is False for nan, so the segment goes on to `_equilb`
  with `seg_flow_width = 0.0`, and `q_init / seg_flow_width`
  (`:227`) divides by zero. Whether numba raises
  `ZeroDivisionError` there or returns inf: not run. With a width
  that comes from the same step's outflow, zero width and positive
  outflow cannot occur together.
- 12.10 (the cloud-cover block): it is still at `:1723-1739`. In
  the package it runs only inside the probes of
  `derive_aggregation_weights`, on stand-in values (13.8). The tests
  use it as the reference for `ccov_hru`.

## Notes for later slices

- Inactive HRUs (7.2; no finding per process): the aggregation
  weights are built from `hru_segment` and `hru_area` for every
  HRU; pywatershed's stream temperature has no `hru_type` either.
- Slice 14 (translation): `seg_close` goes from the PRMS parameter
  file to `derive_aggregation_weights` unchanged (13.1); the
  artifact file holds the four dense weight matrices (13.8);
  `gw_tau`, `ss_tau`, `seg_humidity`, `segshade_sum`,
  `segshade_win` are not named in `prms_translate/prms_units.py`,
  so their units come from PRMS's metadata strings; the humidity
  Map takes `humidity_hru` in percent to `seg_humid` as a fraction
  through weights that hold the factor 0.01, and units mode does
  not compare a Map's two ends (3.16).
- Slice 15 (harness): no kernel units test for this family. Before
  one can be written: the unreachable branch of `_twavg` needs its
  marker (13.6); the kernels assign and compare bare sentinels in
  dimensional arrays (`-99.9`, `-98.0`, `NOFLOW_TEMP`, `0.0`,
  `np.nan`); `_equilb` mixes empirical constants with units pint
  cannot follow.
- Slice 17 (restart tests): does
  `tests/test_restart_processes.py::test_stream_temp` read the
  restart under a different `process_dict` key, or from a file
  without the hook variables (3.5, 13.5)? Does any test supply
  `seg_tave_water` (13.3)?
- Slice 21 (stream-temperature tests), from
  `coverage_serial.txt:95-96`. Never executed:
  - the explicit `seg_close` branch and both raises of
    `resolve_aggregation_topology` (`1937-1953`): why 13.1 was not
    caught;
  - every `seginc_swrad < -99.0` branch (`520`, `629`, `785-786`,
    `1819`) and the `NotImplementedError` at `2103`;
  - the three no-flow exits of `_compute_water_temp` (`790-791`,
    `798-799`, `810-811`);
  - the `seg_length` raise (`1182`);
  - in the shade module, the level-bank branch that the split had
    to restate (`268-271`), the sunrise and sunset shortcuts
    (`277-278`, `281`, `314`), every clamp of `sti` and `svi`
    (`423-429`), and the three guards of the rewritten `_rprnvg`
    (`790`, `793`, `795`).
  No test has a segment without HRUs at a confluence of a headwater
  without HRUs and a tributary with HRUs (13.2), a `gw_tau` above
  366 (13.4), or a latitude above the polar circle (13.6). The
  constant-shade leaf has no parity test (`docs/PORTS.md:85`). All
  parity tests compare against pywatershed, which shares 13.1,
  13.2, 13.4 and 13.6.
