<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 6: the STARFIT node types](#slice-6-the-starfit-node-types)
  - [Findings](#findings)
    - [6.1 A warm-started daily node writes nan `lake_release` and `lake_spill` on its first day: two values it carries across days are not restart variables](#61-a-warm-started-daily-node-writes-nan-lake_release-and-lake_spill-on-its-first-day-two-values-it-carries-across-days-are-not-restart-variables)
    - [6.2 A nan NOR parameter gives a finite release without a message; pywatershed gives nan](#62-a-nan-nor-parameter-gives-a-finite-release-without-a-message-pywatershed-gives-nan)
    - [6.3 A zero divisor stops the run with `ZeroDivisionError` inside the graph kernel; pywatershed continues with inf or nan](#63-a-zero-divisor-stops-the-run-with-zerodivisionerror-inside-the-graph-kernel-pywatershed-continues-with-inf-or-nan)
    - [6.4 `source_sink_storage_min` and the request are never checked for nan, and with nan the port and pywatershed take different branches](#64-source_sink_storage_min-and-the-request-are-never-checked-for-nan-and-with-nan-the-port-and-pywatershed-take-different-branches)
    - [6.5 In a cfs graph the hourly types write `lake_storage_old` 35.3 times too large; the comment calls the double conversion harmless](#65-in-a-cfs-graph-the-hourly-types-write-lake_storage_old-353-times-too-large-the-comment-calls-the-double-conversion-harmless)
    - [6.6 Four unit declarations do not match the values](#66-four-unit-declarations-do-not-match-the-values)
    - [6.7 The kernels use bare `0.0` for zeros of MCM and m^3/s](#67-the-kernels-use-bare-00-for-zeros-of-mcm-and-m%5E3s)
    - [6.8 `starfit_advance_type` writes every row of the graph, and a second call in one advance sets `lake_storage_change` to zero](#68-starfit_advance_type-writes-every-row-of-the-graph-and-a-second-call-in-one-advance-sets-lake_storage_change-to-zero)
    - [6.9 The comment at `post_release_calculations` and the line 20 lines below it disagree about `max(x, 0.0)` and nan](#69-the-comment-at-post_release_calculations-and-the-line-20-lines-below-it-disagree-about-maxx-00-and-nan)
    - [6.10 The own-row lookup and the three hook wrappers are written out several times](#610-the-own-row-lookup-and-the-three-hook-wrappers-are-written-out-several-times)
  - [Checklist C, item by item](#checklist-c-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 6: the STARFIT node types

- pws_phoenix commit: c2de500 (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Paths: `pws_phoenix/hydrology/starfit_flow_node.py`,
  `pws_phoenix/hydrology/starfit_daily_flow_node.py`,
  `pws_phoenix/hydrology/starfit_source_sink_flow_node.py`
- Checklist: C (checklist B, plus). Reviewed 2026-11-17.
- Method: `/code-review high` on the three files (10 candidates;
  8 kept, some merged; 1 dropped, see the checklist section), then a
  read of each file against checklist C beside its pywatershed source
  (`pywatershed/hydrology/starfit.py` `_calc_istarf_release`,
  `max_nor`, `min_nor`, `StarfitFlowNode`;
  `pywatershed/hydrology/starfit_source_sink_flow_node.py`), with the
  callers outside the slice (`flow_graph.py` `advance`, `calculate`,
  `check_initial_state`; `Model` restart restore, `model.py:1111`)
  and `tests/test_restart_flow_graph.py`. CodeGraph
  (`codegraph_explore`) worked; it warned that the index was built
  by an older extraction engine (v1.6.0 vs v1.6.3). The session
  executed no project code. A finding marked "not run" rests on
  reading and needs a short run to confirm before it is fixed.

The numerics of `istarf_release`, the hourly pre- and post-release
steps, the daily substep and the source/sink diversion agree with
pywatershed line for line in order of operations, except where a
finding says otherwise. 6.1-6.4 give a wrong answer, or a different
answer from pywatershed, for some input; none of them is seen by a
test.

## Findings

### 6.1 A warm-started daily node writes nan `lake_release` and `lake_spill` on its first day: two values it carries across days are not restart variables

- Where: `pws_phoenix/hydrology/starfit_daily_flow_node.py:138-144`,
  `:183`, `:194-195`, `:58-60`
- Code:

      # last substep of a day: read the rates computed at the
      # END of the previous day
      state.lake_release_sub[inode] = (
          state.lake_release_sub[inode] * io_to_cms
      )
      state.lake_spill_sub[inode] = state.lake_spill_sub[inode] * io_to_cms
      ...
      state.lake_release[inode] = state.lake_release_sub[inode]
      state.lake_spill[inode] = state.lake_spill_sub[inode]

  and the module docstring:

      pywatershed allocates lake_{release,spill}_sub_next and
      lake_availability_status_next for compute_daily but never uses
      them -- NOT ported; only lake_outflow_sub_next is real state.

- Consequence: `lake_release_sub` and `lake_spill_sub` are set in the
  tail of one day (`:160-164`, `:183`) and read on the last substep
  of the next day, so they cross the day boundary, like
  `lake_outflow_sub_next`. Only `lake_outflow_sub_next` is
  `restart=True`. A warm start resumes at `_start_index >= 1`
  (`pws_phoenix/model.py:1111`), so the first-run block
  (`tctx.itime_step == 0`) is skipped and the two fields hold the
  allocation fill (nan) when the first resumed day's last substep
  reads them. That day's `lake_release` and `lake_spill` are nan;
  from the next day on they are correct. The routed flow is not
  affected (`node_outflows` = `lake_outflow`, from
  `lake_outflow_sub_next`). The docstring's "only
  lake_outflow_sub_next is real state" is wrong for the port.
  `perfect_restart` (`tests/test_restart_processes.py:70`) compares
  only the values at the END of the run, so
  `tests/test_restart_flow_graph.py::test_starfit_graph[starfit_daily]`
  passes. Smaller, same cause: the daily node reads `lake_storage_sub`
  only on the run's first substep (`:118`), so its restart files
  carry the cold-start storage under a `restart=True` name, and its
  written `lake_storage_sub` output never changes.
- Source: /code-review, confirmed by reading (not run)
- Outcome: open

### 6.2 A nan NOR parameter gives a finite release without a message; pywatershed gives nan

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:131-148`,
  `:378-385`; also `nor_midpoint_initial_storage`, `:458-475`
- Code:

      max_normal = min(
          state.NORhi_max[inode],
          max(
              state.NORhi_min[inode],
              state.NORhi_mu[inode]
              + state.NORhi_alpha[inode] * np.sin(...)
              + state.NORhi_beta[inode] * np.cos(...),
          ),
      )

  pywatershed (`max_nor`): `np.minimum(NORhi_max, np.maximum(NORhi_min, ...))`.
- Consequence: numba's builtin `min` / `max` follow Python's rule
  (keep the first argument unless the later one compares smaller /
  larger), and every comparison with nan is False. So a nan harmonic
  (nan `NORhi_mu`, `_alpha` or `_beta`) gives `max(NORhi_min, nan) =
  NORhi_min`, and a nan `NORhi_min` gives `min(NORhi_max, nan) =
  NORhi_max`: a finite `max_normal`. pywatershed's `np.minimum` /
  `np.maximum` give nan, then a nan release and a nan storage, which
  are visible. The port computes a release from a bound the data did
  not give. Same for `min_normal` and for the NOR midpoint in
  `nor_midpoint_initial_storage` (plain Python `min` / `max`). Only
  `Obs_MEANFLOW_CUMECS` is checked for nan on the type's rows
  (`:379`); nothing checks the other 19 STARFIT parameters. Not
  logged in `docs/MAINTENANCE.md`.
- Source: checklist B / A ("an error that passes silently"). Not
  run: rests on numba's documented Python-compatible `min` / `max`.
- Outcome: open

### 6.3 A zero divisor stops the run with `ZeroDivisionError` inside the graph kernel; pywatershed continues with inf or nan

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:157`, `:170-172`,
  `:379`
- Code:

      standardized_inflow = (forecasted_weekly_volume / mean_weekly_volume) - 1.0
      ...
      availability_status = (100.0 * storage / capacity - min_normal) / (
          max_normal - min_normal
      )

- Consequence: `@numba.njit` uses `error_model="python"` by default
  (nothing in `pws_phoenix/` sets `error_model`), which raises
  `ZeroDivisionError` on a float division by zero. A row with
  `Obs_MEANFLOW_CUMECS == 0.0`, `GRanD_CAP_MCM == 0.0`, or NOR bounds
  that clamp to one value (`max_normal == min_normal`) passes the
  build check and stops the run on the first substep, with no node
  index or name in the message. pywatershed divides numpy values:
  inf or nan with a RuntimeWarning, and for `max_normal ==
  min_normal` with storage outside the NOR the inf status selects
  the above- or below-normal release, which the min / max clamps
  then make finite. So pywatershed runs this input and the port
  does not. Not logged in `docs/MAINTENANCE.md`. Under MPI the graph
  is a replicated serial grid, so every rank raises: no hang.
- Source: /code-review (not run)
- Outcome: open

### 6.4 `source_sink_storage_min` and the request are never checked for nan, and with nan the port and pywatershed take different branches

- Where: `pws_phoenix/hydrology/starfit_source_sink_flow_node.py:85-103`
- Code:

      if source_sink_vol >= 0.0:
          storage += source_sink_vol
      elif storage < min_storage:
          source_sink_vol = 0.0
      else:
          if (storage + source_sink_vol) < min_storage:
              ...
          else:
              storage += source_sink_vol

  pywatershed:

      if source_sink_vol >= zero: ...
      elif (source_sink_vol < zero) and (storage < min_storage): ...
      elif (source_sink_vol < zero) and (storage >= min_storage): ...
      # no else

- Consequence: the port's `elif` / `else` drop the conditions
  pywatershed spells out. They are equivalent for finite numbers,
  not for nan:
  - nan `source_sink_storage_min`: pywatershed takes no branch, so
    `lake_storage_after_source_sink` = the undiverted storage while
    the storage change still applies the sink. The port reaches the
    inner `else` and diverts: `storage += source_sink_vol`. The
    release is computed from different storages, and the sink is
    applied in full with no minimum.
  - nan request with `storage < min_storage`: pywatershed keeps the
    nan (visible); the port sets the diversion to 0.0.
  Neither code checks either value; pywatershed raises only on a
  missing date. Same family as 5.3 (`SourceSinkFlowNode`).
- Source: /code-review (the nan minimum), checklist B (the branch
  difference). Not run.
- Outcome: open

### 6.5 In a cfs graph the hourly types write `lake_storage_old` 35.3 times too large; the comment calls the double conversion harmless

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:336-342`, `:350`,
  `:433`
- Code:

      # lake_storage_old *= factor is pywatershed-verbatim
      # (it double-converts a value already in io units -- harmless:
      # storage_old is rewritten from storage in the next advance, ...
      state.lake_storage_old[inode] = state.lake_storage_old[inode] * cms_to_io

- Consequence: `advance` copies `lake_storage` (already in Mcf after
  the previous `finalize`) into `lake_storage_old`; the hourly types
  never convert it back (only the daily `_prepare` does, `:90-91` of
  the daily module, which is why pywatershed calls the line
  "necessary"). `finalize` then multiplies by 35.31 again. The value
  is harmless for the state, as the comment says, but
  `lake_storage_old` is a declared `kind="variable"` with `units="io
  storage units"`, and output is collected after `calculate`: every
  written value is 35.31 times the previous storage. pywatershed has
  the same arithmetic, but its node attribute is not a model output.
  Quirk kept with a comment that understates it.
- Source: /code-review, confirmed by reading (not run)
- Outcome: open

### 6.6 Four unit declarations do not match the values

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:522-534`,
  `:557-569`, `:650-667`, `:720-726`, `:741-747`;
  `pws_phoenix/hydrology/starfit_daily_flow_node.py:194-195`
- Code:

      "NORhi_alpha": DataArrayMeta(... units="fraction", ...)
      ...
      state.NORhi_mu[inode] + state.NORhi_alpha[inode] * np.sin(...)

      "io_to_cms": DataArrayMeta(... units="fraction", ...)

      "lake_release_sub": DataArrayMeta(... units="cms", ...)
      # daily module:
      state.lake_release_sub[inode] = state.lake_release_sub[inode] * cms_to_io

- Consequence:
  - `NORhi_alpha`, `NORhi_beta`, `NORlo_alpha`, `NORlo_beta` are
    added to `NORhi_mu` / `NORlo_mu` (`"percent"`), so they are
    percent. As `"fraction"`, units mode `"convert"` would scale them
    by 100 against a file that says percent, and a pint run of the
    kernel would rescale one operand of the sum.
  - `io_to_cms` / `cms_to_io` are cfs <-> cms factors (0.0283 /
    35.31 in a cfs graph); `"fraction"` is defined as a proportion.
  - In a cfs graph the daily type leaves `lake_release_sub` and
    `lake_spill_sub` in cfs between days; the shared declaration
    (right for the hourly types) says cms. One field, two units
    depending on the row's type.
- Source: /code-review, checklist B ("a float field without units")
- Outcome: open

### 6.7 The kernels use bare `0.0` for zeros of MCM and m^3/s

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:233`, `:253`,
  `:258`; `pws_phoenix/hydrology/starfit_daily_flow_node.py:120`,
  `:160`, `:180`; `pws_phoenix/hydrology/starfit_source_sink_flow_node.py:88`,
  `:94`
- Code:

      if (state.lake_storage_sub[inode] + state.lake_storage_change_sub[inode]) < 0.0:
      state.lake_storage_sub[inode] = max(..., 0.0)  # MCM
      state.lake_spill_sub[inode] = 0.0

- Consequence: `CLAUDE.md` ("Units mode"): "A zero or a threshold of
  a working unit is a named module float ... never a bare literal."
  The FlowGraph node types have no kernel units test yet (known
  before the start), so nothing fails today; the planned test would
  need these renamed first. The accumulator zeroing in the prepare
  functions (`:95-101` etc.) is the same case.
- Source: /code-review
- Outcome: open

### 6.8 `starfit_advance_type` writes every row of the graph, and a second call in one advance sets `lake_storage_change` to zero

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:423-433`;
  `pws_phoenix/flow_graph.py:501-503`
- Code:

      change = dataset["lake_storage_change"].values
      change[:] = dataset["lake_storage"].values
      change -= dataset["lake_storage_old"].values
      dataset["lake_storage_old"].values[:] = dataset["lake_storage"].values

- Consequence: checklist C asks for writes to the type's own rows
  only; this writes all rows, including channel and other rows (nan
  minus nan there). `FlowGraph.advance` calls it once per STARFIT
  type, so in a graph with two STARFIT types the second call computes
  `storage - storage` = 0.0 for `lake_storage_change`. The docstring
  says "the storage_old march is idempotent and lake_storage_change
  is transient": true today (every type overwrites
  `lake_storage_change` before it is read or harvested), but the
  safety rests on that, not on the row rule.
- Source: checklist C
- Outcome: open

### 6.9 The comment at `post_release_calculations` and the line 20 lines below it disagree about `max(x, 0.0)` and nan

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:228-230`, `:251-254`
- Code:

      # ... pywatershed's max(x, x * 0.0) form kept -- it
      # propagates nan where max(x, 0.0) would not)
      ...
      state.lake_storage_sub[inode] = max(
          state.lake_storage_sub[inode] + state.lake_storage_change_sub[inode],
          0.0,
      )  # MCM

  pywatershed: `np.maximum(self._lake_storage_sub + ..., zero)`.
- Consequence: one of the two is wrong. Under Python's rule (6.2),
  `max(nan, 0.0)` is nan, so the comment is wrong and line 251
  matches pywatershed. If the comment were right, line 251 would turn
  a nan storage into 0.0 where pywatershed keeps nan. The branch the
  comment sits in cannot see a nan (`nan < 0.0` is False), so its
  `max(x, x * 0.0)` form changes nothing either way.
- Source: checklist B (not run)
- Outcome: open

### 6.10 The own-row lookup and the three hook wrappers are written out several times

- Where: `pws_phoenix/hydrology/starfit_flow_node.py:374-376`,
  `:411-412`, `:857-869`; `starfit_daily_flow_node.py:243-270`;
  `starfit_source_sink_flow_node.py:221-238`
- Code:

      names = list(dataset["node_type"].attrs["node_type_names"])
      mine = dataset["node_type"].values == names.index(type_name)

- Consequence: two copies of the row lookup, and three classes with
  the same `initialize_type` / `check_initial_state_type` /
  `advance_type` bodies that differ only in `type_name`. A change to
  how a type finds its rows is made in each copy. Cleanup only.
- Source: /code-review
- Outcome: open

## Checklist C, item by item

- Order of operations per element: matches pywatershed in
  `istarf_release` (including the `7.0 * flow * 24.0 * 60.0 * 60.0`
  form and `(1 + Release_min) / 7.0`), the hourly pre- and
  post-release steps (the added `+ sink_source_sub` is `+ 0.0` for
  the plain node), the daily substep (`release * (m3ps_to_MCM / 24 /
  60 / 60)`, `* (1.0 / m3ps_to_mcm)` = pywatershed's precomputed
  `_MCM_to_m3ps`), and the source/sink storage change. `epiweek`:
  `min(tctx.epiweek, 52)` as pywatershed's call site;
  `nor_midpoint_initial_storage` uses the raw week, as pywatershed's
  `datetime_epiweek` in node `__init__`. Exceptions: 6.2 (nan in
  `min` / `max`), 6.4 (nan in the branch structure).
- Kernel allocates, returns, or fastmath: no allocation, no
  fastmath. `istarf_release` returns a tuple of two scalars (a
  helper, not a kernel; pywatershed returns the same pair).
- Frozen parameter edited: no. `initialize_starfit_type` writes only
  the three `parameter_internal` fields, on own rows, before the
  freeze; it raises on nan `Obs_MEANFLOW_CUMECS` instead of filling
  it.
- Float field without units: none missing; four wrong (6.6).
- Budget missing a term: none. The hourly types' storage change is
  pywatershed's state delta, so the graph budget checks something
  for them; the source/sink type adds `node_sink_source` as the
  exchange. The daily type's budget is an identity by definition
  (`lake_storage_change_flow_units = lake_inflow - lake_outflow`),
  as `docs/PORTS.md:265-269` already says.
- Upstream quirk without a comment: none without one; 6.5 has a
  comment that understates the effect. The obsin-style "do not fix"
  items for STARFIT (the daily one-day lag) are documented in the
  daily module docstring.
- Departure not logged in `docs/MAINTENANCE.md`: 6.2, 6.3, 6.4.
- Restart variable missing: 6.1 (`lake_release_sub`,
  `lake_spill_sub` on the daily type). Hourly and source/sink: none;
  everything else is zeroed in prepare or recomputed within the day.
- A type function writing other than through `state`: no; every
  njit function writes `state.<field>[inode]` only, and the graph
  kernel dispatches each `inode` to its own type. The numpy hooks
  write own rows except `starfit_advance_type` (6.8).
- A unit branch inside njit code: none; `io_in_cfs` is read only in
  `initialize_starfit_type` and becomes the per-node factors.
- Slice 5 carry-overs: `tctx.itime_step` is `time.current_index`,
  so on a warm start's first step it is `_start_index >= 1`, and the
  daily first-run block is correctly skipped (which exposes 6.1).
  5.4: the STARFIT io fields use the placeholders, except 6.6.
  5.13: STARFIT declares no integer parameter (`grand_id` is not
  ported), so none can be left at 0. 5.9: not checked here what
  `splice_flow_graph` writes into STARFIT fields on non-STARFIT
  rows; `initialize_starfit_type` writes own rows only.
- Dropped from /code-review: `istarf_release` evaluates the same
  `np.sin` / `np.cos` three times each. It is pywatershed's form,
  LLVM can merge the calls, and no checklist item covers it.

## Notes for later slices

- Slice 15: 6.6 and 6.7 must be settled before a kernel units test
  for the STARFIT types can be written.
- Slice 17: `perfect_restart` compares only the end-of-run values,
  so a restart variable missing for one day (6.1) passes. The STARFIT
  restart test runs only `io_in_cfs=False` and does not cover
  `starfit_source_sink`.
- Slice 18: coverage (`coverage_serial.txt:98-100`) shows lines no
  test reaches: the `n_substeps < 2` raise (daily `:246`), the nan
  `Obs_MEANFLOW_CUMECS` raise (`:380`), the nan-storage spill branch
  (`:257->259`), and in the source/sink type both a source
  (`:90`) and a sink with storage below the minimum (`:94`). No test
  gives a nan STARFIT parameter (6.2), a zero divisor (6.3) or a nan
  minimum storage (6.4).
- Fix phase, runs that confirm before fixing: a daily graph warm
  start, looking at `lake_release` on the first resumed day (6.1);
  one reservoir with nan `NORhi_alpha`, port vs pywatershed (6.2);
  one with `Obs_MEANFLOW_CUMECS = 0.0` (6.3); a source/sink node with
  nan `source_sink_storage_min` (6.4); an hourly cfs graph writing
  `lake_storage_old` (6.5); `numba.njit(lambda a, b: max(a, b))(nan,
  0.0)` (6.2, 6.9).
