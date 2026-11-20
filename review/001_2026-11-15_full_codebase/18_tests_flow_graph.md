<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 18: the FlowGraph tests](#slice-18-the-flowgraph-tests)
  - [Findings](#findings)
    - [18.1 The storage diversion of `StarfitSourceSinkFlowNode` is tested with a request too small to change a storage: its sign and its effect are unchecked](#181-the-storage-diversion-of-starfitsourcesinkflownode-is-tested-with-a-request-too-small-to-change-a-storage-its-sign-and-its-effect-are-unchecked)
    - [18.2 The STARFIT types are compared with a reference only as isolated outlets in three of six configurations; the hourly type at 24 substeps and the daily type in cfs have no value check](#182-the-starfit-types-are-compared-with-a-reference-only-as-isolated-outlets-in-three-of-six-configurations-the-hourly-type-at-24-substeps-and-the-daily-type-in-cfs-have-no-value-check)
    - [18.3 The obsin type runs only with an inflow that is the same on every substep: the latch is untested, and pywatershed's test against PRMS answers has no counterpart](#183-the-obsin-type-runs-only-with-an-inflow-that-is-the-same-on-every-substep-the-latch-is-untested-and-pywatersheds-test-against-prms-answers-has-no-counterpart)
    - [18.4 Comparisons of one variable, of the last step, or of time means](#184-comparisons-of-one-variable-of-the-last-step-or-of-time-means)
    - [18.5 Assertions that cannot fail, or that check less than their text says](#185-assertions-that-cannot-fail-or-that-check-less-than-their-text-says)
    - [18.6 No test runs a FlowGraph under MPI](#186-no-test-runs-a-flowgraph-under-mpi)
    - [18.7 Guards and branches with no test, and one test that asserts the missing guard](#187-guards-and-branches-with-no-test-and-one-test-that-asserts-the-missing-guard)
    - [18.8 Configurations of the graph that no test runs](#188-configurations-of-the-graph-that-no-test-runs)
    - [18.9 Comments and docstrings that do not match the code, duplicated setup, side effects](#189-comments-and-docstrings-that-do-not-match-the-code-duplicated-setup-side-effects)
  - [Code findings of slices 5 and 6: why did no test catch this?](#code-findings-of-slices-5-and-6-why-did-no-test-catch-this)
  - [Checklist T, item by item](#checklist-t-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 18: the FlowGraph tests

- pws_phoenix commit: d531cff (branch `review_001`; read from
  `.git/HEAD` and `.git/refs/heads/review_001` as text). The earlier
  slices record that the commits since e33e217 are review commits.
  The 6 files total 1,791 lines, as in the plan.
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text. Checklist T does
  not require it; its autotests were read for comparison.
- Paths (under `tests/`): `test_flow_graph.py` (362),
  `test_flow_graph_build.py` (314),
  `test_mixed_channel_starfit.py` (290),
  `test_obsin_source_sink_nodes.py` (232),
  `test_starfit_daily_parity.py` (257),
  `test_starfit_flow_node.py` (336)
- Checklist: T. Covers code slices 5, 6. Reviewed 2026-11-19.
- Method: `/code-review high` on the 6 files (10 candidates, and 3 it
  cut at its cap of 10). Kept: all 13, as 18.1 (the two unrun
  branches of the storage diversion and the `at_min` assertion,
  merged), 18.2 (the daily type in a cfs graph), 18.3 (the obsin
  latch), 18.4 (the variables a cfs run reads), 18.5 (five
  assertions and the copied splice code), 18.7 (the untested raises)
  and 18.9 (the stale docstring of `_hand_built_big_sandy`, the
  repeated setup). Added by the read: the size of the
  request in 18.1, the table and the constant inflow of 18.2,
  pywatershed's obsin test in 18.3, `outflows` in 18.4, 18.6, 18.8.
  Then every file of the slice, whole, against checklist T.
  - Outside the slice, through CodeGraph and direct reads:
    `pws_phoenix/flow_graph.py:56-86`, `:257-347`, `:394-517`;
    `pws_phoenix/flow_graph_build.py:60-400`;
    `pws_phoenix/hydrology/obsin_flow_node.py:36-75`;
    `pws_phoenix/hydrology/source_sink_flow_node.py:45-93`;
    `pws_phoenix/hydrology/pass_through_flow_node.py:31-52`;
    `pws_phoenix/hydrology/starfit_flow_node.py:208-262`, `:296-352`,
    `:366-393`;
    `pws_phoenix/hydrology/starfit_source_sink_flow_node.py:62-149`;
    `pws_phoenix/hydrology/starfit_daily_flow_node.py:80-200`,
    `:240-250`; `tests/conftest.py:40-62`, `:144-216`;
    `tests/prms_oracle.toml`; `docs/PORTS.md:69-70`, `:255-280`,
    `:452-466`, `:576-584`.
  - pywatershed, `autotest/`: `test_obsin_flow_node.py:55-260`,
    `test_prms_channel_flow_graph.py:38-46`, and text scans of
    `test_starfit_flow_node.py`, `test_starfit_flow_graph.py`,
    `test_starfit_source_sink_flow_node.py`.
  - Before the review: `coverage_serial.txt` (whole); the findings
    files of slices 5 and 6 (whole); 15.9, 15.20, 16.14 and the notes
    for slice 18 at the end of the files of slices 15, 16 and 17.
- Run: the six files in env pwpx2 (pytest 9.1.1), with
  `--error-for-skips`: 55 passed in 34 s. No test was changed to see
  whether it then fails, and no data file was read: a finding that
  depends on a data value says "not read" and names the run that
  would settle it (the last section).

The six files have one must-fail test of a logged departure
(`test_budget_detects_last_substep_harvest`), hand-typed answers for
the splice and for the four branches of `SourceSinkFlowNode`, and a
parity test that drives pywatershed's own daily node. The findings
are ordered by consequence: first the node types whose values no
test checks in the way they are used (18.1-18.3); then partial
comparisons and assertions that cannot fail (18.4, 18.5); then what
the slice does not reach (18.6-18.8); then comments and style
(18.9).

## Findings

### 18.1 The storage diversion of `StarfitSourceSinkFlowNode` is tested with a request too small to change a storage: its sign and its effect are unchecked

- Where: `tests/test_starfit_flow_node.py:129-131`, `:202-212`,
  `:306-336`;
  `pws_phoenix/hydrology/starfit_source_sink_flow_node.py:88-103`
- Code:

      SOURCE_SINK_REQUEST = -28.0e-17
      ...
      params["source_sink_storage_min"] = ("nnodes", np.zeros(n_nodes))
      ...
      assert at_min.sum() < 5  # the drained case is the exception

  and the code under test:

      if source_sink_vol >= 0.0:
          storage += source_sink_vol
      elif storage < min_storage:
          source_sink_vol = 0.0
      else:
          if (storage + source_sink_vol) < min_storage:
              source_sink_vol = min_storage - storage
              storage = min_storage
          else:
              storage += source_sink_vol

- Consequence:
  - With one substep per day `m3ps_to_MCM` is 0.0864, so the request
    is -2.4e-17 MCM per day. Added to a storage of 0.25 MCM or more
    it gives back the same float (half the spacing of doubles at
    0.25 is 2.8e-17). The scenario therefore shows that the combined
    type gives the plain type's answers when the diversion does
    nothing; the module docstring says so ("validated against the
    SAME reference means"). `storage -= source_sink_vol` at `:103`,
    or `+ sink_source_sub` removed from the storage change
    (`pws_phoenix/hydrology/starfit_flow_node.py:222-226`), passes
    every test. The budget cannot see it either: an exchange of
    2.8e-16 against a tolerance of 1e-10.
  - `coverage_serial.txt`: `:90` (a source) and `:94` (a sink skipped
    below the minimum) never run. The minimum is 0 for every
    reservoir, and storage is floored at 0, so `storage <
    min_storage` cannot be true.
  - `test_applied_diversion` reads the last day's in-memory arrays
    only. `at_min.sum() < 5` has no lower bound: when no reservoir
    is at zero on the last day, `applied[at_min]` is empty and
    `assert_array_equal(applied[at_min], 0.0)` passes on an empty
    array. The docstring says "one reservoir drains to empty storage
    on the final day"; the test does not assert that one does. Not
    read: the count.
  - pywatershed's autotest uses the same request and the same
    minimum (`autotest/test_starfit_source_sink_flow_node.py:75`,
    `:148`), so this is its limit, carried over.
    `tests/test_obsin_source_sink_nodes.py` has one hand-computed
    node per branch for `SourceSinkFlowNode`; the storage type has
    nothing like it.
- Source: /code-review (two candidates, merged); the read (the size
  of the request)
- Outcome: open

### 18.2 The STARFIT types are compared with a reference only as isolated outlets in three of six configurations; the hourly type at 24 substeps and the daily type in cfs have no value check

- Where: `tests/test_starfit_flow_node.py:153-158`;
  `tests/test_starfit_daily_parity.py:45-46`, `:107-113`, `:126-127`,
  `:156-161`; `tests/test_mixed_channel_starfit.py:164-171`,
  `:261-290`
- Code:

      graph_class = make_flow_graph(
          (node_class,), ..., n_substeps=1, io_in_cfs=io_in_cfs,
      )
      ...
      for ss in range(N_SUBSTEPS):
          node.calculate_subtimestep(ss, inflow_vals[istep, inode], 0.0)

- Consequence: what each configuration is compared with:

  | type | substeps | units | compared with |
  |---|---|---|---|
  | starfit | 1 | cms, cfs | reference time means, 1e-7, 115 reservoirs |
  | starfit_source_sink | 1 | cms, cfs | the same means (18.1) |
  | starfit | 24 | cfs | nothing: the mixed graph's identities and budget |
  | starfit_source_sink | 24 | | never run |
  | starfit_daily | 24 | cms | pywatershed's node, 1e-10, 15 reservoirs, 365 days |
  | starfit_daily | 24 | cfs | nothing: the mixed graph's identities; its budget is an identity |

  - The hourly type at 24 substeps is how it runs in a channel graph.
    There `m3ps_to_MCM` is 0.0036, the running means divide by
    `istep + 1` over 24 substeps, and
    `lake_storage_change_accum / nsub` has 24 terms
    (`pws_phoenix/hydrology/starfit_flow_node.py:270-301`). With one
    substep each of these is the trivial case. The mixed test checks
    three identities at the reservoir and the budget;
    its docstring says pywatershed has no reference here either.
    pywatershed's hourly node can be driven with `nhrs_substep=1`
    the way `tests/test_starfit_daily_parity.py` drives
    `compute_daily=True`; no test does it.
  - The daily type multiplies by `io_to_cms` / `cms_to_io` in ten
    places
    (`pws_phoenix/hydrology/starfit_daily_flow_node.py:88-91`,
    `:107-110`, `:135-141`, `:190-195`). The parity test is cms on both
    sides (`tests/test_starfit_daily_parity.py:110`, `:160`), where
    every factor is 1.0. A wrong factor in a
    cfs graph changes storage, release and spill, and passes.
  - The parity test gives all 24 substeps of a day one inflow value.
    Then the first-day seed `lake_inflow = lake_inflow_accum` (the
    first substep, `:117`) equals the day mean
    `lake_inflow_accum / (istep + 1)` (`:132`): seeding the first day
    from the wrong one passes. Below a channel the inflow differs
    by substep; that case is the mixed test alone.
  - 15 of the 115 reservoirs, 1995 only. Not read: whether any of
    the 15 spills, or releases its whole storage, in those 365 days
    (`coverage_serial.txt` cannot attribute the lines: the mixed and
    restart tests run them too).
- Source: /code-review (the daily type in cfs); the read
- Outcome: open

### 18.3 The obsin type runs only with an inflow that is the same on every substep: the latch is untested, and pywatershed's test against PRMS answers has no counterpart

- Where: `tests/test_obsin_source_sink_nodes.py:119-129`;
  `tests/test_flow_graph.py:12-18`, `:124-129`, `:200-202`;
  `pws_phoenix/hydrology/obsin_flow_node.py:55-60`
- Code:

      inflow = np.full((N_TIME, 2), 10.0)
      ...
      # headwater insertion: the new node feeds nhm_seg 1829's
      # node but intercepts nothing (zero inflow) -- with
      # negative obs it emits exactly zero

  and the code under test:

      if state.node_outflow_substep[inode] >= 0.0:
          state.obsin_sink_source_sum[inode] += (
              state.node_outflow_substep[inode] - inflow
          )
      else:
          state.node_outflow_substep[inode] = inflow

- Consequence:
  - In both tests the inflow of an obsin node is one value all day
    (10 cfs of lateral inflow at an isolated outlet; zero at the drb
    headwater). A negative observation holds the FIRST substep's
    inflow for the day; with a constant inflow that equals passing
    each substep through. Replacing the latch by a pass-through per
    substep (the change `CLAUDE.md` says not to make) passes every
    test, and so does breaking the nonzero `node_sink_source` the
    latch gives under a varying inflow. The module docstring of
    `tests/test_flow_graph.py` records that an intercepting obsin was
    tried, differed from the pass-through answers, and was replaced
    by the headwater: what was found did not become a test.
  - pywatershed tests the type as it is used: gage observations
    replace the flow below gaged segments of drb_2yr, and four graph
    variables are compared with PRMS answers of `nhm_obsin.control`
    (`autotest/test_obsin_flow_node.py:62-260`). The answers and the
    observations are in the test data (`drb_2yr/output_nhm_obsin`,
    `drb_2yr/drb_2yr_gage_poi_obs.nc`); `tests/prms_oracle.toml`
    pins five drb answer sets and not this one.
    `docs/PORTS.md:581-582`: "NOT yet wired: sf_data (obsin)".
  - So no test has an observation replacing an upstream inflow, an
    observation series with gaps, or a nan observation (5.14).
  - Answer to the note of slice 17: the day-by-day switch of
    `obs` (`:122-129`) does show that the latch and the sum do not
    carry into the next day (10 cfs on day 2 after 25 on day 1).
- Source: /code-review (the latch); the read (pywatershed's test)
- Outcome: open

### 18.4 Comparisons of one variable, of the last step, or of time means

- Where: `tests/test_flow_graph.py:216`, `:268-273`, `:282-294`;
  `tests/test_starfit_flow_node.py:73`, `:266-273`, `:296-304`,
  `:315-336`; `pws_phoenix/flow_graph.py:342-345`
- Code:

      "output_var_names": ["node_outflows"],
      ...
      OUTPUT_VARS = ["lake_storage", "lake_release", "lake_spill"]
      ...
      output_ds[var].values.mean(axis=0),

  and the kernel:

      if to_graph_index[jj] < 0:
          state.outflows[jj] = state.node_outflows[jj]
      else:
          state.outflows[jj] = 0.0

- Consequence:
  1. The channel graph is compared with PRMS on `node_outflows`
     alone. pywatershed's channel-graph autotest compares four
     (`autotest/test_prms_channel_flow_graph.py:38-43`): `outflows`,
     `node_upstream_inflows`, `node_outflows`,
     `node_storage_changes`, against `channel_outflow_vol`,
     `seg_upstream_inflow`, `seg_outflow`, `seg_stor_change`. The
     answer files are in `drb_2yr/output`, and
     `tests/test_prms_channel.py:91-95` reads them for the array
     channel.
  2. `outflows`, the outlet-masked flow and the graph's own named
     output, is read by no test (text scan of `"outflows"` under
     `tests/`). The mask inverted passes.
  3. On channel rows `node_upstream_inflows` and
     `node_storage_changes` are tied to `node_outflows` through the
     budget only, and that is one equation for two values.
  4. The cfs scenarios of the STARFIT matrix read three variables,
     as time means. `lake_storage_old`, `lake_inflow`,
     `lake_outflow`, `lake_storage_change` and `node_storages` are
     not read: finding 6.5 (a written `lake_storage_old` 35.31 times
     too large) passes. The reference holds means only, so the means
     are pywatershed's limit; a per-day comparison of the cfs run
     with the cms run times `cms_to_cfs` needs no reference.
  5. `test_inserted_node_is_transparent`,
     `test_outflow_is_release_plus_spill` and
     `test_applied_diversion` read the in-memory arrays after the
     last step: the last day of the run.
- Source: /code-review (item 4); the read
- Outcome: open

### 18.5 Assertions that cannot fail, or that check less than their text says

- Where: `tests/test_mixed_channel_starfit.py:252`;
  `tests/test_flow_graph_build.py:136-138`, `:313`;
  `tests/test_starfit_flow_node.py:288-292`;
  `tests/test_flow_graph.py:283-285`;
  `tests/test_starfit_daily_parity.py:246-257`
- Code:

      assert unaffected.sum() < n_seg  # the walk found something

      first = nodes[0].node_type.type_name
      assert kit.graph_class.node_type_names[1] == first

      assert int(dis["nnodes"].sel(nnodes=999)) == 999

      assert proc[f"{prefix}_residual"].attrs["units"] in ("cms", "cfs")

      if not graph_run["insert"]:
          assert proc["node_outflows"].values.shape == (graph_run["n_seg"],)
          return

- Consequence:
  1. The walk sets `unaffected[wh_44426] = False` on its first pass,
     so the sum is always below `n_seg`. The docstring says segment
     44409 "is NOT ignored", and the module docstring that PT1's
     transparency "rides free" on this comparison; nothing asserts
     `unaffected[wh_44409]`, and the test never computes that index.
     If 44409 is downstream of 44426, PT1 is compared with nothing.
     Not read: the ucb_2yr topology.
  2. "types numbered by first appearance": `nodes[0]` is a
     pass-through in both parametrizations (`[pt_above, sf,
     pt_below]` and its reverse), so code 1 is `pass_through` either
     way. A fixed or alphabetical numbering passes. The ucb pin has
     STARFIT first, and compares through `code("starfit")`, so it
     pins no number either.
  3. `.sel(nnodes=999)` returns the coordinate value 999 or raises
     `KeyError`. It does not show that 999 is the STARFIT row.
  4. The units of the budget residual are not compared with the
     scenario: the fixture knows `io_in_cfs` and does not return it.
     The next assertion compares a one-term budget sum with its one
     term.
  5. The pure-channel case of `test_inserted_node_is_transparent`
     asserts the size the test itself gave the grid. This is the
     no-skips rule's "asserts something true of that case", and the
     something is trivially true.
  6. `test_outflow_constant_within_day_lags_inflow` asserts neither
     a constant outflow within the day nor the lag: its body is
     `outflow == release + spill`, and it requests `pyws_results`
     without using it. The lag is checked by the other test, through
     parity.
  7. `tests/test_flow_graph.py:89-91`: "The pass-through splice built
     here is pinned equal to the helper's". The pin
     (`tests/test_flow_graph_build.py:221-234`) compares the helper
     with a copy of that code, not with `build_graph_run`; an edit to
     one copy leaves the pin green.
  8. The structure checks use `assert_allclose`, where nan on both
     sides passes; pywatershed's use `(abs(a - b) < 1e-12).all()` and
     `==` (`autotest/test_starfit_flow_graph.py:310-320`), where nan
     fails. Here a nan in a budget term stops the run in error mode,
     so the gap is the variables outside the budget
     (`lake_release`, `lake_spill`).
- Source: /code-review (items 1, 2, 3, 4, 6, 7); the read
- Outcome: open

### 18.6 No test runs a FlowGraph under MPI

- Where: no file (`grep -l -i "flow_graph\|FlowNode" tests/*_mpi.py`
  finds none); `pws_phoenix/flow_graph.py:49-51`
- Code:

      The graph is a Process on its own ``nnodes`` grid: serial in serial
      runs, REPLICATED + Map-fed under MPI (the Step B pattern) -- no
      FlowGraph-specific MPI code.

- Consequence: "no FlowGraph-specific MPI code" is true of the graph
  and not of the path it runs on. Under `ModelMPI` a graph is a
  process on a replicated serial grid: three inputs fed by `MapMPI`
  (a partial product per rank, then `Allreduce`), a required
  `lake_storage_sub` supplied at full extent on a serial grid,
  `check_initial_state` on every rank, the rank-0 zarr `Output`, a
  restart. `tests/test_two_grid_mpi.py` runs that path for the toy
  `Lower`; nothing runs it for a graph, with any node type. Finding
  5.6 (nothing stops a graph on the distributed grid) has no test
  for the same reason. In this slice no test runs a `Map` at all:
  the lateral inflows are aggregated beforehand by a dense product
  (15.20 item 1). The one graph fed by Maps is serial, in
  `tests/test_prms_translate_flow_graph.py` (slice 22).
- Source: checklist T (a serial test with no MPI counterpart)
- Outcome: open

### 18.7 Guards and branches with no test, and one test that asserts the missing guard

- Where: `pws_phoenix/flow_graph.py:395`, `:419`, `:426`;
  `pws_phoenix/hydrology/starfit_daily_flow_node.py:246`;
  `pws_phoenix/hydrology/starfit_flow_node.py:257->259`, `:380`;
  `pws_phoenix/flow_graph_build.py:375->377`;
  `tests/test_flow_graph_build.py:134-135`;
  `tests/test_flow_graph.py:58-65`, `:313-324`
- Code:

      raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
      raise ValueError(f"duplicate node type names: {type_names}")
      raise ValueError(f"field '{name}' is declared differently by ...

      # a declared STARFIT parameter not supplied stays nan
      assert np.isnan(kit.parameters["NORhi_mu"].values).all()

- Consequence:
  - `make_flow_graph` has four raises and one test, of the missing
    contract attribute. `:426` guards the rule that a field shared by
    two types is the same `DataArrayMeta` object; without the check
    the second declaration is dropped without a message.
  - No test shows the daily type refusing `n_substeps < 2`
    (`:246`), a STARFIT type refusing a nan `Obs_MEANFLOW_CUMECS`
    (`:380`), or a nan storage reaching the spill line
    (`:257->259`).
  - `:375->377`: no test splices two nodes that supply the same
    initial-state name (two reservoirs), so the merge into one
    `(nnodes,)` array is run for one row only.
  - `test_series_chain_any_collated_order` gives the STARFIT node
    one of its twenty parameters and asserts that another is nan on
    every row. That pins as wanted the behaviour findings 5.3 and
    6.2 describe: a parameter the caller did not give is nan and
    nothing raises. A fix of those findings must change this
    assertion.
  - `test_incomplete_node_type_raises` reads no data and sits under
    the module's data `skipif`; in the fast CI job (15.12) it is
    skipped with the rest of the file.
  - Guards that the splice does have are all shown to raise
    (`test_errors`, `test_pad_weights`): eight of eight.
- Source: /code-review; coverage
- Outcome: open

### 18.8 Configurations of the graph that no test runs

- Where: `tests/test_flow_graph.py:110-112`, `:138-140`;
  `tests/test_obsin_source_sink_nodes.py:47-51`, `:188-192`; text
  scans of `n_substeps=` and `io_in_cfs=` under `tests/`
- Code:

      graph_class = make_flow_graph(
          (PRMSChannelFlowNode, insert_class), class_name=class_name
      )
      ...
      source_sink = np.tile([3.0, -2.0, -8.0, -3.0], (N_TIME, 1))

- Consequence:
  - The channel type with a substep count other than 24: never
    (5.1). Every graph with channel rows uses the default, or 24.
  - The channel, pass-through, obsin and source/sink types in a cms
    graph: never run (5.4). `io_in_cfs=False` appears with STARFIT
    types alone, here and in `tests/test_restart_flow_graph.py`.
  - A run with more than three node types, or with two STARFIT types
    in one graph (6.8): never. `tests/test_units.py` composes all
    seven types and does not run the class.
  - A source/sink node that applies a nonzero request below a
    channel: never. The drb insertion is neutral (zero request, zero
    `flow_min`), and the four hand-computed nodes are isolated
    outlets with a constant lateral inflow, so each stays in one
    branch for the whole run and its request never changes in time.
    A sink limited on some substeps of a day and not on others does
    not occur. The must-fail test shows the mean harvest matters
    below a channel; its value is checked by the budget, which for
    this type is an identity in the same sums.
  - A `node_type` array with a code outside the composition (5.5),
    two new nodes that flow to one `NewNode` (5.7), a `dis_seg`
    variable on other dims (5.13; the fixture's `hru_segment` covers
    `channel_params` only), a `time_context` name other than
    `"epiweek"` (5.10): no test.
- Source: checklist T; the notes of slices 5 and 6
- Outcome: open

### 18.9 Comments and docstrings that do not match the code, duplicated setup, side effects

- Where and code:
  1. `tests/test_mixed_channel_starfit.py:35-41`, `:150`: the NOR
     midpoint "is replicated HERE, test-side", and the code calls
     `nor_midpoint_initial_storage` from
     `pws_phoenix/hydrology/starfit_flow_node.py:436`. The starting
     storage of the one reservoir comes from the package, and that
     function has no test of a known value (text scan: this is its
     only use under `tests/`).
  2. `tests/test_flow_graph_build.py:237-240`: `_hand_built_big_sandy`
     is described as the code of
     `tests/test_mixed_channel_starfit.py` `graph_run`, which now
     calls `splice_flow_graph`. The module docstring (`:9-12`) says
     the pins exist "so the existing tests can switch to the
     helper": one switched, `tests/test_flow_graph.py` did not
     (18.5 item 7).
  3. `tests/test_flow_graph.py:26-29`: "identical math and float
     order to a per-step Map apply" (15.20 item 1: a dense product
     and `Map.apply` sum in different orders).
  4. `tests/test_flow_graph.py:68-71`: two comments run together
     ("built once times are known values are duck-typed").
  5. `tests/test_flow_graph.py:31-32`, and the same sentence in three
     other module docstrings: "skips with a reason if absent". Under
     `tests/run_tests.sh` a skip is an error; the sentence describes
     the fast CI job (15.12).
  6. `tests/test_flow_graph.py:98-102`: in the pure-channel scenario
     `pad` returns the array it was given, so the Model receives the
     arrays of the session-scoped `pyws_domain` cache and sets them
     read-only for the rest of the session. The class attributes
     `obs`, `flow_min` and `source_sink` of
     `tests/test_obsin_source_sink_nodes.py` become Model buffers the
     same way (`np.asarray` of a float64 array is the array). No
     later test writes to them today. By reading; not run.
  7. `tests/test_starfit_daily_parity.py:57`:
     `sys.path.insert(0, str(PYWS_ROOT))` is the first statement of the
     `try`, so it stays whether or not the imports succeed, and puts
     the pywatershed clone ahead of the environment for the rest of
     the session (17.11 has the same in
     the restart files).
  8. Repeated setup: the isolated-outlet `Discretization` in three
     files of this slice and in `tests/test_restart_flow_graph.py`;
     `vol_input` in two; the STARFIT parameter preparation
     (`Obs_MEANFLOW_CUMECS` filled from `inflow_mean`) in two, and a
     third form in the mixed test; `NODE_INPUT_NAMES` in two.
     `tests/conftest.py` holds the shared pieces of the channel
     graph and none of these.
- Consequence: item 1 is an expected value taken from the code under
  test; it matters little while the reservoir of the mixed graph has
  no value check (18.2). The rest cost a reader time.
- Source: /code-review (items 2, 8); the read
- Outcome: open

## Code findings of slices 5 and 6: why did no test catch this?

"No test" means no test in the six files, by reading and the text
scans named above.

- 5.1: every graph with channel rows has 24 substeps (18.8).
- 5.2: both fixtures supply `mann_n` and `x_coef`;
  `test_pure_channel` asserts the parameter set is exactly those two.
- 5.3, 6.2, 6.4: no test gives a nan parameter or a nan request, and
  one test asserts the nan fill (18.7).
- 5.4, 5.11: no channel, pass-through, obsin or source/sink row runs
  in a cms graph, and no test reads a description (18.8).
- 5.5: six files build `node_type` by hand, each with valid codes;
  `test_node_type_introspection` reads code 0 only.
- 5.6: no FlowGraph under MPI (18.6).
- 5.7: `test_errors` has two nodes on one `nhm_seg`, not two on one
  `NewNode`.
- 5.8, 5.12, 6.10: nothing a test of values can see.
- 5.9: no test reads a channel field on an inserted row.
- 5.10: the three declarations are correct, and no test composes a
  type with another name.
- 5.13: 18.8, last item.
- 5.14: no nan observation (18.3).
- 5.15, 6.9: comments.
- 6.1: slice 17 (17.1). In this slice every run is a cold start.
- 6.3: no zero `Obs_MEANFLOW_CUMECS`, zero capacity or equal NOR
  bounds in the 115 reservoirs (the runs complete).
- 6.5: the cfs runs read three variables (18.4 item 4).
- 6.6, 6.7: no kernel units test for the node types (15.8, 15.9);
  `tests/test_units.py` checks that a unit is in the vocabulary, not
  that it is the right one.
- 6.8: no run with two STARFIT types (18.8).

## Checklist T, item by item

- A raise or a branch that no test reaches: 18.7, and `:90`, `:94`
  of the storage diversion (18.1). The files of code slices 5 and 6
  are otherwise covered at 94-100%, and that figure is what this
  slice qualifies: 18.1-18.3 are lines that run and whose results
  are not compared with anything.
- A test that cannot fail: 18.5 items 1-5. Expected values from the
  code under test: 18.9 item 1. The splice pins compare with a
  second hand-written form of the same steps
  (`tests/test_flow_graph_build.py:221-259`); the synthetic cases
  have typed arrays, which is the independent check.
- A tolerance looser than the standard in `docs/PORTS.md`: none.
  1e-10 against `seg_outflow`, 1e-7 against the STARFIT means, 1e-10
  for the daily parity, 1e-12 to 1e-15 for the identities, 1e-14 for
  the hand-computed nodes. The budget residuals are pinned at an
  absolute 1e-8 and 1e-9 (`docs/PORTS.md:277`), tighter for a large
  flow than the error mode's relative 1e-10.
- A comparison where nan on both sides passes: 18.5 item 8. Not
  read: whether the reference means or pywatershed's daily series
  hold a nan; no test asserts that a compared array is finite.
- A comparison of only some output variables, time steps or
  elements: 18.4; 15 of 115 reservoirs and one year in the parity
  test (18.2); in the mixed graph every segment from 44426 down is
  left out, as in pywatershed.
- A serial test with no MPI counterpart where the MPI code differs:
  18.6.
- A guard with no test showing that it raises: 18.7. Shown to
  raise: the eight guards of the splice, the missing contract
  attribute, and the budget on the reverted departure, with the
  position of the node in the warning (`pytest.warns` inside
  `pytest.raises` checks the warning in pytest 9, also when the
  block raises).

## Notes for later slices

- Answer to slice 16 (16.14): the mixed test does not pin an
  epiweek; its reservoir has no reference. The week is held
  indirectly, by the STARFIT reference means over 1995-2001 (1997
  has an epiweek 53, so the `min(epiweek, 52)` cap runs; by the
  calendar, not run) and by
  the daily parity over 1995.
- Answer to slice 17 (17.9): obsin and source/sink are shown to
  carry nothing across a day, by their values over four days (18.3,
  last item; a sum not zeroed would double the expected mean on
  day 2).
- Slice 19: does `tests/test_prms_channel.py` compare the array
  channel on all of `seg_upstream_inflow`, `seg_stor_change`,
  `channel_outflow_vol` (18.4 item 1)?
- Slice 22: `tests/test_prms_translate_flow_graph.py` is the one
  graph fed by Maps (18.6), and the second copy of the "unaffected
  segments" walk (18.5 item 1): does it assert that the walk leaves
  the pass-through's segment in?
- Fix phase, runs that settle what this slice did not read:
  - In ucb_2yr, walk `tosegment` down from 44426 and see whether
    44409 is on the path (18.5 item 1).
  - In `test_applied_diversion`, print `at_min.sum()` for the two
    source/sink scenarios (18.1).
  - `np.isfinite(...).all()` on the reference means and on
    pywatershed's daily series (checklist item on nan).
  - Change `storage += source_sink_vol` to `-=` at
    `starfit_source_sink_flow_node.py:103` and run
    `tests/test_starfit_flow_node.py`; expect it to pass (18.1).
  - Replace the obsin latch by a pass-through per substep and run
    `tests/test_obsin_source_sink_nodes.py` and
    `tests/test_flow_graph.py`; expect both to pass (18.3).
  - Run the cfs scenario of `tests/test_starfit_flow_node.py` with
    every STARFIT variable in `output_var_names` and compare each
    day with the cms run times the unit factor; expect
    `lake_storage_old` to differ by 35.31 (18.4 item 4, 6.5).
  - Drive pywatershed's hourly node with `nhrs_substep=1` and a
    cfs graph, as the daily parity test does, and compare with the
    hourly type at 24 substeps (18.2).
