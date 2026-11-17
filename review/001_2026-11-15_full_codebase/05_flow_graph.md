<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Slice 5: the FlowGraph, the splicer, and four node types](#slice-5-the-flowgraph-the-splicer-and-four-node-types)
  - [Findings](#findings)
    - [5.1 A graph with channel rows and `n_substeps` other than 24 loses the inflow of most segments, with a closed budget](#51-a-graph-with-channel-rows-and-n_substeps-other-than-24-loses-the-inflow-of-most-segments-with-a-closed-budget)
    - [5.2 `splice_flow_graph` fills a channel parameter it cannot find with nan instead of raising](#52-splice_flow_graph-fills-a-channel-parameter-it-cannot-find-with-nan-instead-of-raising)
    - [5.3 `SourceSinkFlowNode` accepts a nan `flow_min` and a nan request; pywatershed raises on both, and the "verbatim" comment hides the difference](#53-sourcesinkflownode-accepts-a-nan-flow_min-and-a-nan-request-pywatershed-raises-on-both-and-the-verbatim-comment-hides-the-difference)
    - [5.4 The channel type's flow fields and the pass-through accumulator declare `units="cfs"`, not the `io flow units` placeholder](#54-the-channel-types-flow-fields-and-the-pass-through-accumulator-declare-unitscfs-not-the-io-flow-units-placeholder)
    - [5.5 The codes in a supplied `node_type` array are never compared with the composition](#55-the-codes-in-a-supplied-node_type-array-are-never-compared-with-the-composition)
    - [5.6 Nothing stops a FlowGraph on the distributed grid; the kernel would index one rank's block with global indices, in njit code](#56-nothing-stops-a-flowgraph-on-the-distributed-grid-the-kernel-would-index-one-ranks-block-with-global-indices-in-njit-code)
    - [5.7 Two new nodes that flow to the same `NewNode` are accepted and chained in collated order; pywatershed refuses](#57-two-new-nodes-that-flow-to-the-same-newnode-are-accepted-and-chained-in-collated-order-pywatershed-refuses)
    - [5.8 Each step looks arrays up in the dataset, and the class holds every array reference twice](#58-each-step-looks-arrays-up-in-the-dataset-and-the-class-holds-every-array-reference-twice)
    - [5.9 A row where a field does not apply holds 0.0 (or a default coefficient), not nan](#59-a-row-where-a-field-does-not-apply-holds-00-or-a-default-coefficient-not-nan)
    - [5.10 A name in a node type's `time_context` is never compared with the fields the graph serves](#510-a-name-in-a-node-types-time_context-is-never-compared-with-the-fields-the-graph-serves)
    - [5.11 Composition resolves the placeholder in `units` and leaves it in `description`](#511-composition-resolves-the-placeholder-in-units-and-leaves-it-in-description)
    - [5.12 Every composition compiles its own kernel](#512-every-composition-compiles-its-own-kernel)
    - [5.13 `splice_flow_graph`: smaller cases that pass without a message](#513-splice_flow_graph-smaller-cases-that-pass-without-a-message)
    - [5.14 A nan observation makes an obsin node pass its inflow through; the port keeps this and does not say so](#514-a-nan-observation-makes-an-obsin-node-pass-its-inflow-through-the-port-keeps-this-and-does-not-say-so)
    - [5.15 Docstrings and comments that no longer match the code](#515-docstrings-and-comments-that-no-longer-match-the-code)
  - [Checklist C, item by item](#checklist-c-item-by-item)
  - [Notes for later slices](#notes-for-later-slices)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Slice 5: the FlowGraph, the splicer, and four node types

- pws_phoenix commit: 069cf4f (branch `review_001`; the commits since
  e33e217 are review commits).
- pywatershed commit: ba3974b (`develop`), confirmed by reading
  `.git/HEAD` and `.git/refs/heads/develop` as text.
- Paths: `pws_phoenix/{flow_graph,flow_graph_build}.py`,
  `pws_phoenix/hydrology/{obsin,pass_through,prms_channel,source_sink}_flow_node.py`
- Checklist: C (checklist B, plus). Reviewed 2026-11-17.
- Method: `/code-review high` on the six files, then a read of each
  file against checklist C beside its pywatershed source
  (`pywatershed/base/flow_graph.py`,
  `pywatershed/hydrology/prms_channel_flow_graph.py`,
  `pywatershed/hydrology/{obsin,pass_through,source_sink}_flow_node.py`),
  with the code the slice calls (`muskingum_mann_coefficients`,
  `Discretization.topological_order`, `Process.__init__`,
  `Model._add_process_fields`) and its callers
  (`prms_translate.insert_flow_graph`, the tests that set
  `n_substeps` or `io_in_cfs`). CodeGraph: the `codegraph_explore`
  tool failed twice with an internal error; `codegraph explore` from
  the shell worked and was used once, then text scans (grep). The
  session executed no project code. A finding marked "not run" rests
  on reading and needs a short run to confirm before it is fixed.

The numerics of the kernel and of the four node types agree with
pywatershed line for line, except where a finding or the checklist
section says otherwise. The findings that give a wrong answer without
a message come first (5.1-5.6); none of them can be seen by the mass
budget, which is an identity for every node type of this slice (see
the checklist section).

## Findings

### 5.1 A graph with channel rows and `n_substeps` other than 24 loses the inflow of most segments, with a closed budget

- Where: `pws_phoenix/hydrology/prms_channel_flow_node.py:222-223`,
  `:53-54`; `pws_phoenix/flow_graph_build.py:224`, `:264`
- Code:

      def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
          # n_substeps unused: the muskingum ts/tsi ladder assumes 24.
      ...
      remainder = (istep + 1) % state.tsi[inode]
      if remainder == 0:

- Consequence: `splice_flow_graph(..., n_substeps=1)` always composes
  the channel type, and `make_flow_graph`'s docstring offers 1 ("a
  STARFIT-only graph uses 1"). With one substep `istep` is only 0, so
  a segment with `tsi >= 2` never routes: `outflow_ts` keeps its
  starting value (0.0), and `inflow_ts` is zeroed again by `_prepare`
  the next day. The inflow of that segment is gone.
  `node_storage_changes = seg_inflow - seg_outflow`, so
  `control["budget"] = "error"` reports nothing. With 12 or 48
  substeps the hour-based `ts` / `tsi` are applied to substeps that
  are not hours. pywatershed cannot reach this: `FlowGraph.calculate`
  is always called with its default `n_substeps=24`, and the channel
  node divides by the literal `24.0`. `StarfitDailyFlowNode` raises
  on a substep count it does not support
  (`pws_phoenix/hydrology/starfit_daily_flow_node.py:245`); the
  channel type does not. No test composes the channel type with a
  substep count other than 24 (text scan of `n_substeps=` under
  `tests/`). Not run.
- Source: /code-review
- Outcome: open

### 5.2 `splice_flow_graph` fills a channel parameter it cannot find with nan instead of raising

- Where: `pws_phoenix/flow_graph_build.py:329-341`
- Code:

      if name in channel_params and channel_params[name].dims == (seg_dim,):
          params[name] = (...)
      else:
          fill = np.nan if np.issubdtype(dtype, np.floating) else 0
          params[name] = (
              "nnodes",
              np.full(n_nodes, fill, dtype=dtype),

- Consequence: the `else` branch is meant for the parameters of the
  NEW node types. It also takes a channel parameter that is absent
  from `channel_params`, or present on another dim. The Model then
  finds the parameter supplied and does not raise its "declares
  parameter ... but it is in neither" error. With `mann_n` all nan,
  `velocity > 0.0` is False for every segment
  (`pws_phoenix/hydrology/prms_channel.py:172-173`), every Kcoef is
  24 h, and the graph routes with finite, wrong coefficients. A
  missing `x_coef` gives nan `c0` / `c1` / `c2` and nan outflows,
  which is at least visible. The same holds for `seg_length`,
  `seg_slope`, `seg_depth` when they are on neither `dis_seg` nor
  `channel_params`. pywatershed's node maker raises a `KeyError`
  here (`self._discretization.parameters[param]`,
  `prms_channel_flow_graph.py:216`). Not run.
- Source: /code-review
- Outcome: open

### 5.3 `SourceSinkFlowNode` accepts a nan `flow_min` and a nan request; pywatershed raises on both, and the "verbatim" comment hides the difference

- Where: `pws_phoenix/hydrology/source_sink_flow_node.py:56`,
  `:64-78`, `:142-147`
- Code:

      # pywatershed calculate_subtimestep, verbatim branch structure.
      ...
      if source_sink >= 0.0:
          outflow = inflow + source_sink
      elif inflow < min_flow:
          outflow = inflow
          source_sink = 0.0
      else:
          # (source_sink < 0) and (inflow >= min_flow)
          if (inflow + source_sink) < min_flow:

  pywatershed (`source_sink_flow_node.py:100-115`) has no `else`:

      if source_sink >= zero:
      elif (source_sink < zero) and (inflow < min_flow):
      elif (source_sink < zero) and (inflow >= min_flow):

- Consequence:
  - `splice_flow_graph` leaves a parameter the caller did not give as
    nan (`NewNode(SourceSinkFlowNode, flows_to=..., node_id=...)`
    with no `parameters`). For a sink, `inflow < nan` is False and
    `(inflow + source_sink) < nan` is False, so the sink is applied
    with no floor: a sink larger than the inflow sends a negative
    flow into the muskingum rows downstream. `initialize_type` checks
    nothing in its own rows; `CLAUDE.md` "FlowGraph" says "a node
    type raises on bad values in its own rows".
  - A nan in `node_source_sink` with `inflow < flow_min` is replaced
    by a request of zero, without a message. With
    `inflow >= flow_min` the outflow is nan.
  - In pywatershed all three cases match no branch and fail with an
    `UnboundLocalError` on `outflow`: an accident, but a stop. The
    port's `else` turned that stop into an answer.
  - The budget closes in every case (`node_sink_source` is the
    applied amount). Not run.
- Source: /code-review (the nan `flow_min`); checklist B (order and
  structure against pywatershed)
- Outcome: open

### 5.4 The channel type's flow fields and the pass-through accumulator declare `units="cfs"`, not the `io flow units` placeholder

- Where: `pws_phoenix/hydrology/prms_channel_flow_node.py:177-213`
  (`seg_inflow`, `seg_outflow`, `inflow_ts`, `outflow_ts`,
  `inflow_ts_prev`); `pws_phoenix/hydrology/pass_through_flow_node.py:62-69`
  (`accum_inflow`)
- Code:

      "seg_inflow": DataArrayMeta(
          kind="variable",
          dims=("space",),
          dtype=np.float64,
          units="cfs",

- Consequence: both types say they ignore `io_in_cfs` because they
  are linear in flow. In a graph composed with `io_in_cfs=False`
  these arrays hold cms and are declared cfs, beside `node_outflows`
  (the same numbers, declared cms). With `control["units"] = "check"`
  a supplied `seg_inflow` labelled cms raises; with `"convert"` it is
  multiplied by 35.3 on its way into a cms graph. `seg_inflow` and
  `outflow_ts` are restart variables, so restart state is judged the
  same way. `CLAUDE.md` "Units mode": "A FlowGraph field in the
  graph's io units declares a placeholder".
  `tests/test_units.py:157-187` composes every node type in both
  units but asserts only that no placeholder is left and three
  fields of other types.
- Source: /code-review
- Outcome: open

### 5.5 The codes in a supplied `node_type` array are never compared with the composition

- Where: `pws_phoenix/flow_graph.py:476-483`, `:313-318`
- Code:

      self._obj["node_type"].attrs["node_type_names"] = list(type_names)
      ...
      for fn in literal_unroll(substep_fns):
          if ii == code:
              fn(istep, inode, state, tctx, n_substeps)
          ii += 1

- Consequence: `splice_flow_graph` writes valid codes. Six test
  and benchmark files build `node_type` by hand (text scan), which is
  the documented way to compose a graph without a channel. A code
  outside `range(len(node_types))` matches no branch: that row's
  `node_outflow_substep` is never written, and its fill value is
  added to the downstream node every substep. A code that is valid
  but belongs to another order of the same types runs another type's
  physics on the row. `initialize()` then writes this composition's
  names over the array's `node_type_names` attr, whatever the array
  was built for. Not run.
- Source: /code-review
- Outcome: open

### 5.6 Nothing stops a FlowGraph on the distributed grid; the kernel would index one rank's block with global indices, in njit code

- Where: `pws_phoenix/flow_graph.py:311-324`; no check in
  `pws_phoenix/model.py` (text scan for `FlowGraph` and for any
  serial-only marker: none)
- Code:

      inode = node_order[pos]
      ...
      to_node = to_graph_index[inode]
      if to_node >= 0:
          state.node_upstream_inflow_sub[to_node] += (

- Consequence: the module docstring says the graph is "REPLICATED +
  Map-fed under MPI", and `CLAUDE.md` says there is no FlowGraph MPI
  code. That is a convention; no code enforces it. With
  `control["mpi_grid"] = "nnodes"` and `to_graph_index`,
  `node_order`, `node_type` in the input file, `ModelMPI` would
  decompose them. Each rank's block of `node_order` and
  `to_graph_index` holds indices into the whole graph, and numba does
  not check bounds: the kernel reads and writes outside the rank's
  arrays. One rank gives the right answer, so a first trial passes.
  The array `PRMSChannel` has the same exposure (slice 7). Not run.
- Source: the read (a serial / MPI rule that no code enforces)
- Outcome: open

### 5.7 Two new nodes that flow to the same `NewNode` are accepted and chained in collated order; pywatershed refuses

- Where: `pws_phoenix/flow_graph_build.py:151-161`, `:182-193`
- Code:

      if node.flows_to in intercepted:
          raise ValueError(
              f"two new nodes intercept nhm_seg {node.flows_to}; "

  pywatershed (`prms_channel_flow_graph.py:928-930`):

      assert len(new_nodes_flow_to_nhm_seg) == len(
          np.unique(new_nodes_flow_to_nhm_seg)
      ), "Cant have more than one new node flowing to an existing or new node."

- Consequence: the module docstring says "The splice SEMANTICS are
  pywatershed's, verbatim". The duplicate check covers `nhm_seg`
  targets only. With `B.flows_to = A` and `C.flows_to = A`, B is
  placed above A, then C takes everything that flows to A, which is
  now B: the result is the chain `B -> C -> A`, and it depends on the
  order of `new_nodes`. A caller who meant two parallel nodes gets a
  series and no message. `tests/test_flow_graph_build.py` has the
  duplicate `nhm_seg` case (`:160-161`) and no duplicate `NewNode`
  case. Not run.
- Source: checklist B (differs from pywatershed)
- Outcome: open

### 5.8 Each step looks arrays up in the dataset, and the class holds every array reference twice

- Where: `pws_phoenix/flow_graph.py:501-503`, `:487-489`;
  `pws_phoenix/hydrology/prms_channel_flow_node.py:247-252`
- Code:

      def advance(self) -> None:
          for tt_class in node_types:
              tt_class.advance_type(self._obj)
      ...
      dataset["inflow_ts_prev"].values[:] = dataset["seg_inflow"].values
      ...
      self._graph_state = graph_state_type(
          **{nm: self._obj[nm].values for nm in field_names}
      )

- Consequence: `CLAUDE.md` prime directive 4: a process "never does
  `self._obj[name].values` inside a step". The channel type does two
  `Dataset.__getitem__` calls per step; the contract hands every
  type the dataset, so each new type with an advance does the same.
  `Process.__init__` has already captured `self._refs` for the same
  names (`pws_phoenix/process.py:332-335`); `_graph_state` is a
  second capture that rests on `.values` returning the same ndarray
  both times. The cost per step is small beside the kernel; the
  finding is the rule and the two sources of truth.
- Source: /code-review; checklist A item (`self._obj[name].values`
  inside a step method)
- Outcome: open

### 5.9 A row where a field does not apply holds 0.0 (or a default coefficient), not nan

- Where: `pws_phoenix/hydrology/prms_channel_flow_node.py:225-245`;
  `pws_phoenix/hydrology/pass_through_flow_node.py:81`;
  `pws_phoenix/hydrology/obsin_flow_node.py:111`;
  `pws_phoenix/hydrology/source_sink_flow_node.py:146-147`
- Code:

      for name in (
          "seg_inflow",
          "seg_outflow",
          "inflow_ts",
          "outflow_ts",
          "inflow_ts_prev",
      ):
          dataset[name].values[:] = 0.0

- Consequence: `CLAUDE.md` "FlowGraph": "Write OWN rows only ... nan
  marks 'does not apply'". Every `initialize_type` of this slice
  writes the whole array. `_prepare` and `_finalize` then touch own
  rows only, so on an inserted node's row `seg_outflow` is 0.0 for
  the whole run. `insert_flow_graph` keeps `seg_inflow` /
  `seg_outflow` in `output_var_names`
  (`pws_phoenix/prms_translate/assemble.py:452-455`), so the output
  shows a flow of zero at a reservoir whose `node_outflows` is not
  zero. The muskingum call likewise leaves `ts = 24`, `tsi = 24` and
  nan coefficients on rows of other types. The channel module
  docstring states a narrower rule ("its OWN rows of any field it
  shares with another type"), so the two documents disagree.
  `insert_flow_graph` relies on the zeros ("zeros on the new nodes'
  rows (what initialize_type writes there)", `assemble.py:501-502`).
- Source: checklist C (a type function that writes outside its own
  rows)
- Outcome: open

### 5.10 A name in a node type's `time_context` is never compared with the fields the graph serves

- Where: `pws_phoenix/flow_graph.py:462-468`
- Code:

      time_context_needed = {
          nm for tt in node_types for nm in getattr(tt, "time_context", ())
      }
      needs_epiweek = "epiweek" in time_context_needed
      TimeContext = namedtuple("TimeContext", ["epiweek", "itime_step"])

- Consequence: only the literal `"epiweek"` is acted on. A type that
  declares `("epi_week",)` composes, and `tctx.epiweek` is -1 on
  every step: its seasonal lookup uses -1 without an error. A type
  that declares a new name (`"doy"`) composes and fails at the first
  `calculate()` with a numba typing error. A new time scalar needs an
  edit inside `make_flow_graph`. The three declarations today are
  correct (text scan of `time_context`).
- Source: /code-review
- Outcome: open

### 5.11 Composition resolves the placeholder in `units` and leaves it in `description`

- Where: `pws_phoenix/flow_graph.py:438-442`; field descriptions at
  `:157`, `:164`, `:172`, `:180`, `:187`, `:194`, `:208`, `:216` and
  in the obsin and source_sink types
- Code:

      class_attrs[name] = dataclasses.replace(
          meta, units=IO_PLACEHOLDERS[meta.units][which]
      )

- Consequence: `Model._add_process_fields` writes
  `attrs = {"description": meta.description}` beside the resolved
  `units` (`pws_phoenix/model.py:790-792`). A variable of a cms graph
  reads `units: cms`, `description: "Lateral inflow to each node [io
  flow units]"`. The channel type has the opposite defect: its
  descriptions say `[cfs]` in a cms graph (5.4).
- Source: /code-review
- Outcome: open

### 5.12 Every composition compiles its own kernel

- Where: `pws_phoenix/flow_graph.py:280-283`, `:448-450`, `:474`
- Code:

      # NOT cache=True: built per composition and closes over the node
      # types' Dispatcher tuples (DEVELOPER.md "Numba cache").
      @numba.njit
      def kernel(state, s_per_time, n_substeps, tctx):

- Consequence: each `make_flow_graph` call builds a new `GraphState`
  class and a new kernel closure, so two calls with the same node
  types compile twice (every test, every notebook re-run, every
  rebuild of a model in one session). The kernel depends only on
  `tuple(node_types)`; `n_substeps` is an argument and `io_in_cfs`
  is not used by it. Keeping `(graph_state_type, kernel)` per tuple
  of node types would compile once per session. The cost was not
  measured. Related: 1.12.
- Source: /code-review
- Outcome: open

### 5.13 `splice_flow_graph`: smaller cases that pass without a message

- Where: `pws_phoenix/flow_graph_build.py:299-300`, `:336`, `:355-356`,
  `:375-377`, `:124-127`
- Code:

      if var.dims != (seg_dim,):
          continue
      ...
      for name, value in node.parameters.items():
          params[name][1][n_seg + ii] = value
      ...
      state[name] = np.full(n_nodes, np.nan, dtype=np.float64)

- Consequence (not run):
  - A `dis_seg` variable on any dims other than exactly `(seg_dim,)`
    is left off the graph grid without a message.
  - An integer parameter of a new node that the caller does not give
    is 0, which cannot be told from a given 0; a node type cannot
    make the "nan where it matters" check on it. A float given for
    an integer parameter is truncated by the assignment.
  - A new node type that declares a parameter whose name is also a
    padded `dis_seg` variable fails at `params[name]` with a bare
    `KeyError` (the name was skipped at `:327-328`).
  - Initial state is float64 whatever the variable declares.
  - `pad_weights` builds a second dense `(n_nodes, n_hru)` matrix
    beside the caller's `(n_seg, n_hru)` one (see 2.8), and a 1-D
    `weights` fails at `weights.shape[1]`.
- Source: checklist A items (an error that passes silently; a hidden
  copy)
- Outcome: open

### 5.14 A nan observation makes an obsin node pass its inflow through; the port keeps this and does not say so

- Where: `pws_phoenix/hydrology/obsin_flow_node.py:55-61`
- Code:

      if state.node_outflow_substep[inode] >= 0.0:
          state.obsin_sink_source_sum[inode] += (
              state.node_outflow_substep[inode] - inflow
          )
      else:
          state.node_outflow_substep[inode] = inflow

- Consequence: `nan >= 0.0` is False, so a missing observation
  stored as nan behaves as a negative one: the first substep's inflow
  is held all day. pywatershed does the same
  (`obsin_flow_node.py:51-55`), so the answers agree. The docstring
  documents negative values only and says gaps are "a data-prep
  concern"; nothing tells the caller that a nan left in
  `node_obs_flow` gives a flow and not a nan.
- Source: checklist B (an upstream quirk kept without a comment)
- Outcome: open

### 5.15 Docstrings and comments that no longer match the code

- Where and code:
  - `pws_phoenix/hydrology/pass_through_flow_node.py:12`, `:15`:
    `initialize_type(dataset)` and `substep(istep, inode, state)`.
    The signatures are `initialize_type(dataset, n_substeps,
    io_in_cfs)` and `substep(istep, inode, state, tctx, n_sub)`. This
    is the simplest type, the one a new type is copied from.
  - `pws_phoenix/hydrology/obsin_flow_node.py:10`,
    `pws_phoenix/hydrology/source_sink_flow_node.py:11`: "the
    deferred Budget design". The budget is built, and both types
    feed its exchanges term.
  - `pws_phoenix/flow_graph.py:20`: "the {code: name} map is ...
    stamped into the `node_type` variable's attrs"; `:481` stamps a
    list of names.
  - `pws_phoenix/flow_graph.py:43`, `:84`, `:107`;
    `pws_phoenix/hydrology/pass_through_flow_node.py:22`;
    `pws_phoenix/flow_graph_build.py:12`: "see CLAUDE.md" for the
    dispatch-spike finding and for the list of what was not ported.
    `CLAUDE.md` no longer holds either; the reasons moved to
    `docs/design_description.md`.
  - `pws_phoenix/flow_graph.py:289`: "(map-then-sum, as in
    PRMSChannel)" means the pws_phoenix `PRMSChannel`. pywatershed
    sums the three volumes per HRU, divides, then accumulates over
    HRUs (`prms_channel.py:459-466`).
  - `pws_phoenix/flow_graph.py:26`: "Registry dispatch (Stage 2
    Round A)" is a build-stage label with no meaning for a reader of
    the code.
- Consequence: a new node type written from the pass-through
  docstring fails in the kernel's call
  `fn(istep, inode, state, tctx, n_substeps)`.
- Source: /code-review, and the read
- Outcome: open

## Checklist C, item by item

Checklist B:

- Order of operations per element against pywatershed:
  - Channel type: `_prepare`, `_substep`, `_finalize` and
    `advance_type` match `PRMSChannelFlowNode` and
    `_calculate_subtimestep_numpy` statement for statement. The
    divisor is `n_sub` where pywatershed has the literal `24.0`
    (5.1).
  - Pass-through and obsin: match. The running means
    (`sum / (isubstep + 1)`) are computed once in `_finalize`; the
    value after the last substep is the same.
  - Source/sink: 5.3 (the branch structure), and the logged
    departure (the harvested outflow is the mean).
  - Graph kernel: matches `FlowGraph.calculate` (prepare, zero the
    accumulator, substeps in `node_order` with the hand-off
    downstream, finalize, harvest, outlet mask). The lateral inflow
    is summed after mapping; pywatershed's exchange sums per HRU
    first. This is the Map design, commented at
    `pws_phoenix/hydrology/prms_channel.py:533` and recorded in
    `docs/PORTS.md`.
  - Splice: `_resolve_topology` matches `_build_flow_graph_inputs`
    for interceptions and for chains, except 5.7. It searches the
    current `to_graph_index` where pywatershed searches the original
    `tosegment`; the two give the same rows because a second
    interception of one segment raises.
- A kernel that allocates per step, returns a value, or uses
  fastmath: none. The kernel and the twelve type functions write in
  place through `state` and return nothing. `calculate` builds one
  small `TimeContext` namedtuple per step. pywatershed compiles its
  channel substep with `fastmath=True`; the port does not, by rule.
- A frozen parameter that is edited: none. pywatershed's node maker
  edits `seg_slope` after it has computed the velocity; the port does
  not, and says why (`pws_phoenix/hydrology/prms_channel.py:163-166`).
- A float field without units; a budget missing a term: no field
  lacks units; 5.4 is a wrong unit. The budget has every term. For
  the channel, pass-through, obsin and source/sink types it is an
  identity: `node_storage_changes` or `node_sink_source` is defined
  as the residual. The module docstring says so. It follows that the
  budget cannot see 5.1, 5.3 or 5.5.
- An upstream quirk kept without a comment; a departure not logged:
  5.14. The obsin latch and the source/sink harvest are commented;
  the harvest is logged in `docs/MAINTENANCE.md`. Not ported and
  listed in the module docstring: `plot`, `initialize_netcdf`,
  `InflowExchange`, `type_check_nodes`, the
  `allow_disconnected_nodes` knob. Without that knob the port
  behaves as pywatershed does with it set to True, and without
  pywatershed's warning: `topological_order` puts isolated outlets
  first (the tests of the obsin and source/sink types rely on it).
- A restart variable missing from the declarations: none. Channel:
  `seg_inflow` (read by `advance_type`) and `outflow_ts` (read by
  the recursion across the day boundary) are `restart=True`;
  `inflow_ts`, `seg_outflow` are zeroed in `_prepare`;
  `inflow_ts_prev` is rewritten by `advance_type`. The other three
  types and the graph's work buffers are zeroed every step.

Checklist C:

- A type function that writes other than through `state`, or outside
  its own rows: the njit functions write `state.<field>[inode]`
  only. The numpy `initialize_type` functions write whole arrays
  (5.9). `advance_type` of the channel writes a whole array of a
  field only it declares.
- A unit branch inside njit code: none in this slice.

Checked and not a finding: under MPI the graph's grid is replicated,
so a raise in `initialize_type` that depends on the data happens on
every rank (compare 3.7). A to-index value outside the graph is
caught before the kernel, by the size mismatch when
`Discretization` stores the order (2.7 has the cases that are not
caught). The supplied initial state and a restart are written into
the existing arrays, so the references in `_graph_state` stay valid.
`self._obj["node_type"].attrs[...] = ...` reaches the dataset's own
variable. `FlowGraphKit.initial_state` is nan off the supplying
rows; no type of this slice has a restart variable that a new node
can supply. The work buffers and accumulators are `kind="variable"`,
but output is by `control["output_var_names"]`, so they are written
only when asked for.

## Notes for later slices

- Slice 6: `tctx.itime_step` is `time.current_index`
  (`pws_phoenix/flow_graph.py:508`). Check what it is on the first
  step of a warm start, for the daily node's first-day case. Check
  5.4 and 5.9 for the STARFIT fields, and whether an integer STARFIT
  parameter can be left at the 0 of 5.13.
- Slice 7: 5.6 applies to the array `PRMSChannel`.
  `muskingum_mann_coefficients` accepts nan parameters without a
  message (5.2).
- Slice 14: `insert_flow_graph` passes `n_substeps` through (5.1),
  and fills a new node's nan initial state from the channel's padded
  zeros (`pws_phoenix/prms_translate/assemble.py:520-523`).
- Slice 18: every file of this slice has 97-100% serial coverage,
  and no test caught 5.1-5.7. Why: no test composes the channel type
  with `n_substeps != 24` or with `io_in_cfs=False` and looks at the
  channel fields; no test gives a nan parameter, a nan request, or a
  missing channel parameter; the budget is an identity for these
  types. Three raises have no test
  (`pws_phoenix/flow_graph.py:395`, `:419`, `:426`:
  `n_substeps < 1`, duplicate type names, one field declared twice
  differently). `docs/PORTS.md:70` gives budget tolerances of
  1e-8 / 1e-9 for the graph; the default is 1e-10.
- Fix phase, runs that confirm before fixing: a drb channel graph
  with `n_substeps=1` (5.1); `splice_flow_graph` without `mann_n`
  (5.2); a source/sink node with no `flow_min` and a sink larger than
  the inflow, in both codes (5.3); two nodes with `flows_to` the same
  `NewNode`, in both codes (5.7).
