---
inclusion: manual
---

# Local breadcrumb: channel routing DAG vs. parallel solve

A design/literature analysis was produced in this repo but belongs to another
project. The canonical note lives at user level so any project can reach it:

`~/.kiro/steering/river-routing-dag-vs-parallel.md`

It covers: why the topological sweep in `PRMSChannel` is an exact triangular
solve rather than a legacy artifact; what breaks if upstream inflow is lagged by
a substep (phase and amplitude error, not stability, not mass); the Muskingum
coefficient positivity window `2*K*x <= Δt <= 2*K*(1-x)` and why it blocks
naive Δt reduction; relevant literature (David et al. RAPID line, Water 2024
halo-reach method, Ponce & Theurer accuracy criteria); and a proposed experiment
sequence.

Repo-specific entry points identified, in case work resumes here:

- `pywatershed/hydrology/prms_channel.py` — `_initialize_channel_data`
  (topological sort ~line 262, coefficient clamps ~lines 364-374),
  `_muskingum_mann_numpy` (serial solver, cross-segment write ~line 617).
- `pywatershed/hydrology/prms_channel_flow_graph.py` — best place to experiment.
  `PRMSChannelFlowNode` already exposes `outflow_substep`, so a lagged variant is
  a change in the FlowGraph driver, not in node physics. Note that
  `PRMSChannelFlowNodeMaker._init_data` duplicates the coefficient derivation
  from `prms_channel.py` and lacks the `_adjust_parameters` guard.

- `pywatershed/utils/mmr_to_mf6_dfw.py`, `examples/07_mmr_to_mf6_chf_dfw.ipynb`,
  `autotest/test_mmr_to_mf6_dfw.py` — existing MMR to MODFLOW 6 CHF/DFW
  (diffusive wave) capability. Directly relevant: it supplies a
  higher-fidelity reference solution, so routing experiments can be scored
  against the diffusive wave rather than only against the legacy DAG-Muskingum
  answer.

`examples/parallel/` is a parameter-ensemble sweep, not spatial parallelism.

No code was changed and no experiments were run.
