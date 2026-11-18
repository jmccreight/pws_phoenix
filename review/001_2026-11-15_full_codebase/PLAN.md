<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Review 1: whole code base](#review-1-whole-code-base)
  - [Phases](#phases)
  - [One session](#one-session)
  - [Finding format](#finding-format)
  - [Code slices](#code-slices)
  - [Test slices](#test-slices)
  - [Checklists](#checklists)
  - [Known before the start (slice 15 records these as findings)](#known-before-the-start-slice-15-records-these-as-findings)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Review 1: whole code base

Started 2026-11-15. pws_phoenix at e33e217 (`develop`);
pywatershed at ba3974b (`develop`).
Method: `/code-review high <paths>`, one slice per session.
Paths are relative to the repository root.

Code lookup, from slice 5 on: pws_phoenix is indexed by CodeGraph
(`.codegraph/` at the repository root, created 2026-11-16). Before
grep or a whole-file read, ask it: the `codegraph_explore` tool, or
`codegraph explore "<symbol names or question>"`. It returns the
source of the named symbols and the call paths between them. The
pywatershed clone is not indexed: read it directly.

## Phases

0. Serial coverage, measured once: `coverage_serial.txt`.
1. Code slices 1-14. All code is reviewed, tested or not.
2. Test slices 15-22, after the code slices.
3. Fixes, after every slice is `reviewed`. Not part of a review
   session. Also deferred to here: MPI coverage, change-one-line
   probes of the tests, units tests for the remaining kernels.

## One session

Review only: a session edits nothing outside this directory.

1. Read this file. Take the first slice whose status is `todo`.
2. Create the slice's findings file. Record the pws_phoenix commit
   at its top (read `.git/HEAD` as text; no git commands).
3. Slices that use checklist B or C: JLM has granted read access to
   the pywatershed clone at the mpix_meta root. Confirm its
   `.git/HEAD` resolves to ba3974b; stop if it does not.
4. Run `/code-review high <paths>`.
5. Read the slice against its checklist for anything the command
   did not report. Use CodeGraph to follow each function and buffer
   of the slice to its callers and writers OUTSIDE the slice (the
   Model, other processes, the MPI path), and to confirm a finding
   before writing it.
6. Write the findings (format below), outcome `open`.
7. Set the slice's status to `reviewed`.

A test slice also reads, before step 4: `coverage_serial.txt` for
the code it covers, and the findings files of those code slices.
For each code finding it asks: why did no test catch this?

## Finding format

    ### 8.3 <one-sentence statement>
    - Where: `pws_phoenix/hydrology/prms_runoff.py:1234`
    - Code: <the offending lines, quoted>
    - Consequence: <what goes wrong, for which input>
    - Source: /code-review | checklist B | coverage
    - Outcome: open

## Code slices

| # | Findings file | Paths | Lines | Checklist | Status |
|---|---|---|---|---|---|
| 1 | 01_foundations.md | pws_phoenix/{clock,process,units,budget,processes_concrete}.py | 1,247 | A | reviewed |
| 2 | 02_grids_io.md | pws_phoenix/{maps,discretization,data_io}.py | 640 | A | reviewed |
| 3 | 03_model.md | pws_phoenix/model.py | 1,890 | A | reviewed |
| 4 | 04_config_contract.md | pws_phoenix/{config,model_contract_graph}.py | 595 | A | reviewed |
| 5 | 05_flow_graph.md | pws_phoenix/{flow_graph,flow_graph_build}.py, pws_phoenix/hydrology/{obsin,pass_through,prms_channel,source_sink}_flow_node.py | 1,544 | C | reviewed |
| 6 | 06_starfit.md | pws_phoenix/hydrology/starfit*_flow_node.py | 1,377 | C | reviewed |
| 7 | 07_small_ports.md | pws_phoenix/hydrology/prms_{groundwater,canopy,channel,hydraulic_geometry}.py | 1,772 | B | reviewed |
| 8 | 08_runoff.md | pws_phoenix/hydrology/prms_runoff.py | 3,071 | B | reviewed |
| 9 | 09_soilzone.md | pws_phoenix/hydrology/prms_soilzone.py | 2,002 | B | reviewed |
| 10 | 10_soilzone_ag.md | pws_phoenix/hydrology/prms_soilzone_ag.py | 2,196 | B | reviewed |
| 11 | 11_snow.md | pws_phoenix/hydrology/prms_snow.py | 2,474 | B | reviewed |
| 12 | 12_atmosphere.md | pws_phoenix/atmosphere/ | 1,256 | B | reviewed |
| 13 | 13_stream_temp.md | pws_phoenix/hydrology/prms_stream_{temp,shade}.py | 3,002 | B | todo |
| 14 | 14_prms_translate.md | pws_phoenix/prms_translate/, pws_phoenix/utils/ | 2,429 | D | todo |

## Test slices

All use checklist T.

| # | Findings file | Paths (under tests/) | Lines | Covers code slices | Status |
|---|---|---|---|---|---|
| 15 | 15_tests_harness.md | conftest.py, kernel_units.py, run_tests.sh, test_kernel_units.py | 2,560 | all | todo |
| 16 | 16_tests_framework.md | test_{budget,discretization,input_spec,model_contract_graph,model_step,output_finalize,process_fields,ref_behaviors,time,two_grid,units,units_check,up_low_regression,benchmark_coverage}*.py | 3,737 | 1-4 | todo |
| 17 | 17_tests_restart.md | test_restart*.py | 1,784 | 1, 3, 5-13 | todo |
| 18 | 18_tests_flow_graph.md | test_{flow_graph,flow_graph_build,mixed_channel_starfit,obsin_source_sink_nodes,starfit_daily_parity,starfit_flow_node}.py | 1,791 | 5, 6 | todo |
| 19 | 19_tests_ports_a.md | test_prms_{groundwater,canopy,channel,hydraulic_geometry,atmosphere,solar_geometry,snow}*.py | 2,873 | 7, 11, 12 | todo |
| 20 | 20_tests_ports_b.md | test_prms_{runoff,no_dprst,soilzone}*.py | 3,023 | 7-10 | todo |
| 21 | 21_tests_stream_temp.md | test_prms_stream_*.py | 2,309 | 13 | todo |
| 22 | 22_tests_translate.md | test_preprocess.py, test_prms_translate*.py, test_prms_units.py | 1,807 | 14 | todo |

MPI test files are read with their serial counterparts.

## Checklists

A. Framework
- A hidden copy: `deepcopy`, `np.asarray` / `astype` on an existing
  array, fancy indexing, an xarray operation that materializes.
- An allocation that is not model state, a declared per-step
  buffer, or a temporary inside a kernel.
- `self._obj[name].values` inside a step method.
- Under MPI: a raise on one rank alone; a collective call inside a
  rank-0 branch.
- A serial / MPI difference beyond the table in `CLAUDE.md`.
- An error that passes silently, or a check that cannot fail.

B. Ports
- Order of operations per element differs from pywatershed.
- A kernel that allocates per step, returns a value, or uses
  fastmath.
- A frozen parameter that is edited.
- A float field without units; a budget missing a term.
- An upstream quirk kept without a comment; a departure not logged
  in `docs/MAINTENANCE.md`.
- A restart variable missing from the declarations.

C. FlowGraph (checklist B, plus)
- A type function that writes other than through `state`, or
  outside its own rows.
- A unit branch inside njit code.

D. Translation
- A PRMS value that is changed or dropped without a raise or a
  message.
- An entry of the units table that disagrees with the declarations.

T. Tests
- A raise or a branch that no test reaches (`coverage_serial.txt`).
- A test that cannot fail: the expected value comes from the code
  under test, or a case added for the no-skips rule asserts
  something trivially true.
- A tolerance looser than the process's standard in
  `docs/PORTS.md`.
- A comparison where nan on both sides passes.
- A comparison of only some output variables, time steps or
  elements.
- A serial test with no MPI counterpart where the MPI code differs.
- A guard with no test showing that it raises.

## Known before the start (slice 15 records these as findings)

- No kernel units test for snow, atmosphere, the stream-temperature
  group, or the FlowGraph node types.
- `tests/kernel_units.py` requires statement coverage, not branch
  coverage.
- The composed FlowGraph class does not store its kernel, so a
  test cannot reach `kernel.py_func` from the class.
- `if sink > stor:` (`pws_phoenix/hydrology/prms_groundwater.py:308`)
  is taken only by the units tests, never by
  `tests/test_prms_groundwater.py` on its domain.
- The mypy pre-commit hook, in a hook environment built fresh on
  2026-11-15, reports 34 errors in 10 files at e33e217 (2 in
  `pws_phoenix/`, 32 in `tests/`). Not yet determined: real errors,
  or stubs of the hook's unpinned xarray / pytest.
