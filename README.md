<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [pws phoenix](#pws-phoenix)
  - [Internal design considerations](#internal-design-considerations)
  - [External design considerations](#external-design-considerations)
    - [mpixarray](#mpixarray)
    - [landlab](#landlab)
  - [TODOs:](#todos)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# pws phoenix

[![CI](https://github.com/jmccreight/pws_phoenix/actions/workflows/ci.yaml/badge.svg)](https://github.com/jmccreight/pws_phoenix/actions/workflows/ci.yaml)

## Internal design considerations

See pws_phoenix/incarnations/xr/design_summary.md

1. Within process parallelism (over space): embarassing parallel; DAG ordered parallelism
2. Use xarray accessor pattern (?)
3. Define chunking in control, particularly time chunking (handle time buffering internally on in put and output classes)
4. Discretization object(s) that would manage MPI and other discretization methods.
5. Input mangement, consolidation, chunking, loading
6. Separate input and output (internal) chunking?
7. Interpreted vs mpi-execution; how to build a model and then scale it up? Could the model be defined and interacted on solely a single spatial chunk or some such subset?
8. Output issues: deprecate NetCdf4 package , use zarr, mpixarry (HDF5)
9. For loop over time: constructable as numba compiled at run time.
10. Can we rely on xarray -> numpy reference behavior? Developed mre_buffer_share_testing.py as a test of what we are relying on. test written.
11. hierarchical xarray ("datatree") for extensibility (composed processes)
12. attrs, like xarray-simlab. pydantic?
13. templating (jinja), or solutions to contract-forward vs standard subclassing (does this work with attrs)
14. Between process parallelism: execute when all upstream dependencies at current/next time are satisfied. Noted by CS4.6 in the above document.
15. Distinguish data structures for the 3 phases: 1) input, 2) simulation, 3) output. These can be solved computationally in the current pywatershed apriori (given a model without finding input files). Build utilities to do this.
16. metadata handling
17. unit treatment
18. What are the eventual base classes:

- Time discretization
- Control? (options)
- Discretization class (based on Landlab?)
- Input (time varying)
- Output
- Process
- ConservativeProcess
- ComposedProcess (?)
- Budget/Balance
- Model
- metadata module

## External design considerations

1. mpixarray
2. landlab
3. xarray-simlab
4. differentiability

### mpixarray

Challenges:

### landlab

## TODOs:

1. pre-commit hooks: ruff, pyright, mypy, GS security checks
2. CI: GS safety
3.
