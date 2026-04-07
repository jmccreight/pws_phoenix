# pws phoenix

[![CI](https://github.com/jmccreight/pws_phoenix/actions/workflows/ci.yaml/badge.svg)](https://github.com/jmccreight/pws_phoenix/actions/workflows/ci.yaml)

## Internal design considerations

See /Users/jamesmcc/usgs/pws_phoenix/incarnations/xr/design_summary.md

0. Use xarray accessor pattern.
1. Within process parallelism: embarassing parallel; DAG ordered parallelism
2. Between process parallelism: execute when all upstream dependencies at current/next time are satisfied. Noted by CS4.6 in the above document.
3. Define chunking in control, particularly time chunking
4. Input mangement, consolidation, chunking, loading
5. Separate input and output (internal) chunking.
6. Interpreted vs mpi-execution; how to build a model and then scale it up? Could the model be defined and interacted on solely a single spatial chunk or some such subset?
7. Output issues: deprecate netcdf4, use zarr, mpixarry
8. For loop over time: constructable as numba at run time.
9. Can we rely on xarray -> numpy reference behavior? Developed mre_buffer_share_testing.py as a test of what we are relying on. There is a comment in the code that one of these fails in some circumstance.

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
