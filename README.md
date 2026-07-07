<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [pws phoenix](#pws-phoenix)
  - [Installation](#installation)
    - [Environment (local)](#environment-local)
    - [The mpixarray ecosystem](#the-mpixarray-ecosystem)
    - [CI setup (one-time, repo admin)](#ci-setup-one-time-repo-admin)
    - [Running tests](#running-tests)
  - [Internal design considerations](#internal-design-considerations)
  - [External design considerations](#external-design-considerations)
    - [mpixarray](#mpixarray)
    - [landlab](#landlab)
  - [TODOs:](#todos)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# pws phoenix

[![CI](https://github.com/jmccreight/pws_phoenix/actions/workflows/ci.yaml/badge.svg)](https://github.com/jmccreight/pws_phoenix/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/jmccreight/pws_phoenix/graph/badge.svg?token=4DPR5CDRBS)](https://codecov.io/gh/jmccreight/pws_phoenix)

## Installation

### Environment (local)

`environment.yaml` defines the ONE `pwpx` environment (parallel; a
serial env is not maintained separately):

```
conda env create -f environment.yaml   # or: conda env update -n pwpx
conda activate pwpx
```

The MPI + parallel-HDF5/NetCDF stack lives in the marked
`BEGIN/END parallel-io` block of `environment.yaml`. A serial
environment (e.g. on a platform conda cannot solve the parallel stack
for -- Windows has no conda-forge mpich/parallel hdf5) is derived by
deleting that block; CI's Windows job does exactly this. Serial runs
of `incarnations/mpixarray` never need the block (the `mpixarray`
import is optional).

### The mpixarray ecosystem

The MPI path needs the local packages installed in dependency order
(the mpix meta-repo's `setup_environment.sh` does this against
side-by-side clones):

```
pip install -e numba_stdlib
pip install -e mpi4mpi4py
pip install -e ncxarray
pip install -e mpixarray
```

The originals live on code.usgs.gov (`arc/py-hpc`); their
`requirements.txt` `git+` cross-references must be commented out when
installing from local clones (`comment_remote_deps.sh` in the
meta-repo, or the sed in `ci_ecosystem.yaml`).

### CI setup (one-time, repo admin)

`.github/workflows/ci_ecosystem.yaml` runs the full environment +
ecosystem + MPI suite on ubuntu/macos (and the serial derivation on
windows). It needs:

1. Private GitHub mirrors of `numba_stdlib`, `mpi4mpi4py`,
   `ncxarray`, and `mpixarray` under one org/user (e.g.
   `jmccreight`).
2. `ECOSYSTEM_ORG` at the top of `ci_ecosystem.yaml` edited to that
   org/user.
3. A fine-grained personal access token (Contents: read-only, scoped
   to just those four repos) stored as the `ECOSYSTEM_READ_PAT`
   repository secret.

Fork PRs do not receive secrets, so the ecosystem jobs run on
push/internal PRs only. Test data for the pywatershed-validation
tests are not available in CI; those tests skip cleanly.

### Running tests

```
cd incarnations/mpixarray
./tests/run_tests.sh          # serial suite + each MPI file under mpirun
pytest tests/ -q              # serial only
```

## Internal design considerations

See CLAUDE.md for design notes and discussion (incl. the xarray-simlab /
Landlab comparisons distilled from the retired xr design summary).

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
