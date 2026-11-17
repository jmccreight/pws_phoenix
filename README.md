<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [pws phoenix](#pws-phoenix)
  - [Installation](#installation)
    - [Environment (local)](#environment-local)
    - [The mpixarray ecosystem](#the-mpixarray-ecosystem)
    - [mpixarray on Hovenweep](#mpixarray-on-hovenweep)
    - [pwpx on Hovenweep](#pwpx-on-hovenweep)
    - [CI setup (one-time, repo admin)](#ci-setup-one-time-repo-admin)
    - [Running tests](#running-tests)
    - [Run-time switches (environment variables, read at import)](#run-time-switches-environment-variables-read-at-import)
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
pip install -e . --no-deps              # the pws_phoenix package itself
```

The MPI + parallel-HDF5/NetCDF stack lives in the marked
`BEGIN/END parallel-io` block of `environment.yaml`. A serial
environment (e.g. on a platform conda cannot solve the parallel stack
for -- Windows has no conda-forge mpich/parallel hdf5) is derived by
deleting that block; CI's Windows job does exactly this. Serial runs
never need the block (the `mpixarray`
import is optional).

### The mpixarray ecosystem

The MPI path needs the two local packages installed in dependency
order (the mpix meta-repo's `setup_environment.sh` does this against
side-by-side clones; mpixarray branch `1.0.0`, mpix `main`):

```
pip install -e mpix
pip install -e mpixarray
```

The originals live on code.usgs.gov (`arc/py-hpc`). `mpixarray`
requires `mpix` by plain name, so installing `mpix` first makes pip
take the local clone. Their other requirements are listed in
`environment.yaml` (the "ecosystem requirements" group inside the
parallel-io block) so the editable installs pull nothing from PyPI.

### mpixarray on Hovenweep

Hovenweep provides MPI and parallel HDF5 as Cray modules. `mpi4py`
and `h5py` are compiled against them; everything else comes from pip.
Recorded Nov 2026 with `main` of both mpix and mpixarray and these
modules:

- `PrgEnv-gnu` version 8.4.0
- `cray-mpich` version 8.1.26
- `cray-hdf5-parallel` version 1.12.2.5

Run from the directory holding the two clones:

```
module load conda
conda create --name=mpixr python=3.14
conda activate mpixr

# mpi4py against cray-mpich (cc is the Cray compiler wrapper)
export MPICC=cc
python -m pip install --force-reinstall --no-cache-dir --no-deps \
    --no-binary=mpi4py mpi4py

# h5py against cray-hdf5-parallel
module load cray-hdf5-parallel
export CC=cc
export HDF5_MPI="ON"
python -m pip install --force-reinstall --no-cache-dir --no-deps \
    --no-binary=h5py h5py

# the two packages, in dependency order
pip install -e "./mpix[all]"
pip install -e "./mpixarray[all]"
```

Check the build:

```
python -c "import h5py; print(h5py.get_config().mpi)"   # True
ldd "$(python -c 'import h5py.h5; print(h5py.h5.__file__)')"
# expect libhdf5_parallel_gnu_*.so under /opt/cray/pe/hdf5-parallel
```

Every later session (login shell or batch script) needs the same
modules loaded, for example by sourcing a file like this:

```
module load conda cray-hdf5-parallel
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
conda activate mpixr
```

### pwpx on Hovenweep

The same steps as above, with conda supplying everything except the
MPI stack and HDF5. conda must install NO HDF5 here: `h5py` is built
against Cray's, and conda's `netCDF4` / `libnetcdf` would bring a
second, serial HDF5 that `h5py` then loads instead (Nov 2026: "Headers
are 1.12.2, library is 1.14.6", then an abort on `import h5py`). pip
supplies `netCDF4`; its wheel carries a private HDF5. Run from the
directory holding the three clones:

```
module load conda
sed -e '/BEGIN mpi-binaries/,/END mpi-binaries/d' \
    -e '/^  - netCDF4=/d' -e '/^  - libnetcdf=/d' \
    pws_phoenix/environment.yaml > environment_hovenweep.yaml
conda env create -n pwpx -f environment_hovenweep.yaml
conda activate pwpx
conda list | grep -i -E "hdf5|netcdf"   # expect nothing

# mpi4py against cray-mpich (cc is the Cray compiler wrapper)
export MPICC=cc
python -m pip install --force-reinstall --no-cache-dir --no-deps \
    --no-binary=mpi4py mpi4py

# h5py against cray-hdf5-parallel
module load cray-hdf5-parallel
export CC=cc
export HDF5_MPI="ON"
python -m pip install --force-reinstall --no-cache-dir --no-deps \
    --no-binary=h5py h5py

# pip only after the two compiled packages, so nothing replaces them
pip install netCDF4==1.7.3 h5netcdf hdf5plugin pytest-mpi
pip install -e ./mpix
pip install -e ./mpixarray
pip install -e ./pws_phoenix --no-deps
```

Check the build as for mpixarray above (`h5py.get_config().mpi` is
True; `ldd` on `h5py.h5` shows Cray's `libhdf5_parallel` and nothing
under `$CONDA_PREFIX`).

Session setup is the file above with `conda activate pwpx`. Tests:
`MPI_CMD="srun -n 4" ./tests/run_tests.sh`, after populating
`test_data/` (see "Running tests"; without it the data-dependent tests
fail, since the script allows no skips). The answer
keys in `tests/prms_oracle.toml` are pinned per platform and only
`macos-arm64` is pinned so far: a tree generated on Hovenweep needs
its `linux-x86_64` keys added there first, or the session aborts.

### CI setup (one-time, repo admin)

`.github/workflows/ci_ecosystem.yaml` runs the full environment +
ecosystem + MPI suite on ubuntu/macos (and the serial derivation on
windows). It needs:

1. Private GitHub mirrors of `mpix` and `mpixarray` under one
   org/user (e.g. `jmccreight`).
2. `ECOSYSTEM_ORG` at the top of `ci_ecosystem.yaml` edited to that
   org/user.
3. A fine-grained personal access token (Contents: read-only, scoped
   to just those two repos) stored as the `ECOSYSTEM_READ_PAT`
   repository secret.

Fork PRs do not receive secrets, so the ecosystem jobs run on
push/internal PRs only. The test data are not available in CI yet (the
oracle's answers are not published), so the data-dependent tests
cannot run there.

### Running tests

```
./tests/run_tests.sh                 # serial suite + each MPI file under mpirun
./tests/run_tests.sh --serial-only   # the serial suite alone (no skips)
./tests/run_tests.sh --parallel-only # the MPI files alone
```

The data-dependent tests read `test_data/` in this repo (gitignored
except `test_data/generate/`). Populate it once, from a clone of the
[pws-prms-oracle](https://github.com/DOI-USGS/pws-prms-oracle) whose
answers have been generated:

```
PWS_ORACLE_DIR=<oracle clone>/domains python test_data/generate/oracle_populate.py
python test_data/generate/starfit_fetch.py
```

`oracle_populate.py` links each domain's inputs and PRMS answers into
`test_data/<domain>/` and writes the derived answer variables (the ones
PRMS does not output) beside them; `--domain` limits it to some
domains. A control `prms_translate` cannot translate (sagehen's
`strmflow`) is linked but not derived. `starfit_fetch.py` downloads the
STARFIT data from pywatershed's 2.0.0 release into `test_data/starfit/`
and checks its sha256. `test_data/generate/compare_derived.py --root
test_data` checks the derived files against pywatershed's tree (while
pywatershed's derive step still exists).

Set `PWS_TEST_DATA` to use another such tree; pytest prints the
resolved path and whether it exists in its header. The answer keys the
tests expect are pinned in `tests/prms_oracle.toml`; a session aborts
when the tree's `.oracle_ikey` / `.oracle_okey` / `.derived_okey`
markers disagree with the pins, so a regenerated oracle shows up as a
reviewed key bump.

### Run-time switches (environment variables, read at import)

- `PWS_NUMBA_CACHE` (default `1`): numba keeps compiled kernels on disk
  beside the source, so a second Python process (a notebook, each MPI
  rank) skips the compile. `0` turns it off; the tests always run with
  it off. Rules and what invalidates it: `DEVELOPER.md` "Numba cache".
- `PWS_NUMBA_PARALLEL` (default `0`): `1` threads the per-HRU element
  loops with numba `prange` (every hru-grid process's main kernel:
  atmosphere, canopy, snow, runoff, soilzone, groundwater and the
  NoDprst/Ag variants; light helper loops and the sequential segment
  kernels stay single-threaded). Results are bit-identical. The thread count is numba's `NUMBA_NUM_THREADS`, which
  defaults to every logical core; under `mpirun` set it so that
  ranks x threads fits the machine, e.g. `NUMBA_NUM_THREADS=3 mpirun -n
  4 ...` on 14 cores. Pays off only for large domains (thousands of
  HRUs per process); measurements: `DEVELOPER.md` "Numba threading".

## Internal design considerations

See CLAUDE.md for design notes and discussion (incl. the xarray-simlab /
Landlab comparisons distilled from the retired xr design summary).

Decided designs, with their evidence and the invariants they impose on a
`Process`, are written up in `DEVELOPER.md` (first entry: per-step data
access -- hold the ndarray, not the Dataset lookup).

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
