"""pws_phoenix: a pywatershed rewrite on xarray + mpixarray + numba."""

import os

# numba on-disk kernel cache (DEVELOPER.md "Numba cache"): on by default;
# PWS_NUMBA_CACHE=0 turns it off for every kernel (tests/conftest.py does).
NUMBA_CACHE = os.environ.get("PWS_NUMBA_CACHE", "1") != "0"

# numba threading over the hru element loops (prange): OFF by default;
# PWS_NUMBA_PARALLEL=1 turns it on. Results are bit-identical (no
# reductions cross elements). Threads x MPI ranks must fit the cores:
# numba defaults to every core per process, so set NUMBA_NUM_THREADS
# under mpirun.
NUMBA_PARALLEL = os.environ.get("PWS_NUMBA_PARALLEL", "0") == "1"
