"""pws_phoenix: a pywatershed rewrite on xarray + mpixarray + numba."""

import os

# numba on-disk kernel cache (DEVELOPER.md "Numba cache"): on by default;
# PWS_NUMBA_CACHE=0 turns it off for every kernel (tests/conftest.py does).
NUMBA_CACHE = os.environ.get("PWS_NUMBA_CACHE", "1") != "0"

# numba threading over the hru element loops (prange): the main kernel
# of every hru-grid process is ALWAYS compiled threaded (parallel=True),
# and the Model holds numba at control["numba_n_threads"] threads,
# default 1 (Model._set_numba_threads; DEVELOPER.md "Numba threading").
# There is no switch here: one kind of kernel is ever compiled, so the
# on-disk cache cannot serve the other kind.
