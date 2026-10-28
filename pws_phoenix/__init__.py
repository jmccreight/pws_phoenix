"""pws_phoenix: a pywatershed rewrite on xarray + mpixarray + numba."""

import os

# numba on-disk kernel cache (DEVELOPER.md "Numba cache"): on by default;
# PWS_NUMBA_CACHE=0 turns it off for every kernel (tests/conftest.py does).
NUMBA_CACHE = os.environ.get("PWS_NUMBA_CACHE", "1") != "0"
