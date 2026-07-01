"""
map.py
======
Map: couples two discretizations by remapping a variable from a source grid to a
target grid via a (dense, for now) weight matrix. Foundational module -- imports
only numpy/xarray.

Step-A scope: one-way, serial, dense weights. ``apply()`` fills the map's own
``target_values`` buffer, which the Model wires as the consumer process's
cross-grid input -- so writing it feeds the consumer directly (zero-copy).
Bidirectional (fwd/rev) transforms, sparse weights, and the cross-rank comm for
a distributed source grid (Step B) are future work (see pws_phoenix/CLAUDE.md).
"""

from typing import Any

import numpy as np
import xarray as xr


class Map:
    """Couples ``source_grid``'s ``source_var`` to ``target_grid``'s
    ``target_var`` as ``target = weights @ source``.

    ``weights`` is ``(n_target, n_source)``. The map owns ``target_values`` (a
    ``(n_target,)`` buffer); the Model wires it as the consumer's cross-grid
    input, so ``apply()`` writing it feeds the consumer with no copy.
    """

    def __init__(
        self,
        source_grid: str,
        source_var: str,
        target_grid: str,
        target_var: str,
        weights: Any,
    ) -> None:
        self.source_grid = source_grid
        self.source_var = source_var
        self.target_grid = target_grid
        self.target_var = target_var
        self.weights = np.asarray(weights)
        self.target_values = xr.DataArray(
            np.zeros(self.weights.shape[0]), dims=["space"]
        )

    def apply(self, source_ds: Any) -> None:
        """Remap ``source_ds[source_var]`` into ``target_values`` (in place)."""
        self.target_values.values[:] = (
            self.weights @ source_ds[self.source_var].values
        )
