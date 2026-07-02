"""
map.py
======
Map: couples two discretizations by remapping a variable from a source grid to
a target grid via a (dense, for now) weight matrix. Foundational module --
imports only numpy/xarray.

Construction is keyword-only; the ``{source: target}`` dicts read "from -> to":

    Map(weights=w, grid={"hru": "segment"}, variable={"flow": "flow"})

Step-A scope: one-way, serial, dense weights, single-entry dicts (one Map
carrying several variables across the same grid pair with the same weights is
a possible future extension). ``apply()`` fills the map's own
``target_values`` buffer, which the Model wires as the consumer process's
cross-grid input -- so writing it feeds the consumer directly (zero-copy).
Scheduling/validation (apply once per step, before the first consumer, after
all declared source-grid writers; weights shape vs grid sizes) lives in
``Model._resolve_maps``. Bidirectional (fwd/rev) transforms, sparse weights,
and the cross-rank comm for a distributed source grid (Step B) are future
work (see pws_phoenix/CLAUDE.md).
"""

from typing import Any

import numpy as np
import xarray as xr


class Map:
    """Couples two grids as ``target = weights @ source``.

    Args (keyword-only):
        weights: ``(n_target, n_source)`` array (shape validated against the
            two grids' sizes at Model assembly).
        grid: single-entry ``{source_grid: target_grid}`` dict.
        variable: single-entry ``{source_var: target_var}`` dict.

    The map owns ``target_values`` (a ``(n_target,)`` buffer); the Model
    wires it as the consumer's cross-grid input, so ``apply()`` writing it
    feeds the consumer with no copy.
    """

    def __init__(
        self,
        *,
        weights: Any,
        grid: dict[str, str],
        variable: dict[str, str],
    ) -> None:
        for name, mapping in (("grid", grid), ("variable", variable)):
            if len(mapping) != 1:
                raise ValueError(
                    f"Map {name!r} must be a single-entry "
                    f"{{source: target}} dict, got {mapping!r}"
                )
        ((self.source_grid, self.target_grid),) = grid.items()
        ((self.source_var, self.target_var),) = variable.items()
        self.weights = np.asarray(weights)
        self.target_values = xr.DataArray(
            np.zeros(self.weights.shape[0]), dims=[self.target_grid]
        )

    def apply(self, source_ds: Any) -> None:
        """Remap ``source_ds[source_var]`` into ``target_values`` (in place)."""
        self.target_values.values[:] = (
            self.weights @ source_ds[self.source_var].values
        )
