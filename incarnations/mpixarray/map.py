"""
map.py
======
Map: couples two discretizations by remapping a variable from a source grid to
a target grid via a (dense, for now) weight matrix. Foundational module --
imports only numpy/xarray.

CONCEPT (design principle, JLM July 2026): a Map is a grid-to-grid
CORRESPONDENCE, per variable -- one variable in, one variable out, the
same physical quantity on both sides. Renaming across the boundary is
part of the correspondence (``variable={source: target}``, e.g.
``sroff_vol -> seg_sroff_vol``); so are static unit/normalization
factors folded into the weights. What a Map must NEVER do is
ORIGINATE a quantity: computing a new variable from others is process
work, and belongs on the grid where its inputs live ("what map would
own the calculation?" has no good answer, because calculations don't
belong to correspondences). A consequence worth knowing: any
transform that respects this principle and has static coefficients IS
a weight matrix -- when an upstream code buries a nonlinear
computation inside an aggregation (PRMS stream_temp's in-loop cloud
cover), the fix is to relocate the computation to a process on the
source grid (ccov_hru on PRMSAtmosphereBase), after which the
remaining correspondence is exactly linear and rides this class
unchanged (weights can even be RECOVERED by probing the reference
implementation with basis vectors; a nonzero zero-input probe means
an affine constant, which a pure-matmul Map cannot express -- extend
deliberately if a real case ever needs it). The rejected alternative
-- a "computed Map" invoking kernels directly -- would have been a
second cross-grid concept beside this one, with its own MPI story and
(in the ccov case) a time dependence Maps otherwise don't have. The
full case study: incarnations/mpixarray/PORTS.md "The
stream-temperature chain".

Construction is keyword-only; the ``{source: target}`` dicts read "from -> to":

    Map(weights=w, grid={"hru": "segment"}, variable={"flow": "flow"})

Scope: one-way, dense weights, single-entry dicts (one Map carrying several
variables across the same grid pair with the same weights is a possible
future extension). ``apply()`` fills the map's own ``target_values`` buffer,
which the Model wires as the consumer process's cross-grid input -- so
writing it feeds the consumer directly (zero-copy). Scheduling/validation
(apply once per step, before the first consumer, after all declared
source-grid writers; weights shape vs grid sizes) lives in
``Model._resolve_maps``.

``MapMPI`` (Step B) crosses the parallel boundary: source grid distributed,
target grid serial/replicated -- a distributed mat-vec (local partial
product + Allreduce). This is the INTERIM cross-grid comm; the mpixarray
streaming-datatree work is expected to absorb this role (see
pws_phoenix/CLAUDE.md). Bidirectional (fwd/rev) transforms and sparse
weights are future work.
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
        """Remap ``source_ds[source_var]`` into ``target_values`` (in place,
        no per-step allocation)."""
        np.matmul(
            self.weights,
            source_ds[self.source_var].values,
            out=self.target_values.values,
        )


class MapMPI(Map):
    """Map whose SOURCE grid is space-decomposed (MPI) and whose target grid
    is serial, replicated on every rank.

    Construction is identical to ``Map`` (global weights); the decomposition
    is configured afterwards by ModelMPI via ``set_decomposition()``, once
    ``parallelize`` has determined each rank's extent.

    ``apply()`` is a distributed mat-vec: each rank multiplies its local
    weight columns by its local source slice, then ``Allreduce`` (SUM, the
    mpi4py default op) fills ``target_values`` identically on every rank --
    exactly what the replicated target grid consumes, with no further
    exchange. Communicates ``(n_target,)`` doubles per apply (vs allgathering
    the ``(n_source,)`` input) -- the right trade for a big source grid
    feeding a small target grid.
    """

    def set_decomposition(self, comm: Any, start: int, stop: int) -> None:
        """Configure this rank's slice ``[start, stop)`` of the source dim.
        Called by ModelMPI at build time."""
        self._comm = comm
        # One-time contiguous copy of this rank's weight columns (the sliced
        # view is non-contiguous and would slow every per-step matmul).
        self._weights_local = np.ascontiguousarray(self.weights[:, start:stop])
        # Per-step partial-product window buffer, reused every apply.
        self._partial = np.zeros(self.weights.shape[0])

    def apply(self, source_ds: Any) -> None:
        """Distributed remap: local partial product, then Allreduce(SUM)
        into ``target_values`` (in place, on every rank)."""
        np.matmul(
            self._weights_local,
            source_ds[self.source_var].values,
            out=self._partial,
        )
        self._comm.Allreduce(self._partial, self.target_values.values)
