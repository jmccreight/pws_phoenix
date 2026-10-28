"""
map.py
======
Map: couples two discretizations by remapping a variable from a source grid to
a target grid via a weight matrix (dense at construction, applied in
row-compressed sparse form). Foundational module -- imports only
numpy/xarray/numba.

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
full case study: docs/PORTS.md "The
stream-temperature chain".

Construction is keyword-only; the ``{source: target}`` dicts read "from -> to":

    Map(weights=w, grid={"hru": "segment"}, variable={"flow": "flow"})

Scope: one-way, dense weights at construction, single-entry dicts (one Map carrying several
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
pws_phoenix/CLAUDE.md). Bidirectional (fwd/rev) transforms are future
work.
"""

from typing import Any

import numba
import numpy as np
import xarray as xr


def _compress_rows(weights: np.ndarray) -> tuple:
    """The nonzeros of ``weights`` by row: ``(indptr, cols, vals)``, row
    ``ii`` holding entries ``indptr[ii]:indptr[ii + 1]``. Built once per
    Map; the dense matrix stays the construction and validation form."""
    rows, cols = np.nonzero(weights)
    counts = np.bincount(rows, minlength=weights.shape[0])
    indptr = np.zeros(weights.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, cols.astype(np.int64), weights[rows, cols]


@numba.njit
def _apply_sparse(target, indptr, cols, vals, source):
    """target = weights @ source over the nonzeros only; each row summed
    in column order (a dense BLAS product sums in its own blocked
    order, so the last bit can differ where a row has >= 3 entries)."""
    for ii in range(target.shape[0]):
        acc = 0.0
        for kk in range(indptr[ii], indptr[ii + 1]):
            acc += vals[kk] * source[cols[kk]]
        target[ii] = acc


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
        derivation: str | None = None,
    ) -> None:
        for name, mapping in (("grid", grid), ("variable", variable)):
            if len(mapping) != 1:
                raise ValueError(
                    f"Map {name!r} must be a single-entry "
                    f"{{source: target}} dict, got {mapping!r}"
                )
        ((self.source_grid, self.target_grid),) = grid.items()
        ((self.source_var, self.target_var),) = variable.items()
        # supply-or-derive provenance for the CONTRACT (Model.input_spec
        # "maps" section): how the weights are obtained, when the map's
        # author knows (None = the modeler supplies the matrix)
        self.derivation = derivation
        self.weights = np.asarray(weights)
        self.target_values = xr.DataArray(
            np.zeros(self.weights.shape[0]), dims=[self.target_grid]
        )

    def bind(self, source_ds: Any) -> None:
        """Hold the source and target ndarrays (called once, by
        ``Model._resolve_maps``): ``apply`` never looks a name up
        (DEVELOPER.md "Per-step data access" -- the one-ndarray-per-name
        invariant covers Map sources, which are process variables)."""
        self._source: np.ndarray = source_ds[self.source_var].values
        self._target: np.ndarray = self.target_values.values
        self._indptr, self._cols, self._vals = _compress_rows(self.weights)

    def apply(self) -> None:
        """Remap the bound source into ``target_values`` (in place, no
        per-step allocation)."""
        _apply_sparse(
            self._target, self._indptr, self._cols, self._vals, self._source
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
        self._local = (start, stop)
        # Per-step partial-product window buffer, reused every apply.
        self._partial = np.zeros(self.weights.shape[0])

    def bind(self, source_ds: Any) -> None:
        super().bind(source_ds)
        # this rank's weight columns only (column indices local to its
        # source slice), replacing the global form the base bind built
        start, stop = self._local
        self._indptr, self._cols, self._vals = _compress_rows(
            self.weights[:, start:stop]
        )

    def apply(self) -> None:
        """Distributed remap: local partial product, then Allreduce(SUM)
        into ``target_values`` (in place, on every rank)."""
        _apply_sparse(
            self._partial, self._indptr, self._cols, self._vals, self._source
        )
        self._comm.Allreduce(self._partial, self._target)
