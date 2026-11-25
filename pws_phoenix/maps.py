"""
map.py
======
Map: couples two discretizations by remapping a variable from a source grid to
a target grid via a weight matrix, held as its nonzero entries only
(``SparseWeights``: row index, column index, value) and applied row by
row. Foundational module -- imports only numpy/xarray/numba.

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

``weights`` is a ``SparseWeights`` or a dense ``(n_target, n_source)``
array; a dense array is converted at construction and not kept.

Scope: one-way, single-entry dicts (one Map carrying several
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

from typing import Any, Self

import numba
import numpy as np
import xarray as xr

from pws_phoenix import NUMBA_CACHE


class SparseWeights:
    """The nonzero entries of an ``(n_target, n_source)`` weight matrix:
    entry ``kk`` is ``vals[kk]`` at row ``rows[kk]`` (target index) and
    column ``cols[kk]`` (source index); every entry not listed is zero.
    Its size is the number of nonzeros, not ``n_target * n_source``.

    The entries are held sorted by (row, column) with zero values
    dropped: the order ``np.nonzero`` gives for the dense matrix, and
    the order ``_apply_sparse`` sums in. The sort makes new arrays, once,
    at construction (each the length of the nonzeros); the arrays given
    are not kept. A repeated (row, column) raises.
    """

    def __init__(
        self, rows: Any, cols: Any, vals: Any, shape: tuple[int, int]
    ) -> None:
        shape = (int(shape[0]), int(shape[1]))
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        vals = np.asarray(vals, dtype=np.float64)
        if not (rows.ndim == 1 and rows.shape == cols.shape == vals.shape):
            raise ValueError(
                "SparseWeights: rows, cols and vals must be 1-D and of "
                f"one length, got shapes {rows.shape}, {cols.shape}, "
                f"{vals.shape}."
            )
        for name, index, size in (
            ("rows", rows, shape[0]),
            ("cols", cols, shape[1]),
        ):
            if not np.issubdtype(index.dtype, np.integer):
                raise ValueError(
                    f"SparseWeights: {name} must be integers, got "
                    f"{index.dtype}."
                )
            if index.size and (index.min() < 0 or index.max() >= size):
                raise ValueError(
                    f"SparseWeights: {name} must be in [0, {size}), got "
                    f"{index.min()} to {index.max()}."
                )
        keep = vals != 0.0
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        order = np.lexsort((cols, rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
        repeated = (rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])
        if repeated.any():
            first = int(np.nonzero(repeated)[0][0])
            raise ValueError(
                "SparseWeights: (row, column) "
                f"({rows[first]}, {cols[first]}) is given more than once."
            )
        self.rows: np.ndarray = rows.astype(np.int64)
        self.cols: np.ndarray = cols.astype(np.int64)
        self.vals: np.ndarray = vals
        self.shape = shape

    @classmethod
    def from_dense(cls, weights: np.ndarray) -> Self:
        """The nonzero entries of a dense ``(n_target, n_source)``
        matrix."""
        if weights.ndim != 2:
            raise ValueError(
                "Map weights must be a SparseWeights or a 2-D "
                f"(n_target, n_source) array, got shape {weights.shape}."
            )
        rows, cols = np.nonzero(weights)
        return cls(rows, cols, weights[rows, cols], weights.shape)

    def by_row(
        self, col_start: int = 0, col_stop: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(indptr, cols, vals)`` for ``_apply_sparse``: row ``ii``
        holds entries ``indptr[ii]:indptr[ii + 1]``. With ``col_start``
        / ``col_stop``: only the columns ``[col_start, col_stop)``, with
        ``cols`` counted from ``col_start`` (one rank's block of the
        source grid). Built once per Map, at bind."""
        if col_stop is None:
            col_stop = self.shape[1]
        keep = (self.cols >= col_start) & (self.cols < col_stop)
        counts = np.bincount(self.rows[keep], minlength=self.shape[0])
        indptr = np.zeros(self.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return indptr, self.cols[keep] - col_start, self.vals[keep]

    def dense(self) -> np.ndarray:
        """The full ``(n_target, n_source)`` matrix, newly allocated at
        every call: for tests and small domains, never for a run."""
        out = np.zeros(self.shape)
        out[self.rows, self.cols] = self.vals
        return out

    def to_dataset(self, name: str) -> xr.Dataset:
        """The form in a file: ``<name>_rows``, ``<name>_cols`` and
        ``<name>_vals`` on the dim ``<name>_entry``, the shape in the
        ``n_target`` / ``n_source`` attributes of ``<name>_vals``. The
        indices are 0-based positions on the two grids. The variables
        hold these arrays (no copy)."""
        dim = f"{name}_entry"
        shape_attrs = {"n_target": self.shape[0], "n_source": self.shape[1]}
        return xr.Dataset(
            {
                f"{name}_rows": (dim, self.rows),
                f"{name}_cols": (dim, self.cols),
                f"{name}_vals": (dim, self.vals, shape_attrs),
            }
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, name: str) -> Self:
        """The weights that ``to_dataset(name)`` wrote into `ds`."""
        parts = [f"{name}_{part}" for part in ("rows", "cols", "vals")]
        missing = [part for part in parts if part not in ds]
        if missing:
            raise ValueError(
                f"weights {name!r}: the dataset has no {missing}. Weights "
                f"are stored as {parts}; a file that holds a dense "
                f"{name!r} matrix must be written again."
            )
        rows, cols, vals = (ds[part] for part in parts)
        return cls(
            rows.values,
            cols.values,
            vals.values,
            (vals.attrs["n_target"], vals.attrs["n_source"]),
        )


@numba.njit(cache=NUMBA_CACHE)
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
        weights: a ``SparseWeights``, or a dense ``(n_target, n_source)``
            array (converted here, not kept). The shape is validated
            against the two grids' sizes at Model assembly.
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
        self.weights: SparseWeights = (
            weights
            if isinstance(weights, SparseWeights)
            else SparseWeights.from_dense(np.asarray(weights))
        )
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
        self._indptr, self._cols, self._vals = self.weights.by_row()

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
        self._indptr, self._cols, self._vals = self.weights.by_row(
            col_start=start, col_stop=stop
        )

    def apply(self) -> None:
        """Distributed remap: local partial product, then Allreduce(SUM)
        into ``target_values`` (in place, on every rank)."""
        _apply_sparse(
            self._partial, self._indptr, self._cols, self._vals, self._source
        )
        self._comm.Allreduce(self._partial, self._target)
