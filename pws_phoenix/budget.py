"""
budget.py
=========
Conservation budgets (mass, energy, ...) for a Process, modeled on
pywatershed ``base/budget.py`` but re-expressed as DATA on the grid
dataset: every budget quantity is an ordinary variable (allocated once
at assembly, written by the normal output path, restartable), and the
per-step work is one in-place numba pass with zero allocation.

The balance (pywatershed's form; lhs vs rhs so the relative tolerance
is not measured against zero when storage does not change):

    inputs_sum + exchanges_sum  ==  outputs_sum + storage_changes_sum
    residual = lhs - rhs;  residual_accum += residual

``exchanges`` is the optional component for fluxes that may carry
either sign (e.g. a reservoir's sink/source); ``inputs`` and
``outputs`` are directional. A location is imbalanced when
``not (|residual| <= atol + rtol * |rhs|)`` -- a nan residual therefore
COUNTS as imbalanced (pywatershed's ``ignore_nans`` is not ported;
add it when a budget with nan locations arrives).

Ownership split (decided Oct 2026): the declaration
(``Process.budgets``), the generated fields and this kernel live on
the owning Process (``Process.check_budgets``); the Model decides WHEN
(right after the process's ``calculate``) and handles ``error`` mode
collectively under MPI (``Model._raise_if_imbalanced``). Per-rank
``warn`` mode needs no communication.

What is not ported from pywatershed: the ``global`` basis (every term
here is ``(space,)``, so the global residual is the spatial sum of the
written per-unit residual -- post-processing), per-term accumulations
(cumsums of written variables), the private netCDF writer and the
``__repr__`` table.
"""

import warnings

import numba
import numpy as np

# Component order is also the order of the generated fields.
COMPONENTS = ("inputs", "outputs", "storage_changes", "exchanges")
REQUIRED_COMPONENTS = ("inputs", "outputs", "storage_changes")
# Generated per budget: a sum per present component, then these two.
RESIDUAL_FIELDS = ("residual", "residual_accum")


def field_name(class_name: str, budget_name: str, suffix: str) -> str:
    """Name of a generated budget variable on the grid dataset. Prefixed
    by the owning class so two budgeted processes on one grid (flat
    namespace) cannot silently share a ``residual``."""
    return f"{class_name}_{budget_name}_{suffix}"


@numba.njit
def _sum_terms(out: np.ndarray, terms: tuple) -> None:
    """out[i] = sum over the (non-empty) tuple of 1-D term arrays."""
    for ii in range(out.shape[0]):
        acc = 0.0
        for tt in terms:
            acc += tt[ii]
        out[ii] = acc


@numba.njit
def _balance(
    residual: np.ndarray,
    residual_accum: np.ndarray,
    inputs_sum: np.ndarray,
    outputs_sum: np.ndarray,
    storage_changes_sum: np.ndarray,
    exchanges_sum: np.ndarray,
    has_exchanges: bool,
    rtol: float,
    atol: float,
) -> tuple[int, float]:
    """Residual + accumulation in place; returns (n_imbalanced,
    max |residual|). ``exchanges_sum`` is read only when
    ``has_exchanges`` (callers pass any same-shaped array otherwise)."""
    n_bad = 0
    max_abs = 0.0
    for ii in range(residual.shape[0]):
        lhs = inputs_sum[ii]
        if has_exchanges:
            lhs += exchanges_sum[ii]
        rhs = outputs_sum[ii] + storage_changes_sum[ii]
        res = lhs - rhs
        residual[ii] = res
        residual_accum[ii] += res
        diff = abs(res)
        if not (diff <= atol + rtol * abs(rhs)):
            n_bad += 1
        if diff > max_abs:
            max_abs = diff
    return n_bad, max_abs


class Budget:
    """One named budget of one Process: the term names per component and
    the names of its generated variables. Holds NO array references --
    ``calculate`` reads them off the bound dataset each call, exactly
    as a Process's ``calculate`` does (the MPI path rebinds the dataset
    every step)."""

    def __init__(
        self, class_name: str, budget_name: str, terms: dict[str, tuple]
    ) -> None:
        self.name = budget_name
        self.terms = {cc: tuple(terms.get(cc, ())) for cc in COMPONENTS}
        self.has_exchanges = len(self.terms["exchanges"]) > 0
        self.sum_names = {
            cc: field_name(class_name, budget_name, f"{cc}_sum")
            for cc in COMPONENTS
            if self.terms[cc]
        }
        self.residual_name = field_name(class_name, budget_name, "residual")
        self.residual_accum_name = field_name(
            class_name, budget_name, "residual_accum"
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """All generated variable names, in declaration order."""
        return (
            *self.sum_names.values(),
            self.residual_name,
            self.residual_accum_name,
        )

    def initialize(self, dataset) -> None:
        """Zero the accumulator (allocated nan-filled like any variable).
        Called by the Model before any restart restore, which may then
        overwrite it."""
        dataset[self.residual_accum_name].values[:] = 0.0

    def calculate(
        self, dataset, rtol: float, atol: float, warn: bool, time
    ) -> int:
        """Sum the terms, update residual/residual_accum in place, return
        the LOCAL count of imbalanced locations (warned about here when
        ``warn``; raising is the Model's, collectively under MPI)."""
        for cc, sum_name in self.sum_names.items():
            _sum_terms(
                dataset[sum_name].values,
                tuple(dataset[tt].values for tt in self.terms[cc]),
            )
        residual = dataset[self.residual_name].values
        inputs_sum = dataset[self.sum_names["inputs"]].values
        n_bad, max_abs = _balance(
            residual,
            dataset[self.residual_accum_name].values,
            inputs_sum,
            dataset[self.sum_names["outputs"]].values,
            dataset[self.sum_names["storage_changes"]].values,
            # any same-shaped array when there are no exchanges (unread)
            dataset[self.sum_names["exchanges"]].values
            if self.has_exchanges
            else inputs_sum,
            self.has_exchanges,
            rtol,
            atol,
        )
        if n_bad and warn:
            # Failure path only: the index lookup allocates.
            rhs = (
                dataset[self.sum_names["outputs"]].values
                + dataset[self.sum_names["storage_changes"]].values
            )
            close = np.abs(residual) <= atol + rtol * np.abs(rhs)
            where = np.flatnonzero(~close)
            warnings.warn(
                f"budget '{self.residual_name}' imbalanced at {time.current} "
                f"at {n_bad} location(s), max |residual| = {max_abs:.3e}, "
                f"first indices {where[:10].tolist()}",
                UserWarning,
                stacklevel=3,
            )
        return int(n_bad)
