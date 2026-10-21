"""Synthetic pins for conservation budgets (budget.py + Process.budgets).

No external data: a hand-built bucket process whose budget closes to
roundoff, a leaky variant whose budget does not (a loss the budget
does not list), and an exchange variant with a signed term. Covers
the declaration-time checks (terms exist, units present and agree,
components valid, no name collision), the generated fields, the three
control["budget"] modes, and residual_accum = running sum of residual.
"""

import numpy as np
import pytest
import xarray as xr
from pws_phoenix.clock import Time
from pws_phoenix.model import Model
from pws_phoenix.process import DataArrayMeta, Process

N_SPACE = 5
N_TIME = 6


def _meta(kind, desc, units="inches", **kw):
    return DataArrayMeta(
        kind=kind,
        dims=("space",),
        dtype=np.float64,
        description=desc,
        units=units,
        **kw,
    )


class _Bucket(Process):
    """storage += inflow - outflow; outflow = leak_coef * storage_prev."""

    leak_coef = _meta("parameter", "fraction leaving per step", units=None)
    inflow = _meta("input", "inflow")
    storage = _meta(
        "variable", "storage", restart=True, initial="storage_init"
    )
    storage_prev = _meta(
        "variable", "storage, previous step", initial="storage_init"
    )
    outflow = _meta("variable", "outflow")
    storage_change = _meta("variable", "storage - storage_prev")

    budgets = {
        "mass": {
            "inputs": ("inflow",),
            "outputs": ("outflow",),
            "storage_changes": ("storage_change",),
        },
    }

    def advance(self) -> None:
        self._obj["storage_prev"].values[:] = self._obj["storage"].values

    def _loss(self) -> np.ndarray:
        return 0.0

    def calculate(self, dt: np.float64, time: Time) -> None:
        oo = self._obj
        oo["outflow"].values[:] = (
            oo["leak_coef"].values * oo["storage_prev"].values
        )
        oo["storage"].values[:] = (
            oo["storage_prev"].values
            + oo["inflow"].values
            - oo["outflow"].values
            - self._loss()
        )
        oo["storage_change"].values[:] = (
            oo["storage"].values - oo["storage_prev"].values
        )


class _LeakyBucket(_Bucket):
    """A loss the budget does not account for: imbalanced by 0.1/step."""

    def _loss(self) -> np.ndarray:
        return 0.1


class _ExchangeBucket(_Bucket):
    """Adds a signed exchange term (either direction) to the balance."""

    exchange = _meta("input", "signed exchange flux")

    budgets = {
        "mass": {
            "inputs": ("inflow",),
            "outputs": ("outflow",),
            "storage_changes": ("storage_change",),
            "exchanges": ("exchange",),
        },
    }

    def calculate(self, dt: np.float64, time: Time) -> None:
        super().calculate(dt, time)
        oo = self._obj
        oo["storage"].values[:] += oo["exchange"].values
        oo["storage_change"].values[:] = (
            oo["storage"].values - oo["storage_prev"].values
        )


# ---------------------------------------------------------------------------
# declaration-time checks (pure introspection)
# ---------------------------------------------------------------------------


def test_generated_fields():
    names = tuple(_Bucket.get_variables())
    for suffix in (
        "inputs_sum",
        "outputs_sum",
        "storage_changes_sum",
        "residual",
        "residual_accum",
    ):
        assert f"_Bucket_mass_{suffix}" in names
    assert "_Bucket_mass_exchanges_sum" not in names
    ex_names = tuple(_ExchangeBucket.get_variables())
    assert "_ExchangeBucket_mass_exchanges_sum" in ex_names
    # the variant does NOT inherit the base's generated names
    assert "_Bucket_mass_residual" not in ex_names
    # only the accumulator is restart state; units follow the terms
    assert _Bucket.get_restart_variables() == (
        "storage",
        "_Bucket_mass_residual_accum",
    )
    assert _Bucket.get_variables()["_Bucket_mass_residual"].units == "inches"


def _declare(budgets_dict, extra=None):
    body = {
        "budgets": budgets_dict,
        "advance": lambda self: None,
        "calculate": lambda self, dt, time: None,
    }
    body.update(extra or {})
    # type(_Bucket) = the ABC metaclass (bare type() would conflict)
    cls = type(_Bucket)("_Bad", (_Bucket,), body)
    # deliberately invalid: keep it out of the registry that other
    # tests (test_restart.py) introspect wholesale
    Process._registry.pop("_Bad", None)
    return cls


def _terms(**components):
    return {"mass": components}


@pytest.mark.parametrize(
    "budgets_dict, match",
    [
        (
            _terms(inputs=("nope",), outputs=(), storage_changes=()),
            "not a declared field",
        ),
        (
            _terms(
                inputs=("inflow",), outputs=("leak_coef",), storage_changes=()
            ),
            "every term needs units",
        ),
        (
            _terms(inputs=("inflow",), storage_changes=()),
            "components must be",
        ),
        (
            _terms(
                inputs=("inflow",), outputs=(), storage_changes=(), sinks=()
            ),
            "components must be",
        ),
    ],
)
def test_declaration_errors(budgets_dict, match):
    cls = _declare(budgets_dict)
    with pytest.raises(ValueError, match=match):
        cls.get_variables()


def test_units_must_agree():
    cls = _declare(
        _Bucket.budgets,
        {"outflow": _meta("variable", "outflow in other units", units="cfs")},
    )
    with pytest.raises(ValueError, match="must agree"):
        cls.get_variables()


def test_generated_name_collision():
    cls = _declare(
        _Bucket.budgets,
        {"_Bad_mass_residual": _meta("variable", "squatter")},
    )
    with pytest.raises(ValueError, match="collides"):
        cls.get_variables()


# ---------------------------------------------------------------------------
# Model runs
# ---------------------------------------------------------------------------


def _inputs(rng):
    times = np.arange("2000-01-01", "2000-01-07", dtype="datetime64[D]")
    assert len(times) == N_TIME
    coords = {"time": times, "space": np.arange(N_SPACE)}
    return {
        "inflow": xr.DataArray(
            rng.uniform(0.0, 2.0, (N_TIME, N_SPACE)),
            dims=("time", "space"),
            coords=coords,
        ),
        "exchange": xr.DataArray(
            rng.uniform(-1.0, 1.0, (N_TIME, N_SPACE)),
            dims=("time", "space"),
            coords=coords,
        ),
        "parameters": xr.Dataset(
            {"leak_coef": ("space", rng.uniform(0.1, 0.5, N_SPACE))}
        ),
        "storage_init": xr.DataArray(
            rng.uniform(1.0, 3.0, N_SPACE), dims=("space",)
        ),
    }


def _model(cls, control, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    inp = _inputs(rng)
    entry = {
        "class": cls,
        "inflow": inp["inflow"],
        "parameters": inp["parameters"],
        "storage_init": inp["storage_init"],
    }
    if cls is _ExchangeBucket:
        entry["exchange"] = inp["exchange"]
    return Model({"bucket": entry}, control)


def test_bad_control_value():
    with pytest.raises(ValueError, match="control\\['budget'\\]"):
        _model(_Bucket, {"budget": "shout"})


@pytest.mark.parametrize("cls", [_Bucket, _ExchangeBucket])
@pytest.mark.parametrize("mode", [None, "warn", "error"])
def test_closed_budget(cls, mode):
    """A conservative process closes to roundoff in every mode; the
    residual and accumulator are ordinary dataset variables."""
    pre = cls.__name__
    with _model(cls, {"budget": mode}) as model:
        ds = model.discretizations["space"].dataset
        assert ds[f"{pre}_mass_residual"].attrs["units"] == "inches"
        np.testing.assert_array_equal(
            ds[f"{pre}_mass_residual_accum"].values, 0.0
        )
        accum = np.zeros(N_SPACE)
        for tt in range(N_TIME):
            model.time.set_index(tt)
            model.advance()
            model.calculate(np.float64(1.0))
            res = ds[f"{pre}_mass_residual"].values
            assert np.all(np.abs(res) < 1e-14)
            accum += res
            np.testing.assert_array_equal(
                ds[f"{pre}_mass_residual_accum"].values, accum
            )
        # the sums are what they say
        np.testing.assert_array_equal(
            ds[f"{pre}_mass_inputs_sum"].values, ds["inflow"].values
        )
        if cls is _ExchangeBucket:
            np.testing.assert_array_equal(
                ds[f"{pre}_mass_exchanges_sum"].values, ds["exchange"].values
            )


def test_leaky_none_is_silent():
    """mode None: computed and recorded, never judged."""
    with _model(_LeakyBucket, {"budget": None}) as model:
        model.run(np.float64(1.0), np.int32(N_TIME))
        ds = model.discretizations["space"].dataset
        np.testing.assert_allclose(
            ds["_LeakyBucket_mass_residual"].values, 0.1, rtol=1e-12
        )
        np.testing.assert_allclose(
            ds["_LeakyBucket_mass_residual_accum"].values,
            0.1 * N_TIME,
            rtol=1e-12,
        )


def test_leaky_warns():
    with _model(_LeakyBucket, {"budget": "warn"}) as model:
        with pytest.warns(UserWarning, match="imbalanced at 2000-01-01"):
            model.run(np.float64(1.0), np.int32(1))


def test_leaky_errors():
    """Error mode warns first (the per-rank detail: date, max
    |residual|, indices), then raises."""
    with _model(_LeakyBucket, {"budget": "error"}) as model:
        with pytest.raises(ValueError, match="_LeakyBucket budget imbalanced"):
            with pytest.warns(UserWarning, match="imbalanced at 2000-01-01"):
                model.run(np.float64(1.0), np.int32(1))


def test_leaky_within_tolerance_passes():
    with _model(
        _LeakyBucket, {"budget": "error", "budget_atol": 0.2}
    ) as model:
        model.run(np.float64(1.0), np.int32(N_TIME))
