"""Perfect-restart tests for the Ag family (fgr_ag_2yr data).

Extends the R2 recipe (test_restart_processes.perfect_restart) to the
three ag processes. The ag-specific wrinkle is the PER-STEP AREAS
under time-varying ag_frac. Both processes compute their areas from
the current ag_frac before their kernels on every step, so no area is
restart state. The one thing soilzone-ag must remember is the ag_frac
its storages are distributed for (ag_frac_prev: _update_areas moves
water when the current ag_frac differs from it) -- restart=True as a
prognostic marker, though it is not a storage.
The storages mirror plain soilzone's set (incl. the derived
soil_lower, and ag_soil_lower for symmetry). ObsET adds NO restart
state: its It0 buffers are per-step scratch (overwritten from current
values at each step start before any read).

Coverage: the live RunoffAg -> SoilzoneAg chain (static tiled ag_frac,
the spinup configuration) is the definitive test; the ObsET standalone
runs the DYNAMIC ag_frac + observed-AET iteration configuration.
Three more tests restart by SUPPLIED state (a cold start at clock
index 0), each resuming on a step where ag_frac CHANGES: runoff-ag
alone, soilzone-ag alone, and the live chain.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent))
from conftest import PWS_TEST_DATA, PYWS_ROOT
from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_runoff import PRMSRunoff, PRMSRunoffAg
from pws_phoenix.hydrology.prms_soilzone_ag import (
    PRMSSoilzoneAg,
    PRMSSoilzoneAgObsET,
)
from pws_phoenix.model import Model
from test_restart_processes import perfect_restart

DOMAIN_DIR = PWS_TEST_DATA / "fgr_ag_2yr"
GEN_SPINUP = DOMAIN_DIR / "output_spinup"
GEN_ANALYSIS = DOMAIN_DIR / "output_analysis"

# the live-chain configuration (test_prms_runoff_soilzone_ag.py)
RUNOFF_DISK_INPUTS = (
    "net_rain",
    "net_ppt",
    "net_snow",
    "potet",
    "snowmelt",
    "snow_evap",
    "pkwater_equiv",
    "pptmix_nopack",
    "snowcov_area",
    "through_rain",
    "hru_intcpevap",
    "intcp_changeover",
)
SOILZONE_DISK_INPUTS = ("transp_on",)
# standalone runoff-ag also takes the soil zone's back-edges from disk
# (test_prms_runoff_ag.py)
RUNOFF_STANDALONE_INPUTS = RUNOFF_DISK_INPUTS + (
    "soil_lower_prev",
    "soil_rechr_prev",
    "ag_soil_moist_prev",
    "ag_soil_rechr_prev",
)

# the ObsET standalone configuration (test_prms_soilzone_ag_obs_et.py)
OBS_DISK_INPUTS = (
    "dprst_evap_hru",
    "dprst_seep_hru",
    "hru_impervevap",
    "hru_intcpevap",
    "infil",
    "infil_ag",
    "potet",
    "transp_on",
    "snow_evap",
    "snowcov_area",
    "sroff",
    "sroff_vol",
)

_chain_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "ag_frac_static.nc",
] + [
    GEN_SPINUP / f"{nn}.nc" for nn in RUNOFF_DISK_INPUTS + SOILZONE_DISK_INPUTS
]
_chain_missing = [str(ff) for ff in _chain_needed if not ff.exists()]
chain_skipif = pytest.mark.skipif(
    bool(_chain_missing),
    reason=(
        "pywatershed fgr_ag_2yr spinup data not present/generated; "
        "missing: " + ", ".join(_chain_missing[:3])
    ),
)

_runoff_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "ag_frac_static.nc",
] + [GEN_SPINUP / f"{nn}.nc" for nn in RUNOFF_STANDALONE_INPUTS]
_runoff_missing = [str(ff) for ff in _runoff_needed if not ff.exists()]
runoff_skipif = pytest.mark.skipif(
    bool(_runoff_missing),
    reason=(
        "pywatershed fgr_ag_2yr spinup data not present/generated; "
        "missing: " + ", ".join(_runoff_missing[:3])
    ),
)

_obs_needed = [
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "dyn_ag_frac.param",
    DOMAIN_DIR / "aet_observed.nc",
    PYWS_ROOT / "pywatershed" / "utils" / "prms_dyn_param.py",
] + [GEN_ANALYSIS / f"{nn}.nc" for nn in OBS_DISK_INPUTS]
_obs_missing = [str(ff) for ff in _obs_needed if not ff.exists()]
obs_skipif = pytest.mark.skipif(
    bool(_obs_missing),
    reason=(
        "pywatershed fgr_ag_2yr analysis data not present/generated; "
        "missing: " + ", ".join(_obs_missing[:3])
    ),
)


def _load_forcings(names, gen_dir):
    return {
        nn: xr.load_dataarray(gen_dir / f"{nn}.nc").rename({"nhm_id": "nhru"})
        for nn in names
    }


def _static_ag_frac(template):
    """The spinup configuration: the static ag_frac time-tiled."""
    ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
    return xr.DataArray(
        np.tile(ag_static.values, (template.sizes["time"], 1)),
        dims=("time", "nhru"),
        coords={"time": template["time"], "nhru": template["nhru"]},
        name="ag_frac",
    )


def _dynamic_ag_frac(template):
    """Forward-fill the PRMS dynamic parameter file onto the forcing
    time axis (as in test_prms_soilzone_ag_obs_et.py)."""
    sys.path.insert(0, str(PYWS_ROOT))
    from pywatershed.utils.prms_dyn_param import PrmsDynamicParameter

    dp = PrmsDynamicParameter.load(
        DOMAIN_DIR / "dyn_ag_frac.param", dtype="float"
    )
    file_dates = np.array(
        [
            np.datetime64(f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}")
            for yy, mm, dd in dp.dates
        ]
    )
    times = template["time"].values.astype("datetime64[D]")
    idx = np.searchsorted(file_dates, times, side="right") - 1
    idx = np.clip(idx, 0, len(file_dates) - 1)
    return xr.DataArray(
        dp.data[idx, :],
        dims=("time", "nhru"),
        coords={"time": template["time"], "nhru": template["nhru"]},
        name="ag_frac",
    )


def test_flags_derivation_pin():
    """The ag flag sets, exactly (data-free)."""
    # runoff-ag adds NO restart state: its per-step areas are
    # recomputed from the input ag_frac before anything reads them
    assert set(PRMSRunoffAg.get_restart_variables()) == (
        set(PRMSRunoff.get_restart_variables())
        - {"PRMSRunoff_mass_residual_accum"}
    ) | {"PRMSRunoffAg_mass_residual_accum"}
    assert set(PRMSSoilzoneAg.get_restart_variables()) == {
        "soil_moist",
        "soil_rechr",
        "soil_lower",
        "slow_stor",
        "pref_flow_stor",
        "ag_soil_moist",
        "ag_soil_rechr",
        "ag_soil_lower",
        # the ag_frac the storages are distributed for; no AREA is
        # restart state
        "ag_frac_prev",
        # the mass budget's accumulator (restart=True by design)
        "PRMSSoilzoneAg_mass_residual_accum",
    }
    # ObsET adds iteration diagnostics + It0 scratch only -- nothing
    # prognostic beyond its base (its budget accumulator carries ITS
    # class-name prefix: a redeclared budget generates its own fields)
    assert set(PRMSSoilzoneAgObsET.get_restart_variables()) == (
        set(PRMSSoilzoneAg.get_restart_variables())
        - {"PRMSSoilzoneAg_mass_residual_accum"}
    ) | {"PRMSSoilzoneAgObsET_mass_residual_accum"}


@chain_skipif
def test_runoff_soilzone_ag_chain(tmp_path):
    """Live RunoffAg -> SoilzoneAg chain (spinup config), 90 days."""
    runoff_forcings = _load_forcings(RUNOFF_DISK_INPUTS, GEN_SPINUP)
    soilzone_forcings = _load_forcings(SOILZONE_DISK_INPUTS, GEN_SPINUP)
    ag_frac = _static_ag_frac(runoff_forcings["potet"])
    runoff_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoffAg.nc")
    soilzone_params = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc"
    )

    def make_model(control):
        process_dict = {
            "prms_runoff_ag": {
                "class": PRMSRunoffAg,
                "discretization": "nhru",
                "parameters": runoff_params,
                "ag_frac": ag_frac,
                **runoff_forcings,
            },
            "prms_soilzone_ag": {
                "class": PRMSSoilzoneAg,
                "discretization": "nhru",
                "parameters": soilzone_params,
                **soilzone_forcings,
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(
            process_dict, dict(control), discretizations=discretizations
        )

    perfect_restart(make_model, np.float64(1.0), 90, tmp_path)


@runoff_skipif
def test_runoff_ag_supplied_state():
    """Restart by SUPPLIED state: every restart variable is handed to
    a cold start (clock index 0) under its own name, with inputs that
    begin at the resume step. The three per-step areas are not
    among them: PRMSRunoffAg computes them from ag_frac before its
    kernel on every step, the resume step included.

    ag_frac is halved FROM the resume step on. The resumed model has
    only that step's ag_frac to work from, so a bit-identical result
    pins that the continuous run's kernel also used the CURRENT
    step's areas on the step of the change, and nothing remembered
    from the step before. The resume step is chosen for pervious-area
    runoff on an hru whose ag_frac changes (asserted): on a step
    without it a different pervious area changes nothing (step 30 is
    such a step)."""
    n_total, idx_b = 230, 200
    dt = np.float64(1.0)
    name = "prms_runoff_ag"
    forcings = _load_forcings(RUNOFF_STANDALONE_INPUTS, GEN_SPINUP)
    forcings["ag_frac"] = _static_ag_frac(forcings["potet"])
    forcings["ag_frac"][idx_b:] *= 0.5
    ag_frac = forcings["ag_frac"].values
    parameters = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoffAg.nc")

    def make_model(start, state):
        process_dict = {
            name: {
                "class": PRMSRunoffAg,
                "discretization": "nhru",
                "parameters": parameters,
                **{
                    nn: da.isel(time=slice(start, None))
                    for nn, da in forcings.items()
                },
                **state,
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(process_dict, {}, discretizations=discretizations)

    with make_model(0, {}) as m_ac:
        m_ac.run(dt, np.int32(idx_b + 1))
        proc_ac = m_ac.model_dict[name]
        assert (
            (proc_ac["hru_sroffp"].values > 0.0)
            & (ag_frac[idx_b] != ag_frac[idx_b - 1])
        ).any()
        m_ac.run(dt, np.int32(n_total - idx_b - 1))
    with make_model(0, {}) as m_ab:
        m_ab.run(dt, np.int32(idx_b))
    proc_ab = m_ab.model_dict[name]
    state = {
        nn: proc_ab[nn].values.copy()
        for nn in PRMSRunoffAg.get_restart_variables()
    }
    with make_model(idx_b, state) as m_bc:
        m_bc.run(dt, np.int32(n_total - idx_b))

    proc_bc = m_bc.model_dict[name]
    for nn in PRMSRunoffAg.get_var_names():
        np.testing.assert_array_equal(
            proc_bc[nn].values,
            proc_ac[nn].values,
            err_msg=f"variable '{nn}' not bit-identical after a "
            "restart by supplied state",
        )


@obs_skipif
def test_soilzone_ag_obs_et(tmp_path):
    """ObsET standalone: DYNAMIC ag_frac + AET iteration, 120 days."""
    forcings = _load_forcings(OBS_DISK_INPUTS, GEN_ANALYSIS)
    template = forcings["potet"]
    ag_frac = _dynamic_ag_frac(template)
    aet_observed = (
        xr.load_dataarray(DOMAIN_DIR / "aet_observed.nc")
        .sel(time=template["time"])
        .assign_coords(nhru=template["nhru"])
    )
    parameters = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc")

    def make_model(control):
        process_dict = {
            "prms_soilzone_ag_obs": {
                "class": PRMSSoilzoneAgObsET,
                "discretization": "nhru",
                "parameters": parameters,
                "ag_frac": ag_frac,
                "aet_observed": aet_observed,
                **forcings,
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(
            process_dict, dict(control), discretizations=discretizations
        )

    perfect_restart(make_model, np.float64(1.0), 120, tmp_path)


@obs_skipif
def test_soilzone_ag_supplied_state():
    """Restart by SUPPLIED state: every restart variable is handed to
    a cold start (clock index 0) under its own name, with inputs that
    begin at the resume step. Among them is ag_frac_prev, the ag_frac
    the storages are distributed for: _update_areas compares it with
    the resume step's ag_frac to move water between the storages.
    Without it (a cold start: nan) no change is seen and nothing
    moves.

    So the resume step must be one where ag_frac changes on an hru
    that holds ag soil water (asserted): 2001-01-01, the one change
    in the analysis period's dynamic ag_frac. On any other step
    ag_frac_prev equals the step's ag_frac and its absence changes
    nothing."""
    start, idx_b, n_after = 336, 366, 20
    dt = np.float64(1.0)
    name = "prms_soilzone_ag_obs"
    forcings = _load_forcings(OBS_DISK_INPUTS, GEN_ANALYSIS)
    template = forcings["potet"]
    forcings["ag_frac"] = _dynamic_ag_frac(template)
    forcings["aet_observed"] = (
        xr.load_dataarray(DOMAIN_DIR / "aet_observed.nc")
        .sel(time=template["time"])
        .assign_coords(nhru=template["nhru"])
    )
    parameters = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc")

    def make_model(first, state):
        process_dict = {
            name: {
                "class": PRMSSoilzoneAgObsET,
                "discretization": "nhru",
                "parameters": parameters,
                **{
                    nn: da.isel(time=slice(first, None))
                    for nn, da in forcings.items()
                },
                **state,
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(process_dict, {}, discretizations=discretizations)

    ag_frac = forcings["ag_frac"].values
    with make_model(start, {}) as m_ac:
        m_ac.run(dt, np.int32(idx_b - start))
        proc_ac = m_ac.model_dict[name]
        state = {
            nn: proc_ac[nn].values.copy()
            for nn in PRMSSoilzoneAgObsET.get_restart_variables()
        }
        assert (
            (ag_frac[idx_b] != ag_frac[idx_b - 1])
            & (state["ag_soil_moist"] > 0.0)
        ).any()
        m_ac.run(dt, np.int32(n_after))
    with make_model(idx_b, state) as m_bc:
        m_bc.run(dt, np.int32(n_after))

    proc_bc = m_bc.model_dict[name]
    for nn in PRMSSoilzoneAgObsET.get_var_names():
        np.testing.assert_array_equal(
            proc_bc[nn].values,
            proc_ac[nn].values,
            err_msg=f"variable '{nn}' not bit-identical after a "
            "restart by supplied state",
        )


_chain_dyn_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "dyn_ag_frac.param",
    PYWS_ROOT / "pywatershed" / "utils" / "prms_dyn_param.py",
] + [
    GEN_ANALYSIS / f"{nn}.nc"
    for nn in RUNOFF_DISK_INPUTS + SOILZONE_DISK_INPUTS
]
_chain_dyn_missing = [str(ff) for ff in _chain_dyn_needed if not ff.exists()]
chain_dyn_skipif = pytest.mark.skipif(
    bool(_chain_dyn_missing),
    reason=(
        "pywatershed fgr_ag_2yr analysis data not present/generated; "
        "missing: " + ", ".join(_chain_dyn_missing[:3])
    ),
)


@chain_dyn_skipif
def test_runoff_soilzone_ag_chain_dynamic_ag_frac():
    """The live RunoffAg -> SoilzoneAg chain on one grid under
    DYNAMIC ag_frac, across the step where ag_frac changes
    (2001-01-01), with both mass budgets in error mode.

    Both classes declare ag_area and hru_frac_perv, and on one grid
    each is ONE array. The soil zone must still see the change of
    ag_frac and move water between its storages: ag depths are scaled
    down where the ag area grows, and the excess goes to slow_stor
    where it shrinks (both asserted against the kernel's arithmetic,
    on hrus that hold ag soil water). When the soil zone took the
    previous ag area from the shared ag_area, runoff-ag had already
    overwritten it and nothing moved.

    The same run is then resumed on that step by SUPPLIED state and
    compared bit for bit, for both processes."""
    start, idx_b, n_after = 336, 366, 20
    dt = np.float64(1.0)
    classes = {
        "prms_runoff_ag": PRMSRunoffAg,
        "prms_soilzone_ag": PRMSSoilzoneAg,
    }
    runoff_forcings = _load_forcings(RUNOFF_DISK_INPUTS, GEN_ANALYSIS)
    runoff_forcings["ag_frac"] = _dynamic_ag_frac(runoff_forcings["potet"])
    forcings = {
        "prms_runoff_ag": runoff_forcings,
        "prms_soilzone_ag": _load_forcings(SOILZONE_DISK_INPUTS, GEN_ANALYSIS),
    }
    parameters = {
        "prms_runoff_ag": xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSRunoffAg.nc"
        ),
        "prms_soilzone_ag": xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc"
        ),
    }

    def make_model(first, state):
        process_dict = {
            name: {
                "class": cls,
                "discretization": "nhru",
                "parameters": parameters[name],
                **{
                    nn: da.isel(time=slice(first, None))
                    for nn, da in forcings[name].items()
                },
                **state.get(name, {}),
            }
            for name, cls in classes.items()
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(
            process_dict, {"budget": "error"}, discretizations=discretizations
        )

    ag_frac = runoff_forcings["ag_frac"].values
    old_frac, new_frac = ag_frac[idx_b - 1], ag_frac[idx_b]
    with make_model(start, {}) as m_ac:
        m_ac.run(dt, np.int32(idx_b - start))
        procs_ac = m_ac.model_dict
        state = {
            name: {
                nn: procs_ac[name][nn].values.copy()
                for nn in cls.get_restart_variables()
            }
            for name, cls in classes.items()
        }
        m_ac.run(dt, np.int32(1))

        # the step of the change: water moves as _update_areas says
        soilzone = procs_ac["prms_soilzone_ag"]
        ag_water = state["prms_soilzone_ag"]["ag_soil_moist"]
        seen = np.abs(new_frac - old_frac) > 1.0e-8 + 1.0e-5 * old_frac
        grew = seen & (new_frac > old_frac) & (ag_water > 0.0)
        shrank = (
            seen & (new_frac < old_frac) & (new_frac > 0.0) & (ag_water > 0.0)
        )
        assert grew.any() and shrank.any()
        np.testing.assert_allclose(
            soilzone["ag_soil_moist_redistribution"].values[grew],
            ag_water[grew] * (old_frac[grew] / new_frac[grew] - 1.0),
            rtol=1.0e-10,
        )
        assert (
            soilzone["ag_soil_moist_redistribution"].values[grew] < 0
        ).all()
        np.testing.assert_allclose(
            soilzone["slow_stor_redistribution"].values[shrank],
            ag_water[shrank] * (old_frac[shrank] - new_frac[shrank]),
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        assert (soilzone["slow_stor_redistribution"].values[shrank] > 0).all()

        m_ac.run(dt, np.int32(n_after - 1))
    with make_model(idx_b, state) as m_bc:
        m_bc.run(dt, np.int32(n_after))

    for name, cls in classes.items():
        for nn in cls.get_var_names():
            np.testing.assert_array_equal(
                m_bc.model_dict[name][nn].values,
                procs_ac[name][nn].values,
                err_msg=f"'{name}' variable '{nn}' not bit-identical "
                "after a restart by supplied state",
            )
