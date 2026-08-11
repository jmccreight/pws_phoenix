"""Serial submodel regression: the FULL ported model chain vs
pywatershed answers (drb_2yr).

PRMSAtmosphere + PRMSCanopy + PRMSSnow + PRMSRunoff + PRMSSoilzone +
PRMSGroundwater -> Maps -> PRMSChannel, every process LIVE on the hru
grid's ONE shared dataset (NHM order), inter-process fluxes all by
structural sharing: atmosphere feeds everything downstream (hru_ppt /
hru_rain / hru_snow / t* / prmx / swrad / orad_hru / potet /
transp_on, and pptmix -- which canopy EDITS in place after it);
canopy feeds snow (net_* / hru_intcpevap) and gets snow's prior-step
pack state BACK (pk_ice_prev / freeh2o_prev); snow feeds runoff and
soilzone; runoff feeds soilzone (incl. the MUTABLE sroff/sroff_vol)
and groundwater; soilzone feeds groundwater and feeds runoff BACK its
prior-step soil state. The prior-step back-edges are correct because
the Model runs ALL advance() hooks before any calculate(). The solar
tables come from the LIVE compute_soltabs factory (dis variables in,
(ndoy, space) parameters out). In full-live mode the ONLY disk
forcings are the three CBH files (prcp/tmax/tmin) -- the whole model
from raw inputs. Three Maps aggregate the VOLUMES to the segment
grid; the channel routes muskingum_mann.

dt is SECONDS (86400.0); only the channel reads dt.

TWO modes, because tolerances differ fundamentally:

- "snow_disk": canopy -> runoff -> soilzone -> gw with the SNOW and
  ATMOSPHERE products from disk. STRICT 1e-10 (the ported-chain
  floor; see test_prms_runoff.py) with only the seg_stor_change
  cancellation carve-out -- the sensitive plumbing canary.
- "snow_live": the full 7-process chain from CBH. The generated
  answers come from pywatershed's fastmath numba path; snow's state
  carries ~1e-8-relative drift vs strict IEEE ALL season (plus
  pack-survival knife-edge flips, see test_prms_snow.py), which feeds
  every downstream flux and smears through muskingum's long memory
  (measured July 2026: 15%/1.6%/0.3%/0.015% of seg_lateral_inflow
  segment-days outside 1e-10/1e-8/1e-4/1e-2; atmosphere adds its own
  1e-5-level differences, see test_prms_atmosphere.py). The full
  chain vs these answers is therefore validated at (1e-2, 1e-2) with
  an outlier fraction -- the fastmath-answers ceiling. The PRECISION
  guarantees live in the per-process tests (snow itself is
  BIT-IDENTICAL to pywatershed's strict path;
  test_prms_snow_ab_numpy.py).

Requires GENERATED pywatershed test data; skips with a reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from atmosphere.prms_atmosphere import PRMSAtmosphere
from atmosphere.prms_solar_geometry import compute_soltabs
from discretization import Discretization
from hydrology.prms_canopy import PRMSCanopy
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import PRMSGroundwater
from hydrology.prms_runoff import PRMSRunoff
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone
from map import Map
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

# snow_live mode: the ONLY disk forcings -- the raw CBH files
CBH_NAMES = ("prcp", "tmax", "tmin")
# snow_disk mode: atmosphere + snow products fed from disk instead of
# PRMSAtmosphere/PRMSSnow; shared ones are fed ONCE, to the FIRST
# consumer in the schedule -- every consumer reads the shared field
CANOPY_INPUT_NAMES = (
    "transp_on",
    "hru_ppt",
    "hru_rain",
    "hru_snow",
    "potet",
    "pptmix",  # mutable: canopy zeroes it in place; snow reads after
)
SNOW_DISK_CANOPY_EXTRA = ("pk_ice_prev", "freeh2o_prev")
SNOW_DISK_RUNOFF_NAMES = (
    "snowmelt",
    "snow_evap",
    "pkwater_equiv",
    "pptmix_nopack",
    "snowcov_area",
    "through_rain",
)
ANSWER_NAMES = (
    "seg_lateral_inflow",
    "seg_upstream_inflow",
    "seg_inflow",
    "seg_outflow",
    "seg_stor_change",
    "channel_outflow_vol",
)
# per-mode criteria (see module docstring): (rtol/atol, max fraction
# of segment-days allowed outside tolerance)
MODE_TOL = {
    "snow_disk": (1.0e-10, 0.0),
    "snow_live": (1.0e-2, 1.0e-3),  # observed outliers: 0.014%
}
# seg_stor_change = (seg_inflow - seg_outflow) * s_per_time: a
# DIFFERENCE of near-equal numbers. Both operands match pywatershed at
# the chain tolerance (validated above it in ANSWER_NAMES), but
# cancellation strips the leading digits and amplifies the benign
# residue (mapped-sum float order + the kernels' transcendental-op
# ulps) -- the chain tolerance is mathematically unattainable for this
# diagnostic.
PER_VAR_TOL = {"seg_stor_change": (1.0e-7, 1.0e-4)}  # (rtol, atol)
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSAtmosphere.nc",
        DOMAIN_DIR / "parameters_PRMSCanopy.nc",
        DOMAIN_DIR / "parameters_PRMSSnow.nc",
        DOMAIN_DIR / "parameters_PRMSRunoff.nc",
        DOMAIN_DIR / "parameters_PRMSSoilzone.nc",
        DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
        DOMAIN_DIR / "parameters_PRMSChannel.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "parameters_dis_seg.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [
        GEN_DIR / f"{nn}.nc"
        for nn in CANOPY_INPUT_NAMES
        + SNOW_DISK_CANOPY_EXTRA
        + SNOW_DISK_RUNOFF_NAMES
        + ANSWER_NAMES
    ]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


def _open_hru_forcing(name):
    """pywatershed output files put variables on 'nhm_id'; rename to the
    hru grid dim. load_dataarray = open, LOAD, CLOSE: tests must not
    accumulate open netCDF handles -- past ~128 the xarray file-manager
    LRU churns reopen/evict on every access (a de-facto hang)."""
    return xr.load_dataarray(GEN_DIR / f"{name}.nc").rename({"nhm_id": "nhru"})


@pytest.fixture(scope="module")
def channel_parameters():
    return xr.load_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")


@pytest.fixture(scope="module")
def weights(channel_parameters):
    """0/1 aggregation weights from hru_segment (1-based; < 1 = the flux
    leaves the model -- a zero column, pywatershed's mass discard)."""
    hru_segment = channel_parameters["hru_segment"].values
    n_seg = channel_parameters.sizes["nsegment"]
    ww = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            ww[hru_segment[ihru] - 1, ihru] = 1.0
    return ww


@pytest.fixture(scope="module")
def answers():
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module", params=["snow_disk", "snow_live"])
def model_run(request, channel_parameters, weights, tmp_path_factory):
    mode = request.param
    out_dir = tmp_path_factory.mktemp(f"prms_channel_output_{mode}")
    gw_parameters = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
    )
    runoff_parameters = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSRunoff.nc"
    )
    soilzone_parameters = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSSoilzone.nc"
    )
    canopy_parameters = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSCanopy.nc"
    )
    snow_parameters = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSnow.nc")
    # the LIVE solar-geometry factory supplies the (ndoy, nhru) tables
    # (snow_live; the snow_disk mode has no soltab consumer)
    soltabs = compute_soltabs(
        xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    )
    snow_parameters = xr.merge(
        [snow_parameters, soltabs[["soltab_horad_potsw"]]]
    )
    atmosphere_parameters = xr.merge(
        [
            xr.load_dataset(DOMAIN_DIR / "parameters_PRMSAtmosphere.nc"),
            soltabs[["soltab_potsw", "soltab_horad_potsw"]],
        ]
    )

    # dict order = schedule: NHM order ([atmosphere] -> canopy ->
    # [snow] -> runoff -> soilzone -> gw) so each consumes its
    # producers' same-step shared buffers; the *_prev inputs (canopy's
    # pack state, runoff's soil state) are PRIOR-step by construction
    # (all advance() hooks run before any calculate())
    snow_live = mode == "snow_live"
    canopy_forcings = (
        ()
        if snow_live  # everything from PRMSAtmosphere
        else CANOPY_INPUT_NAMES + SNOW_DISK_CANOPY_EXTRA
    )
    runoff_forcings = () if snow_live else SNOW_DISK_RUNOFF_NAMES
    process_dict = {
        **(
            {
                "prms_atmosphere": {
                    "class": PRMSAtmosphere,
                    "discretization": "nhru",
                    "parameters": atmosphere_parameters,
                    **{
                        nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
                        .rename({"nhm_id": "nhru"})
                        .astype(np.float64)
                        for nn in CBH_NAMES
                    },
                }
            }
            if snow_live
            else {}
        ),
        "prms_canopy": {
            "class": PRMSCanopy,
            "discretization": "nhru",
            "parameters": canopy_parameters,
            **{nn: _open_hru_forcing(nn) for nn in canopy_forcings},
        },
        **(
            {
                "prms_snow": {
                    "class": PRMSSnow,
                    "discretization": "nhru",
                    "parameters": snow_parameters,
                }
            }
            if snow_live
            else {}
        ),
        "prms_runoff": {
            "class": PRMSRunoff,
            "discretization": "nhru",
            "parameters": runoff_parameters,
            **{nn: _open_hru_forcing(nn) for nn in runoff_forcings},
        },
        "prms_soilzone": {
            "class": PRMSSoilzone,
            "discretization": "nhru",
            "parameters": soilzone_parameters,
        },
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
            "parameters": gw_parameters,
            "gwstor_init": gw_parameters["gwstor_init"],
        },
        "prms_channel": {
            "class": PRMSChannel,
            "discretization": "nsegment",
            "parameters": channel_parameters,
            "segment_flow_init": channel_parameters["segment_flow_init"],
        },
    }
    maps = {
        "sroff": Map(
            weights=weights,
            grid={"nhru": "nsegment"},
            variable={"sroff_vol": "seg_sroff_vol"},
        ),
        "ssres": Map(
            weights=weights,
            grid={"nhru": "nsegment"},
            variable={"ssres_flow_vol": "seg_ssres_flow_vol"},
        ),
        "gw": Map(
            weights=weights,
            grid={"nhru": "nsegment"},
            variable={"gwres_flow_vol": "seg_gwres_flow_vol"},
        ),
    }
    discretizations = {
        "nhru": Discretization(
            ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
        ),
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_channel.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict,
        control,
        maps=maps,
        discretizations=discretizations,
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return {"model": model, "control": control, "mode": mode}


def _mode_tols(mode, nn):
    """(rtol, atol) for a variable in a mode: the seg_stor_change
    cancellation carve-out applies only when stricter than the mode
    base (i.e. in snow_disk)."""
    base, _ = MODE_TOL[mode]
    if mode == "snow_disk":
        return PER_VAR_TOL.get(nn, (base, base))
    return (base, base)


class TestPRMSChannelSubmodel:
    def test_all_variables_all_timesteps(self, model_run, answers):
        """Every channel output matches pywatershed over the full run,
        to the mode's criterion (module docstring)."""
        mode = model_run["mode"]
        _, max_frac = MODE_TOL[mode]
        output_ds = xr.open_zarr(
            model_run["control"]["output_serial_zarr"], consolidated=False
        )
        for nn in ANSWER_NAMES:
            rtol_var, atol_var = _mode_tols(mode, nn)
            bad = ~np.isclose(
                output_ds[nn].values,
                answers[nn].values,
                rtol=rtol_var,
                atol=atol_var,
            )
            frac = bad.mean()
            assert frac <= max_frac, (
                f"[{mode}] variable '{nn}': {frac:.3%} of segment-days "
                f"outside tolerance (allowed {max_frac:.3%})"
            )

    def test_final_state(self, model_run, answers):
        """Final in-memory state to the mode's tolerance (snow_live: a
        knife-edge-affected segment on the final day is possible)."""
        mode = model_run["mode"]
        base, _ = MODE_TOL[mode]
        max_frac = 0.0 if mode == "snow_disk" else 0.01
        proc = model_run["model"].model_dict["prms_channel"]
        for nn in ("seg_outflow", "seg_inflow"):
            bad = ~np.isclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=base,
                atol=base,
            )
            assert bad.mean() <= max_frac, (
                f"[{mode}] '{nn}' final state differs"
            )

    def test_internal_parameters_frozen(self, model_run):
        proc = model_run["model"].model_dict["prms_channel"]
        for nn in ("c0", "c1", "c2", "ts", "tsi", "tosegment0"):
            with pytest.raises(ValueError):
                proc[nn].values[:] = 0

    def test_segment_order_is_dis_derived(self, model_run):
        """segment_order arrived on the grid dataset via the dis
        (topo_order=), read-only, and is a valid ordering."""
        proc = model_run["model"].model_dict["prms_channel"]
        order = proc["segment_order"].values
        assert not order.flags.writeable
        to_seg = proc["tosegment0"].values
        position = np.empty_like(order)
        position[order] = np.arange(order.shape[0])
        for iseg in range(order.shape[0]):
            if to_seg[iseg] >= 0:
                assert position[iseg] < position[to_seg[iseg]]
