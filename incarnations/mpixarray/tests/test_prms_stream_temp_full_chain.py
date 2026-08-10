"""The FULL chain through stream temperature, from raw CBH (drb_2yr).

Stage 4 of the stream-temp chain -- the complete NHM:
PRMSAtmosphere -> PRMSCanopy -> PRMSSnow -> PRMSRunoff ->
PRMSSoilzone -> PRMSGroundwater on the hru grid (all live, as
test_prms_channel.py snow_live mode), THIRTEEN Maps (the three
volume maps feeding PRMSChannel + the ten aggregation Maps feeding
PRMSStreamTemp, weights from derive_aggregation_weights), and the
segment grid running PRMSChannel -> PRMSHydraulicGeometryWidthOnly ->
PRMSStreamTemp with everything LIVE: seg_outflow /
seg_lateral_inflow by structural sharing from the channel,
seg_flow_width from hydraulic geometry, the aggregates from the Maps.
The only disk inputs are the three CBH files + the humidity_hru CBH
forcing (a 1-variable carrier; no live producer exists -- humidity is
external everywhere).

TOLERANCE STORY (the fastmath-answers ceiling, one step further):
the stream-temp answers (output_stream_temp) come from a DIFFERENT
pywatershed generation run than the chain answers (output/), and the
two pywatershed runs differ from EACH OTHER: identical atmosphere,
but snow knife-edge flips (pywatershed's fastmath numba
reproducibility across builds -- see test_prms_snow.py) first appear
at day 80 and cascade (0.2% of snowmelt hru-days, max 1.78 in;
7% of sroff hru-days; muskingum smears to 68% of seg_outflow
seg-days, max 59 cfs). Our chain is bit-identical to pywatershed's
strict-IEEE numpy path, i.e. yet another valid trajectory. Stream
temperature is mostly damped against flow noise, but NOFLOW /
mixed-inlet knife edges flip individual seg-days to sentinels, so the
criterion is the stream-temp family tolerance (5e-3) with a small
allowed outlier fraction, measured and pinned below. The PRECISION
guarantees live in the standalone tests (physics 5e-3 vs Fortran with
answers-fed inputs; aggregation weights == kernels at 1e-12).
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
from hydrology.prms_hydraulic_geometry import PRMSHydraulicGeometryWidthOnly
from hydrology.prms_runoff import PRMSRunoff
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone
from hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    PRMSStreamTemp,
    derive_aggregation_weights,
)
from map import Map
from model import Model
from process import DataArrayMeta, Process

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"  # the chain answers (skip-check only)
GEN_DIR_ST = DOMAIN_DIR / "output_stream_temp"  # the stream-temp answers

CBH_NAMES = ("prcp", "tmax", "tmin")
ST_ANSWER_NAMES = (
    "seg_tave_water",
    "seg_tave_upstream",
    "seg_tave_gw",
    "seg_tave_ss",
    "seg_tave_lat",
    "seg_shade",
)
# rtol/atol + max fraction of finite seg-days allowed outside -- see
# the module docstring. Measured (Jul 2026, first run): seg_tave_water
# 0.16%, seg_tave_upstream 0.14%, seg_tave_lat 0.025%,
# seg_tave_gw / seg_tave_ss / seg_shade 0.0 -- the air-temperature-
# driven variables are exact-in-tolerance (identical atmosphere);
# only the flow-driven ones see the knife-edge cascade. Threshold =
# ~3x the worst measured.
RTOL = ATOL = 5.0e-3
MAX_OUTLIER_FRAC = 5.0e-3
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = (
    [
        DOMAIN_DIR / f"parameters_PRMS{nn}.nc"
        for nn in (
            "Atmosphere",
            "Canopy",
            "Snow",
            "Runoff",
            "Soilzone",
            "Groundwater",
            "Channel",
            "HydraulicGeometryWidthOnly",
            "StreamTemp",
            "StreamShadeDynamic",
        )
    ]
    + [
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "parameters_dis_seg.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [GEN_DIR_ST / f"{nn}.nc" for nn in ST_ANSWER_NAMES]
    + [GEN_DIR_ST / "humidity_hru.nc"]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr (incl. nhm_stream_temp) data not "
        "generated; missing: " + ", ".join(_missing[:3])
    ),
)


class HumidityCarrier(Process):
    """The humidity CBH forcing on the hru grid -- external data
    everywhere (no process produces it); the humidity aggregation Map
    carries it to seg_humid. Compute-free."""

    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH relative humidity [percent]",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time) -> None:
        pass


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR_ST / f"{nn}.nc")
        for nn in ST_ANSWER_NAMES
    }


@pytest.fixture(scope="module")
def model_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stream_temp_full_chain_output")

    def _params(name):
        return xr.load_dataset(DOMAIN_DIR / f"parameters_PRMS{name}.nc")

    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    soltabs = compute_soltabs(dis_hru)

    atmosphere_parameters = xr.merge(
        [
            _params("Atmosphere"),
            soltabs[["soltab_potsw", "soltab_horad_potsw"]],
        ]
    )
    snow_parameters = xr.merge(
        [_params("Snow"), soltabs[["soltab_horad_potsw"]]]
    )
    channel_parameters = _params("Channel")
    gw_parameters = _params("Groundwater")
    st = _params("StreamTemp")
    st_params = xr.merge(
        [
            st,
            _params("StreamShadeDynamic"),
            dis_seg[["lat_temp_adj"]],
            dis_hru[["hru_area"]],
        ],
        compat="no_conflicts",
    )

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
    seg_dis_params = discretizations["nsegment"].parameters
    assert seg_dis_params is not None

    # channel volume maps: 0/1 weights from hru_segment (as
    # test_prms_channel.py)
    hru_segment = channel_parameters["hru_segment"].values
    n_seg = channel_parameters.sizes["nsegment"]
    vol_weights = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            vol_weights[hru_segment[ihru] - 1, ihru] = 1.0

    # aggregation maps: kernel-probed weights, shared by reference
    agg_weights = derive_aggregation_weights(
        st["hru_segment"].values.astype(np.int64),
        dis_hru["hru_area"].values,
        dis_seg["tosegment"].values.astype(np.int64),
        seg_dis_params["segment_order"].values.astype(np.int64),
        st["seg_close"].values,
    )
    maps = {
        "sroff_vol": Map(
            weights=vol_weights,
            grid={"nhru": "nsegment"},
            variable={"sroff_vol": "seg_sroff_vol"},
        ),
        "ssres_vol": Map(
            weights=vol_weights,
            grid={"nhru": "nsegment"},
            variable={"ssres_flow_vol": "seg_ssres_flow_vol"},
        ),
        "gw_vol": Map(
            weights=vol_weights,
            grid={"nhru": "nsegment"},
            variable={"gwres_flow_vol": "seg_gwres_flow_vol"},
        ),
        **{
            target: Map(
                weights=agg_weights[wkey],
                grid={"nhru": "nsegment"},
                variable={source: target},
            )
            for target, (source, wkey) in AGGREGATION_MAP_SPEC.items()
        },
    }

    # dict order = schedule: NHM order on the hru grid, then channel ->
    # hydraulic geometry -> stream temp on the segment grid (each
    # consuming its producers' same-step shared buffers)
    process_dict = {
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
        },
        "prms_canopy": {
            "class": PRMSCanopy,
            "discretization": "nhru",
            "parameters": _params("Canopy"),
        },
        "prms_snow": {
            "class": PRMSSnow,
            "discretization": "nhru",
            "parameters": snow_parameters,
        },
        "prms_runoff": {
            "class": PRMSRunoff,
            "discretization": "nhru",
            "parameters": _params("Runoff"),
        },
        "prms_soilzone": {
            "class": PRMSSoilzone,
            "discretization": "nhru",
            "parameters": _params("Soilzone"),
        },
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
            "parameters": gw_parameters,
            "gwstor_init": gw_parameters["gwstor_init"],
        },
        "humidity_carrier": {
            "class": HumidityCarrier,
            "discretization": "nhru",
            "humidity_hru": xr.load_dataarray(
                GEN_DIR_ST / "humidity_hru.nc"
            ).rename({"nhm_id": "nhru"}),
        },
        "prms_channel": {
            "class": PRMSChannel,
            "discretization": "nsegment",
            "parameters": channel_parameters,
            "segment_flow_init": channel_parameters["segment_flow_init"],
        },
        "prms_hydraulic_geometry": {
            "class": PRMSHydraulicGeometryWidthOnly,
            "discretization": "nsegment",
            "parameters": DOMAIN_DIR
            / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
        },
        "prms_stream_temp": {
            "class": PRMSStreamTemp,
            "discretization": "nsegment",
            "parameters": st_params,
        },
    }
    control = {
        "output_var_names": list(ST_ANSWER_NAMES),
        "output_serial_zarr": out_dir / "stream_temp_full_chain.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict,
        control,
        maps=maps,
        discretizations=discretizations,
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return {"model": model, "control": control}


class TestPRMSStreamTempFullChain:
    def test_stream_temp_all_timesteps(self, model_run, answers):
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        fractions = {}
        for nn in ST_ANSWER_NAMES:
            actual = output_ds[nn].values
            desired = answers[nn].values
            finite = np.isfinite(actual)
            if nn == "seg_tave_water":
                nan_cols = np.isnan(actual).all(axis=0)
                assert (np.isnan(actual) == nan_cols[None, :]).all()
            bad = ~np.isclose(
                actual[finite], desired[finite], rtol=RTOL, atol=ATOL
            )
            fractions[nn] = bad.mean()
        print("outlier fractions at 5e-3:", fractions)
        for nn, frac in fractions.items():
            assert frac <= MAX_OUTLIER_FRAC, (
                f"variable '{nn}': {frac:.3%} of finite seg-days outside "
                f"{RTOL} (allowed {MAX_OUTLIER_FRAC:.3%})"
            )
