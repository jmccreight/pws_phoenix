"""Perfect-restart tests for the REAL processes (drb data).

The R2 completeness police for the ``restart=True`` flags: a
continuous run a->c must be BIT-IDENTICAL, in EVERY variable of EVERY
process (plus all python-attr restart state), to run a->b + write
followed by a fresh model warm-started b->c. A forgotten prognostic
flag (or missing hook state) diverges here, attributed to its
variable and process by the assertion message.

Coverage strategy (agreed R2 shape):

- THE FULL CHAIN test is the definitive one -- all nine NHM processes
  + the 13 Maps + the prior-step back-edges in one model; a missing
  flag in ANY process diverges its own variables at minimum.
- Standalone exemplars add isolation under simpler data: groundwater
  (simplest), snow (the heaviest state carrier), atmosphere (the
  transp_tindex state machine + istep0 first-step block), stream
  temperature (the python-attr silo hooks).

Flags were cross-checked against pywatershed's own hand-maintained
get_restart_variables() lists (gw/canopy/snow/runoff/soilzone/
channel/atmosphere); ours are supersets where OUR bitwise-everything
standard demands it (snow's season/albedo memory; atmosphere's
transp_check; soilzone's soil_lower). Runs are SHORT (correctness
does not need the full 731 days); inputs are the standalone tests'
disk forcings, served full-length with only the first n days run.
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
GEN_DIR = DOMAIN_DIR / "output"
GEN_DIR_ST = DOMAIN_DIR / "output_stream_temp"

pytestmark = pytest.mark.skipif(
    not (GEN_DIR / "gwres_flow.nc").exists()
    or not (GEN_DIR_ST / "seg_tave_water.nc").exists(),
    reason="pywatershed drb_2yr (incl. nhm_stream_temp) data not generated",
)


def _hru_forcing(name, gen_dir=GEN_DIR):
    return xr.load_dataarray(gen_dir / f"{name}.nc").rename({"nhm_id": "nhru"})


def perfect_restart(make_model, dt, n_total, restart_dir, idx_b=None):
    """Run the recipe and bit-compare EVERYTHING."""
    if idx_b is None:
        idx_b = n_total // 2
    with make_model({}) as m_ac:
        m_ac.run(dt, np.int32(n_total))
    with make_model({}) as m_ab:
        m_ab.run(dt, np.int32(idx_b))
        m_ab.write_restart(restart_dir)
    with make_model({"restart_read": restart_dir}) as m_bc:
        assert m_bc._start_index == idx_b
        m_bc.run(dt, np.int32(n_total - idx_b))

    for kk, proc_ac in m_ac.model_dict.items():
        proc_bc = m_bc.model_dict[kk]
        for nn in type(proc_ac).get_var_names():
            np.testing.assert_array_equal(
                proc_bc[nn].values,
                proc_ac[nn].values,
                err_msg=f"process '{kk}' variable '{nn}' not "
                "bit-identical after restart",
            )
        state_ac = proc_ac.get_restart_state()
        state_bc = proc_bc.get_restart_state()
        for key in state_ac:
            np.testing.assert_array_equal(
                state_bc[key],
                state_ac[key],
                err_msg=f"process '{kk}' restart state '{key}' not "
                "bit-identical after restart",
            )


# ----------------------------------------------------------------------
# standalone exemplars
# ----------------------------------------------------------------------


def test_groundwater(tmp_path):
    gw_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSGroundwater.nc")

    def make_model(control):
        process_dict = {
            "prms_groundwater": {
                "class": PRMSGroundwater,
                "discretization": "nhru",
                "parameters": gw_params,
                "gwstor_init": gw_params["gwstor_init"],
                **{
                    nn: _hru_forcing(nn)
                    for nn in ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")
                },
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(process_dict, control, discretizations=discretizations)

    perfect_restart(make_model, np.float64(1.0), 90, tmp_path / "rst")


def test_snow(tmp_path):
    params = xr.merge(
        [
            xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSnow.nc"),
            xr.load_dataarray(GEN_DIR / "soltab_horad_potsw.nc")
            .rename({"doy": "ndoy", "nhm_id": "nhru"})
            .to_dataset(name="soltab_horad_potsw"),
        ]
    )
    forcing_names = (
        "hru_ppt",
        "hru_intcpevap",
        "net_ppt",
        "net_rain",
        "net_snow",
        "orad_hru",
        "potet",
        "pptmix",
        "prmx",
        "swrad",
        "tavgc",
        "tmaxc",
        "tminc",
        "transp_on",
    )

    def make_model(control):
        process_dict = {
            "prms_snow": {
                "class": PRMSSnow,
                "discretization": "nhru",
                "parameters": params,
                **{nn: _hru_forcing(nn) for nn in forcing_names},
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(process_dict, control, discretizations=discretizations)

    # winter into spring: pack builds, knife edges exercised
    perfect_restart(make_model, np.float64(1.0), 180, tmp_path / "rst")


def test_atmosphere(tmp_path):
    params = xr.merge(
        [xr.load_dataset(DOMAIN_DIR / "parameters_PRMSAtmosphere.nc")]
        + [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc")
            .rename({"doy": "ndoy", "nhm_id": "nhru"})
            .to_dataset(name=nn)
            for nn in ("soltab_potsw", "soltab_horad_potsw")
        ]
    )

    def make_model(control):
        process_dict = {
            "prms_atmosphere": {
                "class": PRMSAtmosphere,
                "discretization": "nhru",
                "parameters": params,
                **{
                    nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
                    .rename({"nhm_id": "nhru"})
                    .astype(np.float64)
                    for nn in ("prcp", "tmax", "tmin")
                },
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(process_dict, control, discretizations=discretizations)

    # spans the istep0 transp initialization + season transitions
    perfect_restart(make_model, np.float64(1.0), 180, tmp_path / "rst")


def test_stream_temp(tmp_path):
    st_params = xr.merge(
        [
            xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc"),
            xr.load_dataset(
                DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
            ),
            xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")[
                ["lat_temp_adj"]
            ],
            xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")[
                ["hru_area"]
            ],
        ],
        compat="no_conflicts",
    )
    forcing_names = (
        "seg_outflow",
        "seg_lateral_inflow",
        "seg_flow_width",
        "seg_tave_air",
        "seg_humid",
        "seg_ccov",
        "seg_melt",
        "seg_rain",
        "seg_potet",
        "seginc_swrad",
        "seginc_sroff",
        "seginc_ssflow",
        "seginc_gwflow",
    )

    def make_model(control):
        process_dict = {
            "prms_stream_temp": {
                "class": PRMSStreamTemp,
                "discretization": "nsegment",
                "parameters": st_params,
                **{
                    nn: xr.load_dataarray(GEN_DIR_ST / f"{nn}.nc").rename(
                        {"nhm_seg": "nsegment"}
                    )
                    for nn in forcing_names
                },
            },
        }
        discretizations = {
            "nsegment": Discretization(
                ["nsegment"],
                parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
                topo_order={"segment_order": "tosegment"},
            ),
        }
        return Model(process_dict, control, discretizations=discretizations)

    # > 90 days so the gw/ss silos wrap real history through b
    perfect_restart(make_model, np.float64(1.0), 120, tmp_path / "rst")


# ----------------------------------------------------------------------
# THE full chain (all nine processes + 13 Maps + back-edges)
# ----------------------------------------------------------------------


class HumidityCarrier(Process):
    """The humidity CBH forcing (see the full-chain tests)."""

    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH relative humidity [percent]",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


def test_full_chain(tmp_path):
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    soltabs = compute_soltabs(dis_hru)

    def _params(name):
        return xr.load_dataset(DOMAIN_DIR / f"parameters_PRMS{name}.nc")

    atmos_params = xr.merge(
        [
            _params("Atmosphere"),
            soltabs[["soltab_potsw", "soltab_horad_potsw"]],
        ]
    )
    snow_params = xr.merge([_params("Snow"), soltabs[["soltab_horad_potsw"]]])
    channel_params = _params("Channel")
    gw_params = _params("Groundwater")
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
    cbh = {
        nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
        .rename({"nhm_id": "nhru"})
        .astype(np.float64)
        for nn in ("prcp", "tmax", "tmin")
    }
    humidity = _hru_forcing("humidity_hru", GEN_DIR_ST)

    hru_segment = channel_params["hru_segment"].values
    n_seg = channel_params.sizes["nsegment"]
    vol_weights = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            vol_weights[hru_segment[ihru] - 1, ihru] = 1.0

    def make_model(control):
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
        process_dict = {
            "prms_atmosphere": {
                "class": PRMSAtmosphere,
                "discretization": "nhru",
                "parameters": atmos_params,
                **cbh,
            },
            "prms_canopy": {
                "class": PRMSCanopy,
                "discretization": "nhru",
                "parameters": _params("Canopy"),
            },
            "prms_snow": {
                "class": PRMSSnow,
                "discretization": "nhru",
                "parameters": snow_params,
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
                "parameters": gw_params,
                "gwstor_init": gw_params["gwstor_init"],
            },
            "humidity_carrier": {
                "class": HumidityCarrier,
                "discretization": "nhru",
                "humidity_hru": humidity,
            },
            "prms_channel": {
                "class": PRMSChannel,
                "discretization": "nsegment",
                "parameters": channel_params,
                "segment_flow_init": channel_params["segment_flow_init"],
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
        return Model(
            process_dict,
            control,
            maps=maps,
            discretizations=discretizations,
        )

    perfect_restart(
        make_model,
        np.float64(60.0 * 60.0 * 24.0),
        60,
        tmp_path / "rst",
    )
