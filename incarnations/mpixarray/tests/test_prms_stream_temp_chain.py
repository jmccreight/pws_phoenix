"""Serial live-Maps chain: hru -> aggregation Maps -> stream temp.

Stage 3 of the stream-temp chain: the first LIVE two-grid stream-temp
model. The hru grid runs PRMSAtmosphere (live tavgc / hru_rain /
swrad / potet / ccov_hru from raw CBH) plus a disk-fed carrier for
the variables whose producers are not in this model (sroff /
ssres_flow / gwres_flow from runoff/soilzone/gw, snowmelt from snow,
humidity_hru = the CBH forcing) -- the retired-carrier pattern,
resurrected deliberately to keep this test tight; the full
7-process chain is stage 4. The TEN aggregation Maps (weights from
derive_aggregation_weights(), the three matrices shared by reference
across the Maps; wiring from AGGREGATION_MAP_SPEC) carry them to the
segment grid running PRMSHydraulicGeometryWidthOnly -> PRMSStreamTemp
(LIVE seg_flow_width; seg_outflow / seg_lateral_inflow from disk).

Because humidity rides the Maps from the CBH humidity_hru forcing,
this model IS the strmtemp_humidity_flag = 0 (CBH) configuration --
the core PRMSStreamTemp class, no humidity leaf.

Validation vs the drb nhm_stream_temp answers: stream-temp variables
at the family 5e-3 (the live atmosphere inputs carry its 1e-5 floor,
far inside); hydraulic geometry at its 1e-5. Mapped aggregates are
INPUTS on the segment grid (not Output-writable), so they are checked
directly from the final step's in-memory buffers at 1e-5 -- with the
weights-vs-kernels pin (test_prms_stream_temp_aggregates.py) that
closes the wiring: weights == kernels == Fortran.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from atmosphere.prms_atmosphere import PRMSAtmosphere
from discretization import Discretization
from hydrology.prms_hydraulic_geometry import PRMSHydraulicGeometryWidthOnly
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
GEN_DIR = DOMAIN_DIR / "output_stream_temp"

CBH_NAMES = ("prcp", "tmax", "tmin")
CARRIER_NAMES = (
    "sroff",
    "ssres_flow",
    "gwres_flow",
    "snowmelt",
    "humidity_hru",
)
SEG_INPUT_NAMES = ("seg_outflow", "seg_lateral_inflow")
HG_ANSWER_NAMES = ("seg_flow_width", "seg_res_time")
ST_ANSWER_NAMES = (
    "seg_tave_water",
    "seg_tave_upstream",
    "seg_tave_gw",
    "seg_tave_ss",
    "seg_tave_lat",
    "seg_shade",
)
AGG_NAMES = tuple(AGGREGATION_MAP_SPEC)
RTOL_ST = ATOL_ST = 5.0e-3
RTOL_HG = ATOL_HG = 1.0e-5
RTOL_AGG = ATOL_AGG = 1.0e-5  # the live atmosphere's precision floor

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSAtmosphere.nc",
        DOMAIN_DIR / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
        DOMAIN_DIR / "parameters_PRMSStreamTemp.nc",
        DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "parameters_dis_seg.nc",
        GEN_DIR / "soltab_potsw.nc",
        GEN_DIR / "soltab_horad_potsw.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [
        GEN_DIR / f"{nn}.nc"
        for nn in CARRIER_NAMES
        + SEG_INPUT_NAMES
        + HG_ANSWER_NAMES
        + ST_ANSWER_NAMES
        + AGG_NAMES
    ]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_stream_temp data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


def _carrier_meta(description):
    return DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description=description,
    )


class HruCarrier(Process):
    """Disk-fed hru variables for the aggregation Maps whose live
    producers (runoff/soilzone/gw/snow) are not in this model;
    humidity_hru is a CBH forcing everywhere. Compute-free."""

    sroff = _carrier_meta("Surface runoff [inches] (PRMSRunoff stand-in)")
    ssres_flow = _carrier_meta(
        "Subsurface flow [inches] (PRMSSoilzone stand-in)"
    )
    gwres_flow = _carrier_meta(
        "Groundwater flow [inches] (PRMSGroundwater stand-in)"
    )
    snowmelt = _carrier_meta("Snowmelt [inches] (PRMSSnow stand-in)")
    humidity_hru = _carrier_meta("CBH relative humidity [percent]")

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time) -> None:
        pass


@pytest.fixture(scope="module")
def answers():
    names = HG_ANSWER_NAMES + ST_ANSWER_NAMES + AGG_NAMES
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def model_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stream_temp_chain_output")

    # -- hru side: atmosphere parameters + CBH; carrier forcings --
    atmos_params = xr.merge(
        [xr.load_dataset(DOMAIN_DIR / "parameters_PRMSAtmosphere.nc")]
        + [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc")
            .rename({"doy": "ndoy", "nhm_id": "nhru"})
            .to_dataset(name=nn)
            for nn in ("soltab_potsw", "soltab_horad_potsw")
        ]
    )
    cbh = {
        nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
        .rename({"nhm_id": "nhru"})
        .astype(np.float64)
        for nn in CBH_NAMES
    }
    carrier = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
        for nn in CARRIER_NAMES
    }

    # -- segment side --
    st = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc")
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    st_params = xr.merge(
        [
            st,
            xr.load_dataset(
                DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
            ),
            dis_seg[["lat_temp_adj"]],
            dis_hru[["hru_area"]],
        ],
        compat="no_conflicts",
    )
    seg_forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_seg": "nsegment"}
        )
        for nn in SEG_INPUT_NAMES
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
    seg_dis_params = discretizations["nsegment"].parameters
    assert seg_dis_params is not None

    # -- the ten aggregation Maps: three weight matrices shared by
    # reference across the Maps (channel precedent) --
    weights = derive_aggregation_weights(
        st["hru_segment"].values.astype(np.int64),
        dis_hru["hru_area"].values,
        dis_seg["tosegment"].values.astype(np.int64),
        seg_dis_params["segment_order"].values.astype(np.int64),
        st["seg_close"].values,
    )
    maps = {
        target: Map(
            weights=weights[wkey],
            grid={"nhru": "nsegment"},
            variable={source: target},
        )
        for target, (source, wkey) in AGGREGATION_MAP_SPEC.items()
    }

    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphere,
            "discretization": "nhru",
            "parameters": atmos_params,
            **cbh,
        },
        "hru_carrier": {
            "class": HruCarrier,
            "discretization": "nhru",
            **carrier,
        },
        "prms_hydraulic_geometry": {
            "class": PRMSHydraulicGeometryWidthOnly,
            "discretization": "nsegment",
            "parameters": DOMAIN_DIR
            / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
            "seg_outflow": seg_forcings.pop("seg_outflow"),
        },
        "prms_stream_temp": {
            "class": PRMSStreamTemp,
            "discretization": "nsegment",
            "parameters": st_params,
            **seg_forcings,
        },
    }
    control = {
        "output_var_names": list(HG_ANSWER_NAMES + ST_ANSWER_NAMES),
        "output_serial_zarr": out_dir / "stream_temp_chain.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict,
        control,
        maps=maps,
        discretizations=discretizations,
    ) as model:
        model.run(np.float64(1.0), np.int32(model.ntime))
    return {"model": model, "control": control}


class TestPRMSStreamTempChain:
    def test_segment_outputs_all_timesteps(self, model_run, answers):
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in HG_ANSWER_NAMES + ST_ANSWER_NAMES:
            actual = output_ds[nn].values
            desired = answers[nn].values
            finite = np.isfinite(actual)
            if nn == "seg_tave_water":
                nan_cols = np.isnan(actual).all(axis=0)
                assert (np.isnan(actual) == nan_cols[None, :]).all()
            if nn in HG_ANSWER_NAMES:
                rtol, atol = RTOL_HG, ATOL_HG
            else:
                rtol, atol = RTOL_ST, ATOL_ST
            np.testing.assert_allclose(
                actual[finite],
                desired[finite],
                rtol=rtol,
                atol=atol,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_final_step_aggregates(self, model_run, answers):
        """The mapped aggregates are segment-grid INPUTS (Map target
        buffers; not Output-writable) -- checked at the final step from
        the in-memory grid dataset. All-timestep aggregate parity is
        carried by the weights-vs-kernels pin plus the Fortran pin in
        test_prms_stream_temp_aggregates.py."""
        seg_ds = model_run["model"].discretizations["nsegment"].dataset
        for nn in AGG_NAMES:
            np.testing.assert_allclose(
                seg_ds[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL_AGG,
                atol=ATOL_AGG,
                err_msg=f"final-step aggregate '{nn}' differs",
            )
