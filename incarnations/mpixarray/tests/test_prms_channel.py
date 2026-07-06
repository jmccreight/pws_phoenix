"""Serial submodel regression: PRMSGroundwater -> Maps -> PRMSChannel
vs pywatershed answers (drb_2yr).

The Stage-2 shape with real physics: groundwater computes
gwres_flow_vol LIVE on the hru grid; sroff_vol / ssres_flow_vol come
from disk, hosted on the hru grid by a carrier process (a stand-in for
the not-yet-ported PRMSRunoff / PRMSSoilzone -- shaped exactly like its
future replacements); three Maps (same 0/1 weights from hru_segment, by
reference) aggregate the VOLUMES to the segment grid; the channel sums
them in-kernel (seg_lateral_inflow) and routes muskingum_mann.

dt is SECONDS (86400.0); groundwater never reads dt.

Tolerance: pywatershed's own standard (rtol = atol = 1e-13). The
map-then-sum float-order deviation (pywatershed sums (a+b+c) per HRU
then aggregates; we aggregate each flux then sum) proved BENIGN: all
flow variables match at 1e-13. The one exception is seg_stor_change --
a difference of near-equal numbers, where cancellation amplifies the
residue (see PER_VAR_TOL).

Requires GENERATED pywatershed test data; skips with a reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import PRMSGroundwater
from map import Map
from model import Model
from process import DataArrayMeta, Process

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

GW_INPUT_NAMES = ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")
CARRIER_INPUT_NAMES = ("sroff_vol", "ssres_flow_vol")
ANSWER_NAMES = (
    "seg_lateral_inflow",
    "seg_upstream_inflow",
    "seg_inflow",
    "seg_outflow",
    "seg_stor_change",
    "channel_outflow_vol",
)
# pywatershed's own autotest comparison standard
RTOL = ATOL = 1.0e-13
# seg_stor_change = (seg_inflow - seg_outflow) * s_per_time: a
# DIFFERENCE of near-equal numbers. Both operands match pywatershed at
# 1e-13 (validated above it in ANSWER_NAMES), but cancellation strips
# the leading digits and amplifies the benign mapped-sum float-order
# residue (observed: 1.9e-8 rel / 3.8e-6 abs) -- 1e-13 is
# mathematically unattainable for this diagnostic.
PER_VAR_TOL = {"seg_stor_change": (1.0e-7, 1.0e-4)}  # (rtol, atol)
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = [
    DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
    DOMAIN_DIR / "parameters_PRMSChannel.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in GW_INPUT_NAMES + CARRIER_INPUT_NAMES + ANSWER_NAMES
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


class HruChannelFluxes(Process):
    """Carrier for not-yet-ported producers (PRMSRunoff/PRMSSoilzone):
    hosts their disk-recorded VOLUME outputs on the hru grid so the
    hru->segment Maps have a source. No computation."""

    sroff_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff volume [cf] (from disk)",
    )
    ssres_flow_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Interflow volume [cf] (from disk)",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time) -> None:
        pass


def _open_hru_forcing(name):
    """pywatershed output files put variables on 'nhm_id'; rename to the
    hru grid dim."""
    return xr.open_dataarray(GEN_DIR / f"{name}.nc").rename({"nhm_id": "nhru"})


@pytest.fixture(scope="module")
def channel_parameters():
    return xr.open_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")


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
    return {nn: xr.open_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module")
def model_run(channel_parameters, weights, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("prms_channel_output")
    gw_parameters = xr.open_dataset(
        DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
    )

    process_dict = {
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
            "parameters": gw_parameters,
            "gwstor_init": gw_parameters["gwstor_init"],
            **{nn: _open_hru_forcing(nn) for nn in GW_INPUT_NAMES},
        },
        "hru_channel_fluxes": {
            "class": HruChannelFluxes,
            "discretization": "nhru",
            **{nn: _open_hru_forcing(nn) for nn in CARRIER_INPUT_NAMES},
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
    return {"model": model, "control": control}


class TestPRMSChannelSubmodel:
    def test_all_variables_all_timesteps(self, model_run, answers):
        """Every channel output matches pywatershed over the full run."""
        output_ds = xr.open_zarr(
            model_run["control"]["output_serial_zarr"], consolidated=False
        )
        for nn in ANSWER_NAMES:
            rtol_var, atol_var = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            np.testing.assert_allclose(
                output_ds[nn].values,
                answers[nn].values,
                rtol=rtol_var,
                atol=atol_var,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_final_state(self, model_run, answers):
        proc = model_run["model"].model_dict["prms_channel"]
        for nn in ("seg_outflow", "seg_inflow"):
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
            )

    def test_derived_parameters_frozen(self, model_run):
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
