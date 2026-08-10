"""Serial regression: PRMSStreamTempSegHumidity vs pywatershed.

BOTH drb_2yr strmtemp_humidity_flag = 1 configurations, parametrized:
nhm_stream_temp_seg_humid_matrix (a full (nmonth, nsegment) myparam
matrix) and nhm_stream_temp_seg_humid_scalar (a uniform value) --
upstream they are the SAME code path (humidity from the monthly
``seg_humidity`` PARAMETER), so both are served by the one leaf: it
overrides the core's ``seg_humid`` input declaration to a computed
variable (declaration override, input -> variable) and assigns it per
step. All other HRU-derived aggregates are fed from each
configuration's own answers, as in test_prms_stream_temp.py. Compared
at the stream-temp family standard (5e-3), including seg_humid itself
(exact modulo storage).

Requires drb_2yr with the GENERATED answers per configuration; each
parametrization skips independently with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_stream_temp import PRMSStreamTempSegHumidity
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
CONFIGS = {
    "matrix": DOMAIN_DIR / "output_stream_temp_seg_humid_matrix",
    "scalar": DOMAIN_DIR / "output_stream_temp_seg_humid_scalar",
}

INPUT_NAMES = (
    "seg_outflow",
    "seg_lateral_inflow",
    "seg_flow_width",
    "seg_tave_air",
    "seg_ccov",
    "seg_melt",
    "seg_rain",
    "seg_potet",
    "seginc_swrad",
    "seginc_sroff",
    "seginc_ssflow",
    "seginc_gwflow",
)
ANSWER_NAMES = (
    "seg_tave_water",
    "seg_tave_upstream",
    "seg_tave_gw",
    "seg_tave_ss",
    "seg_tave_lat",
    "seg_shade",
    "seg_humid",
)
RTOL = ATOL = 5.0e-3

_needed = [
    DOMAIN_DIR / "parameters_PRMSStreamTemp.nc",
    DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr parameter/dis files absent; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def gen_dir(request):
    """The configuration's answer directory; skips THIS parametrization
    if its answers are not generated."""
    gg = CONFIGS[request.param]
    files_missing = [
        str(gg / f"{nn}.nc")
        for nn in INPUT_NAMES + ANSWER_NAMES
        if not (gg / f"{nn}.nc").exists()
    ]
    if files_missing:
        pytest.skip(
            f"nhm_stream_temp_seg_humid_{request.param} answers not "
            "generated; missing: " + ", ".join(files_missing[:3])
        )
    return gg


@pytest.fixture(scope="module")
def parameters(gen_dir):
    st = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc")
    shade = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
    )
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")

    # each configuration's seg_humidity differs from the (uniform)
    # values in parameters_PRMSStreamTemp.nc (extracted from another
    # config's myparam); reconstruct the monthly parameter EXACTLY
    # from the seg_humid answers (constant within each month)
    seg_humid_ans = xr.load_dataarray(gen_dir / "seg_humid.nc")
    months = seg_humid_ans["time"].dt.month.values
    seg_humidity = np.empty((12, seg_humid_ans.shape[1]), dtype=np.float64)
    for mm in range(1, 13):
        seg_humidity[mm - 1, :] = seg_humid_ans.values[
            np.where(months == mm)[0][0], :
        ]
    st = st.drop_vars("seg_humidity")
    st["seg_humidity"] = (("nmonth", "nsegment"), seg_humidity)

    return xr.merge(
        [st, shade, dis_seg[["lat_temp_adj"]], dis_hru[["hru_area"]]],
        compat="no_conflicts",
    )


@pytest.fixture(scope="module")
def answers(gen_dir):
    return {nn: xr.load_dataarray(gen_dir / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module")
def model_run(gen_dir, parameters, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stream_temp_seg_humidity_output")
    forcings = {
        nn: xr.load_dataarray(gen_dir / f"{nn}.nc").rename(
            {"nhm_seg": "nsegment"}
        )
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_stream_temp": {
            "class": PRMSStreamTempSegHumidity,
            "discretization": "nsegment",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "stream_temp_seg_humidity.zarr",
        "time_chunk_size": 61,
    }
    discretizations = {
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(np.float64(1.0), np.int32(model.ntime))
    return {"model": model, "control": control}


class TestPRMSStreamTempSegHumidity:
    def test_all_variables_all_timesteps(self, model_run, answers):
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in ANSWER_NAMES:
            actual = output_ds[nn].values
            desired = answers[nn].values
            finite = np.isfinite(actual)
            if nn == "seg_tave_water":
                nan_cols = np.isnan(actual).all(axis=0)
                assert (np.isnan(actual) == nan_cols[None, :]).all()
            np.testing.assert_allclose(
                actual[finite],
                desired[finite],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )
