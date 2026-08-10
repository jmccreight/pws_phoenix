"""Standalone regression: the hru->segment aggregation vs pywatershed.

Two pins on the chain seam, in isolation, before it is wired into a
live model:

1. KERNELS vs FORTRAN: drives the verbatim-extracted aggregation
   kernels (_compute_segment_aggregates_numba + _compute_seg_potet_numba
   + _compute_seg_humid_cbh_numba) with resolve_aggregation_topology
   over the full drb_2yr nhm_stream_temp period, feeding the per-day
   HRU inputs from the generated answers, and compares every segment
   aggregate against the Fortran answers (incl. the seg_close fallback
   machinery, live on drb's 40 no-HRU segments). Tolerance 1e-5
   (linear area-weighted sums).

2. WEIGHTS vs KERNELS: derive_aggregation_weights() probes the same
   kernels with basis vectors to produce the static weight matrices
   the live chain's Maps will carry; W @ x must reproduce the kernel
   outputs over all days to matmul reassociation noise (1e-12). For
   seg_ccov the source is the RELOCATED ccov_hru (PRMSAtmosphere's
   per-HRU cloud cover, reproduced here by its reference formula from
   swrad/soltab/cossl) aggregated with the "met" weights -- pinning
   the relocation's equivalence to the kernel's in-loop ccov path.

One caveat by construction: the auto-seg_close route-order fallback
("previous segment in route order") depends on the SPECIFIC
topological order, and drb has exactly one no-HRU segment without an
upstream. If that segment's copied met values mismatch under our dis
ordering, it is excluded with a documented mask (checked and reported
by the test).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    _compute_seg_humid_cbh_numba,
    _compute_seg_potet_numba,
    _compute_segment_aggregates_numba,
    derive_aggregation_weights,
    resolve_aggregation_topology,
)

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output_stream_temp"

HRU_INPUT_NAMES = (
    "sroff",
    "ssres_flow",
    "gwres_flow",
    "swrad",
    "tavgc",
    "snowmelt",
    "hru_rain",
    "potet",
    "humidity_hru",
)
SEG_ANSWER_NAMES = (
    "seginc_sroff",
    "seginc_ssflow",
    "seginc_gwflow",
    "seginc_swrad",
    "seg_tave_air",
    "seg_ccov",
    "seg_melt",
    "seg_rain",
    "seg_potet",
    "seg_humid",
)
RTOL = ATOL = 1.0e-5  # vs Fortran answers
RTOL_W = ATOL_W = 1.0e-12  # weights vs kernels (matmul reassociation)

_needed = [
    DOMAIN_DIR / "parameters_PRMSStreamTemp.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    GEN_DIR / "soltab_potsw.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in HRU_INPUT_NAMES + SEG_ANSWER_NAMES]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_stream_temp data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def data():
    """Topology arrays, per-day HRU inputs, Fortran answers, soltab."""
    st = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc")
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")

    dis = Discretization(
        ["nsegment"],
        parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
        topo_order={"segment_order": "tosegment"},
    )
    assert dis.parameters is not None

    times = xr.load_dataarray(GEN_DIR / "sroff.nc")["time"].values.astype(
        "datetime64[D]"
    )
    doys = (
        times - times.astype("datetime64[Y]").astype("datetime64[D]")
    ).astype(int) + 1

    return {
        "hru_segment": st["hru_segment"].values.astype(np.int64),
        "hru_area": dis_hru["hru_area"].values,
        "tosegment": dis_seg["tosegment"].values.astype(np.int64),
        "hru_cossl": np.cos(np.arctan(dis_hru["hru_slope"].values)),
        "segment_order": dis.parameters["segment_order"].values.astype(
            np.int64
        ),
        "seg_close_param": st["seg_close"].values,
        "inputs": {
            nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").values
            for nn in HRU_INPUT_NAMES
        },
        "answers": {
            nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").values
            for nn in SEG_ANSWER_NAMES
        },
        "soltab_potsw": xr.load_dataarray(
            GEN_DIR / "soltab_potsw.nc"
        ).values,  # (ndoy, nhru)
        "doys": doys,
    }


@pytest.fixture(scope="module")
def kernel_out(data):
    """The verbatim kernels driven over every day of the run."""
    hru_segment = data["hru_segment"]
    hru_area = data["hru_area"]
    tosegment = data["tosegment"]
    segment_order = data["segment_order"]
    inputs = data["inputs"]
    doys = data["doys"]
    nhru = hru_area.shape[0]
    nsegment = tosegment.shape[0]

    topo = resolve_aggregation_topology(
        hru_segment,
        hru_area,
        tosegment,
        segment_order,
        data["seg_close_param"],
    )

    ntime = doys.shape[0]
    out = {nn: np.zeros((ntime, nsegment)) for nn in SEG_ANSWER_NAMES}
    seginc_sroff = np.zeros(nsegment)
    seginc_ssflow = np.zeros(nsegment)
    seginc_gwflow = np.zeros(nsegment)
    seginc_swrad = np.zeros(nsegment)
    seg_tave_air = np.zeros(nsegment)
    seg_melt = np.zeros(nsegment)
    seg_rain = np.zeros(nsegment)
    seg_ccov = np.zeros(nsegment)
    seg_potet = np.zeros(nsegment)
    seg_humid = np.zeros(nsegment)

    for tt in range(ntime):
        _compute_segment_aggregates_numba(
            nhru,
            nsegment,
            hru_segment,
            hru_area,
            inputs["sroff"][tt, :],
            inputs["ssres_flow"][tt, :],
            inputs["gwres_flow"][tt, :],
            inputs["swrad"][tt, :],
            topo["segment_hruarea"],
            topo["segment_up"],
            tosegment,
            seginc_sroff,
            seginc_ssflow,
            seginc_gwflow,
            seginc_swrad,
            inputs["tavgc"][tt, :],
            inputs["snowmelt"][tt, :],
            inputs["hru_rain"][tt, :],
            data["soltab_potsw"][doys[tt] - 1, :],
            data["hru_cossl"],
            segment_order,
            topo["seg_close"],
            seg_tave_air,
            seg_melt,
            seg_rain,
            seg_ccov,
        )
        _compute_seg_potet_numba(
            nhru,
            nsegment,
            hru_segment,
            hru_area,
            inputs["potet"][tt, :],
            segment_order,
            topo["segment_hruarea"],
            topo["seg_close"],
            seg_potet,
        )
        _compute_seg_humid_cbh_numba(
            nhru,
            nsegment,
            hru_segment,
            hru_area,
            inputs["humidity_hru"][tt, :],
            topo["segment_hruarea"],
            segment_order,
            topo["seg_close"],
            seg_humid,
        )
        out["seginc_sroff"][tt] = seginc_sroff
        out["seginc_ssflow"][tt] = seginc_ssflow
        out["seginc_gwflow"][tt] = seginc_gwflow
        out["seginc_swrad"][tt] = seginc_swrad
        out["seg_tave_air"][tt] = seg_tave_air
        out["seg_ccov"][tt] = seg_ccov
        out["seg_melt"][tt] = seg_melt
        out["seg_rain"][tt] = seg_rain
        out["seg_potet"][tt] = seg_potet
        out["seg_humid"][tt] = seg_humid
    return out


def test_segment_aggregates_all_days(data, kernel_out):
    for nn in SEG_ANSWER_NAMES:
        np.testing.assert_allclose(
            kernel_out[nn],
            data["answers"][nn],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"aggregate '{nn}' differs from pywatershed",
        )


def test_weights_reproduce_kernels(data, kernel_out):
    """W @ x == kernel(x) for all ten aggregates over all days (to
    matmul reassociation noise). Also pins that drb has no -99.9
    marker segments (derive raises otherwise) and that the relocated
    ccov_hru path is equivalent to the kernel's in-loop ccov."""
    weights = derive_aggregation_weights(
        data["hru_segment"],
        data["hru_area"],
        data["tosegment"],
        data["segment_order"],
        data["seg_close_param"],
    )

    # ccov_hru: PRMSAtmosphere's reference formula (the relocated
    # verbatim block) applied to the answers' swrad
    swrad = data["inputs"]["swrad"]
    potsw = data["soltab_potsw"][data["doys"] - 1, :]
    cossl = data["hru_cossl"]
    ccov_hru = np.where(
        potsw <= 10.0,
        1.0 - swrad / 10.0 * cossl,
        1.0 - swrad / potsw * cossl,
    )
    ccov_hru = np.where(ccov_hru < 1.0e-6, 0.0, np.minimum(ccov_hru, 1.0))
    sources = dict(data["inputs"])
    sources["ccov_hru"] = ccov_hru

    for target, (source, wkey) in AGGREGATION_MAP_SPEC.items():
        mapped = sources[source] @ weights[wkey].T  # (ntime, nsegment)
        np.testing.assert_allclose(
            mapped,
            kernel_out[target],
            rtol=RTOL_W,
            atol=ATOL_W,
            err_msg=(
                f"weights '{wkey}' @ '{source}' does not reproduce "
                f"the kernel's '{target}'"
            ),
        )
