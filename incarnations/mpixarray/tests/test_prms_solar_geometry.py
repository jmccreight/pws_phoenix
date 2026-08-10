"""Regression: the ported solar-geometry factory vs pywatershed's
generated soltab tables (drb_2yr).

compute_soltabs (atmosphere/prms_solar_geometry.py -- a parameter
FACTORY, not a Process; see its docstring) is validated against the
three (doy, nhm_id) tables pywatershed generated with its own
PRMSSolarGeometry. Upstream's autotest compares against PRMS 5.2.1
binaries at 1e-10 (5e-5 for GSFLOW); against pywatershed's own
generated answers we hold that same 1e-10 (observed: the potsw tables
agree to 1e-12, soltab_sunhrs carries ~5e-11 float noise).

Requires GENERATED pywatershed test data; skips with a reason if
absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from atmosphere.prms_solar_geometry import compute_soltabs

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

TABLE_NAMES = ("soltab_potsw", "soltab_horad_potsw", "soltab_sunhrs")
RTOL = ATOL = 1.0e-10

_needed = [DOMAIN_DIR / "parameters_dis_hru.nc"] + [
    GEN_DIR / f"{nn}.nc" for nn in TABLE_NAMES
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


class TestSolarGeometry:
    def test_tables_match_pywatershed(self):
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        tables = compute_soltabs(dis_hru)
        for nn in TABLE_NAMES:
            answer = xr.load_dataarray(GEN_DIR / f"{nn}.nc")
            assert tables[nn].shape == answer.shape  # (366, 765)
            np.testing.assert_allclose(
                tables[nn].values,
                answer.values,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"table '{nn}' differs from pywatershed",
            )
