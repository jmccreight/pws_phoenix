"""
atmosphere/prms_solar_geometry.py
=================================
PRMSSolarGeometry, ported from pywatershed
(pywatershed/atmosphere/prms_solar_geometry.py; Swift 1976 / Lee 1963
potential solar radiation; PRMS 5.2.1).

Seventh REAL port (July 2026) -- deliberately NOT a framework Process.
Upstream's PRMSSolarGeometry has no inputs and no time evolution: it
computes three STATIC (ndoy, nhru) tables from three dis_hru variables
at initialization and serves per-doy rows forever after. In this
framework's taxonomy that is a PARAMETER DERIVATION, so the port is a
plain factory function:

    compute_soltabs(dis_hru_dataset) -> xr.Dataset with
        soltab_potsw       (ndoy, nhru)  [cal/cm^2 day]
        soltab_horad_potsw (ndoy, nhru)  [cal/cm^2 day]
        soltab_sunhrs      (ndoy, nhru)  [hours]

Callers merge the result into the parameters dataset of the consuming
processes (PRMSSnow, PRMSAtmosphere), which declare the tables as
(ndoy, space) parameters and index rows by current_doy -- exactly the
seam already used when the tables came from pywatershed's generated
files. This also sidesteps the mpixarray multi-dim derived-buffer
limit (see hydrology/prms_snow.py): under MPI the tables ride the
combined input file (or are computed per rank from the dis variables).

The numerics (compute_soltab / compute_t / func3) are verbatim
vectorized numpy -- init-time work, so the per-step in-place kernel
convention does not apply. The negative-radiation clamp keeps
upstream's warning.

Deliberately NOT ported: the Process/netcdf plumbing;
``from_prms_file`` (soltab_debug loading); the unused ``doy``/
``radj_sppt``/``radj_wppt``/``hru_area`` parameter declarations
(upstream declares them, its computation never uses them).
"""

import warnings

import numpy as np
import xarray as xr

from atmosphere.solar_constants import (
    NDOY,
    pi,
    pi_12,
    r1,
    solar_declination,
    two_pi,
)

# pywatershed constants
_DNEARZERO = 2.23e-16  # dnearzero = epsilon64 (hardcoded upstream)


def tile_space_to_time(arr: np.ndarray) -> np.ndarray:
    return np.tile(arr, (NDOY, 1))


def compute_t(lats: np.ndarray, solar_declination: np.ndarray) -> np.ndarray:
    """The "sunrise" equation: the hour angle from local solar noon to
    sunrise (negative) or sunset (positive), (ndoy, nhru)."""
    nhru = len(lats)
    lats_mat = np.tile(-1 * np.tan(lats), (NDOY, 1))
    sol_dec_mat = np.transpose(np.tile(np.tan(solar_declination), (nhru, 1)))
    tx = lats_mat * sol_dec_mat
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", r"invalid value encountered in arccos"
        )
        result = np.arccos(np.copy(tx))

    result[np.where(tx < -1.0)] = pi
    result[np.where(tx > 1.0)] = 0.0
    return result


def func3(
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Potential solar radiation on the surface [cal/cm^2 day]:
    radian-angle FUNC3 (eqn 6) from Swift 1976 / Lee 1963 eqn 5.

    v: hour offset between actual and equivalent slope (nhru,)
    w: latitude of the equivalent slope (nhru,)
    x: hour angle of sunset on equivalent slope (ndoy, nhru)
    y: hour angle of sunrise on equivalent slope (ndoy, nhru)
    """
    nhru = len(v)
    vv = np.tile(v, (NDOY, 1))
    ww = np.tile(w, (NDOY, 1))
    rr = np.transpose(np.tile(r1, (nhru, 1)))
    dd = np.transpose(np.tile(solar_declination, (nhru, 1)))

    f3 = (
        rr
        * pi_12
        * (
            np.sin(dd) * np.sin(ww) * (x - y)
            + np.cos(dd) * np.cos(ww) * (np.sin(x + vv) - np.sin(y + vv))
        )
    )
    return f3


def compute_soltab(
    slopes: np.ndarray,
    aspects: np.ndarray,
    lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Swift's potential solar radiation (solt) and hours of direct
    sunlight (sunh) on a sloping surface, both (ndoy, nhru) --
    pywatershed compute_soltab verbatim."""
    nhru = len(slopes)

    # slope / aspect / latitude derived quantities
    sl = np.arctan(slopes)
    sl_sin = np.sin(sl)
    sl_cos = np.cos(sl)
    aspects_rad = np.radians(aspects)
    aspects_cos = np.cos(aspects_rad)
    x0 = np.radians(lats)
    x0_cos = np.cos(x0)

    # x1: latitude of the equivalent slope (Lee 1963 eqn 13)
    x1 = np.arcsin(sl_cos * np.sin(x0) + sl_sin * x0_cos * aspects_cos)

    # d1: denominator of Lee 1963 eqn 12
    d1 = sl_cos * x0_cos - sl_sin * np.sin(x0) * aspects_cos
    d1 = np.where(np.abs(d1) < _DNEARZERO, _DNEARZERO, d1)

    # x2: longitude difference between the HRU and the equivalent
    # horizontal surface, in angle hours (Lee 1963 eqn 12)
    x2 = np.arctan(sl_sin * np.sin(aspects_rad) / d1)
    wh_d1_lt_zero = np.where(d1 < 0.0)
    if len(wh_d1_lt_zero[0]) > 0:
        x2[wh_d1_lt_zero] = x2[wh_d1_lt_zero] + pi

    # hour angles of sunrise (t6) / sunset (t7) on the equivalent slope
    tt = compute_t(x1, solar_declination)
    t6 = (-1 * tt) - x2
    t7 = tt - x2

    # hour angles of sunrise (t0) / sunset (t1) on a horizontal
    # surface at the HRU
    tt = compute_t(x0, solar_declination)
    t0 = -1 * tt
    t1 = tt

    # clamp equivalent-slope sunrise/sunset inside the horizontal ones
    t3 = t7
    wh_t7_gt_t1 = np.where(t7 > t1)
    if len(wh_t7_gt_t1[0]) > 0:
        t3[wh_t7_gt_t1] = t1[wh_t7_gt_t1]

    t2 = t6
    wh_t6_lt_t0 = np.where(t6 < t0)
    if len(wh_t6_lt_t0[0]) > 0:
        t2[wh_t6_lt_t0] = t0[wh_t6_lt_t0]

    t6 = t6 + two_pi
    t7 = t7 - two_pi
    wh_t3_lt_t2 = np.where(t3 < t2)
    if len(wh_t3_lt_t2[0]):
        t2[wh_t3_lt_t2] = 0.0
        t3[wh_t3_lt_t2] = 0.0

    # no other conditions met
    solt = func3(x2, x1, t3, t2)
    sunh = (t3 - t2) * pi_12

    wh_t7_gt_t0 = np.where(t7 > t0)
    if len(wh_t7_gt_t0[0]):
        solt[wh_t7_gt_t0] = (
            func3(x2, x1, t3, t2)[wh_t7_gt_t0]
            + func3(x2, x1, t7, t0)[wh_t7_gt_t0]
        )
        sunh[wh_t7_gt_t0] = (t3 - t2 + t7 - t0)[wh_t7_gt_t0] * pi_12

    wh_t6_lt_t1 = np.where(t6 < t1)
    if len(wh_t6_lt_t1[0]):
        solt[wh_t6_lt_t1] = (
            func3(x2, x1, t3, t2)[wh_t6_lt_t1]
            + func3(x2, x1, t1, t6)[wh_t6_lt_t1]
        )
        sunh[wh_t6_lt_t1] = (t3 - t2 + t1 - t6)[wh_t6_lt_t1] * pi_12

    # (near-)flat HRUs use the horizontal-surface values
    mask_sl_lt_dnearzero = tile_space_to_time(np.abs(sl)) < _DNEARZERO
    solt = np.where(
        mask_sl_lt_dnearzero, func3(np.zeros(nhru), x0, t1, t0), solt
    )
    sunh = np.where(mask_sl_lt_dnearzero, (t1 - t0) * pi_12, sunh)

    mask_sunh_lt_dnearzero = sunh < _DNEARZERO
    sunh = np.where(mask_sunh_lt_dnearzero, 0.0, sunh)

    wh_solt_lt_zero = np.where(solt < 0.0)
    if len(wh_solt_lt_zero[0]):
        solt[wh_solt_lt_zero] = 0.0
        warnings.warn(
            f"{len(wh_solt_lt_zero[0])}/{np.prod(solt.shape)} "
            "locations-times with negative potential solar radiation."
        )

    return solt, sunh


def compute_soltabs(dis_hru: xr.Dataset, hru_dim: str = "nhru") -> xr.Dataset:
    """The PRMSSolarGeometry product as a parameters dataset.

    Args:
        dis_hru: dataset providing ``hru_slope``, ``hru_aspect``,
            ``hru_lat`` on ``hru_dim``.
        hru_dim: the spatial dim name of the result (the grid dim).

    Returns:
        Dataset with ``soltab_potsw``, ``soltab_horad_potsw``,
        ``soltab_sunhrs`` on ("ndoy", hru_dim) -- merge into the
        consuming processes' parameters.
    """
    hru_slope = np.asarray(dis_hru["hru_slope"].values, dtype=np.float64)
    hru_aspect = np.asarray(dis_hru["hru_aspect"].values, dtype=np.float64)
    hru_lat = np.asarray(dis_hru["hru_lat"].values, dtype=np.float64)
    nhru = hru_slope.shape[0]

    # potential radiation on a horizontal surface
    soltab_horad_potsw, _ = compute_soltab(
        np.zeros(nhru), np.zeros(nhru), hru_lat
    )
    # potential radiation given slope and aspect
    soltab_potsw, soltab_sunhrs = compute_soltab(
        hru_slope, hru_aspect, hru_lat
    )

    dims = ("ndoy", hru_dim)
    return xr.Dataset(
        {
            "soltab_potsw": xr.DataArray(
                soltab_potsw,
                dims=dims,
                attrs={
                    "description": "Potential shortwave on the sloped "
                    "surface per Julian day [cal/cm^2 day]"
                },
            ),
            "soltab_horad_potsw": xr.DataArray(
                soltab_horad_potsw,
                dims=dims,
                attrs={
                    "description": "Potential shortwave on a horizontal "
                    "plane per Julian day [cal/cm^2 day]"
                },
            ),
            "soltab_sunhrs": xr.DataArray(
                soltab_sunhrs,
                dims=dims,
                attrs={
                    "description": "Hours of direct sunlight per Julian "
                    "day [hours]"
                },
            ),
        }
    )
