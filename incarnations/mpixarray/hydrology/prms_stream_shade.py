"""
hydrology/prms_stream_shade.py
==============================
Stream shade computation for PRMSStreamTemp, ported from pywatershed
(pywatershed/hydrology/prms_stream_shade.py; PRMS 5.2.1 stream_temp
shday algorithm). Stage 2 of the stream-temperature arc.

Upstream structures this as strategy OBJECTS (PRMSStreamShadeConstant
/ PRMSStreamShadeDynamic) composed into its stream-temperature
Process; the actual physics is a set of scalar njit functions
(_shday + _solalt + _snr_sst + _rprnvg + the 15-point Gaussian
quadrature tables). This module carries THOSE, extracted VERBATIM
from upstream (only the parallel=True decorator dropped -- no
threaded numba here, prange degrades to range) -- our stream-temp
leaves call them directly, the same seam as the shared soilzone /
runoff / starfit helpers (no strategy-object machinery needed).

Modes (PRMS stream_temp_shade_flag): 0 = DYNAMIC (topography +
vegetation via _shday; drb nhm_stream_temp runs this), 1 = CONSTANT
(seasonal segshade_sum/segshade_win; trivial selection, implemented
by the stream-temp leaf).

Validated standalone against the generated seg_shade answers
(tests/test_prms_stream_shade.py) before stream temperature consumes
it.
"""

import math

import numba as nb
import numpy as np

# Constants from PRMS
NEARZERO = 1e-6
PI = np.pi
HALF_PI = PI / 2.0
RADTOHOUR = 24.0 / (2.0 * PI)

# Gaussian quadrature points and weights (15-point)
EPSLON = np.array(
    [
        0.006003741,
        0.031363304,
        0.075896109,
        0.137791135,
        0.214513914,
        0.302924330,
        0.399402954,
        0.500000000,
        0.600597047,
        0.697075674,
        0.785486087,
        0.862208866,
        0.924103292,
        0.968636696,
        0.993996259,
    ]
)

WEIGHT = np.array(
    [
        0.015376621,
        0.035183024,
        0.053579610,
        0.069785339,
        0.083134603,
        0.093080500,
        0.099215743,
        0.101289120,
        0.099215743,
        0.093080500,
        0.083134603,
        0.069785339,
        0.053579610,
        0.035183024,
        0.015376621,
    ]
)


@nb.jit(nopython=True)  # upstream: parallel=True (not used here)
def _shday_vectorized(
    seg_lat_rads,
    declination,
    seg_widths,
    azrhs,
    altes,
    altws,
    vces,
    voes,
    vhes,
    vdemxs,
    vdemns,
    summer_flag,
    vcws,
    vows,
    vhws,
    vdwmxs,
    vdwmns,
    maxiter_sntemp,
):
    """Vectorized shade computation for all segments using parallel numba.

    All input arrays should have length nsegment.

    Args:
        seg_lat_rads: Segment latitudes in radians (array)
        declination: Solar declination (radians)
        seg_widths: Flow widths for all segments (meters, array)
        azrhs: Stream azimuth angles (radians, array)
        altes: East bank topographic altitudes (radians, array)
        altws: West bank topographic altitudes (radians, array)
        vces: East bank vegetation crown widths (meters, array)
        voes: East bank vegetation offsets (meters, array)
        vhes: East bank vegetation heights (meters, array)
        vdemxs: Maximum east bank vegetation densities (array)
        vdemns: Minimum east bank vegetation densities (array)
        summer_flag: 1 for summer, 0 for winter (scalar)
        vcws: West bank vegetation crown widths (meters, array)
        vows: West bank vegetation offsets (meters, array)
        vhws: West bank vegetation heights (meters, array)
        vdwmxs: Maximum west bank vegetation densities (array)
        vdwmns: Minimum west bank vegetation densities (array)
        maxiter_sntemp: Maximum iterations for convergence (scalar)

    Returns:
        shades: Array of shade fractions (0-1) for all segments
        svis: Array of vegetation shade indices for all segments
    """
    nseg = len(seg_lat_rads)
    shades = np.zeros(nseg, dtype=np.float64)
    svis = np.zeros(nseg, dtype=np.float64)

    for i in nb.prange(nseg):
        shades[i], svis[i] = _shday(
            seg_lat_rads[i],
            declination,
            seg_widths[i],
            azrhs[i],
            altes[i],
            altws[i],
            vces[i],
            voes[i],
            vhes[i],
            vdemxs[i],
            vdemns[i],
            summer_flag,
            vcws[i],
            vows[i],
            vhws[i],
            vdwmxs[i],
            vdwmns[i],
            maxiter_sntemp,
        )

    return shades, svis


@nb.jit(nopython=True)
def _shday(
    seg_lat,
    declination,
    seg_width,
    azrh,
    alte,
    altw,
    vce,
    voe,
    vhe,
    vdemx,
    vdemn,
    summer_flag,
    vcw,
    vow,
    vhw,
    vdwmx,
    vdwmn,
    maxiter_sntemp,
):
    """Compute daily shade from topography and vegetation.

    This is the shday function from PRMS.

    Args:
        seg_lat: Segment latitude (radians)
        declination: Solar declination (radians)
        seg_width: Flow width for the segment (meters)
        azrh: Stream azimuth angle (radians)
        alte: East bank topographic altitude (radians)
        altw: West bank topographic altitude (radians)
        vce: East bank vegetation crown width (meters)
        voe: East bank vegetation offset (meters)
        vhe: East bank vegetation height (meters)
        vdemx: Maximum east bank vegetation density
        vdemn: Minimum east bank vegetation density
        summer_flag: 1 for summer, 0 for winter
        vcw: West bank vegetation crown width (meters)
        vow: West bank vegetation offset (meters)
        vhw: West bank vegetation height (meters)
        vdwmx: Maximum west bank vegetation density
        vdwmn: Minimum west bank vegetation density
        maxiter_sntemp: Maximum iterations for convergence

    Returns:
        shade: Shade fraction (0-1)
        svi: Vegetation shade index
    """
    if seg_width <= 0.0:
        return 0.0, 0.0

    coso = math.cos(seg_lat)
    if coso == 0.0:
        coso = NEARZERO

    sino = math.sin(seg_lat)
    sin_d = math.sin(declination)
    cos_d = math.cos(declination)
    sinod = sino * sin_d
    cosod = coso * cos_d

    hrsr = 0.0
    hrss = 0.0
    max_solar_altitude = np.arcsin(sinod + cosod)
    # alsmx = max_solar_altitude

    # Calculate level-plain solar azimuth
    temp = -sin_d / coso

    if temp > 1.0:
        temp = 1.0
    elif temp < -1.0:
        temp = -1.0

    level_sunset_azimuth = np.arccos(temp)

    # Check for reach azimuth less than sunrise
    if azrh <= (-level_sunset_azimuth):
        alrs = 0.0
    # Check for reach azimuth greater than sunset
    elif azrh >= level_sunset_azimuth:
        alrs = 0.0
    # Reach azimuth is between sunrise and sunset
    elif azrh == 0.0:
        alrs = max_solar_altitude
    else:
        alrs = _solalt(
            coso, sino, sin_d, azrh, 0.0, max_solar_altitude, maxiter_sntemp
        )

    sin_alrs = np.sin(alrs)

    # Calculate level-plain sunrise/set hour angle
    tano = sino / coso
    tan_d = sin_d / cos_d
    tanod = tano * tan_d
    horizontal_hour_angle = np.arccos(-tanod)
    sinhro = np.sin(horizontal_hour_angle)

    # Calculate total potential shade on level-plain
    total_shade = 2.0 * ((horizontal_hour_angle * sinod) + (sinhro * cosod))
    if total_shade < 0.0:
        total_shade = NEARZERO

    hrso = horizontal_hour_angle
    azso = level_sunset_azimuth
    totsh = total_shade

    if azrh <= (-azso):
        hrrs = -hrso
    elif azrh >= azso:
        hrrs = hrso
    elif azrh == 0.0:
        hrrs = 0.0
    else:
        temp = (sin_alrs - sinod) / cosod

        if temp > 1.0:
            temp = 1.0
        elif temp < -1.0:
            temp = -1.0

        if azrh > 0.0:
            hrrs = np.abs(np.arccos(temp))
        else:
            hrrs = -(np.abs(np.arccos(temp)))

    if alte == 0.0 and altw == 0.0:
        hrsr = -hrso
        hrss = hrso
        sti = 0.0
        svi = _rprnvg(
            hrsr,
            hrrs,
            hrss,
            sino,
            coso,
            sin_d,
            cosod,
            sinod,
            vce,
            voe,
            vhe,
            azrh,
            vdemx,
            vdemn,
            seg_width,
            summer_flag,
            vcw,
            vow,
            vhw,
            vdwmx,
            vdwmn,
        ) / (seg_width * totsh)
    else:
        if -azso <= azrh:
            altop_0 = alte
            aztop_0 = azso * (alte / HALF_PI) - azso
        else:
            altop_0 = altw
            aztop_0 = azso * (altw / HALF_PI) - azso

        if altop_0 == 0.0:
            hrsr = -hrso
        else:
            azmn = -azso
            azmx = 0.0
            azs = aztop_0
            altmx = altop_0
            almn = 0.0
            almx = 1.5708
            als = _solalt(coso, sino, sin_d, azs, almn, almx, maxiter_sntemp)
            azs, als, hrs = _snr_sst(
                coso,
                sino,
                sin_d,
                altmx,
                almn,
                almx,
                azmn,
                azmx,
                azs,
                als,
                azrh,
                maxiter_sntemp,
            )
            hrsr = hrs

        if azso <= azrh:
            altop_1 = alte
            aztop_1 = azso - azso * (alte / HALF_PI)
        else:
            altop_1 = altw
            aztop_1 = azso - azso * (altw / HALF_PI)

        if altop_1 == 0.0:
            hrss = hrso
        else:
            azmn = 0.0
            azmx = azso
            azs = aztop_1
            altmx = altop_1
            almn = 0.0
            almx = 1.5708
            als = _solalt(coso, sino, sin_d, azs, almn, almx, maxiter_sntemp)
            azs, als, hrs = _snr_sst(
                coso,
                sino,
                sin_d,
                altmx,
                almn,
                almx,
                azmn,
                azmx,
                azs,
                als,
                azrh,
                maxiter_sntemp,
            )
            hrss = hrs

        if hrrs < hrsr:
            hrrh = hrsr
        elif hrrs > hrss:
            hrrh = hrss
        else:
            hrrh = hrrs

        # seg_daylight = (hrss - hrsr) * RADTOHOUR
        sti = 1.0 - (
            (
                ((hrss - hrsr) * sinod)
                + ((math.sin(hrss) - math.sin(hrsr)) * cosod)
            )
            / (totsh)
        )
        svi = (
            _rprnvg(
                hrsr,
                hrrh,
                hrss,
                sino,
                coso,
                sin_d,
                cosod,
                sinod,
                vce,
                voe,
                vhe,
                azrh,
                vdemx,
                vdemn,
                seg_width,
                summer_flag,
                vcw,
                vow,
                vhw,
                vdwmx,
                vdwmn,
            )
        ) / (seg_width * totsh)

    if sti < 0.0:
        sti = 0.0
    if sti > 1.0:
        sti = 1.0
    if svi < 0.0:
        svi = 0.0
    if svi > 1.0:
        svi = 1.0

    shade = sti + svi

    return shade, svi


@nb.jit(nopython=True)
def _solalt(coso, sino, sin_d, az, almn, almx, maxiter_sntemp):
    """Determine solar altitude from trigonometric parameters.

    This is the solalt function from PRMS.

    Args:
        coso: Cosine of segment latitude
        sino: Sine of segment latitude
        sin_d: Sine of solar declination
        az: Azimuth angle (radians)
        almn: Minimum altitude (radians)
        almx: Maximum altitude (radians)
        maxiter_sntemp: Maximum iterations for convergence

    Returns:
        al: Solar altitude (radians)
    """
    maxiter_sntemp = int(maxiter_sntemp)

    if abs(abs(az) - HALF_PI) < NEARZERO:
        temp = abs(sin_d / sino)
        if temp > 1.0:
            temp = 1.0
        al = np.arcsin(temp)
    else:
        cosaz = np.cos(az)
        a = sino / (cosaz * coso)
        b = sin_d / (cosaz * coso)

        al = (almn + almx) / 2.0
        kount = 0
        fal = np.cos(al) - (a * np.sin(al)) + b
        delal = fal / (-np.sin(al) - (a * np.cos(al)))

        for kount in range(1, int(maxiter_sntemp + 1)):
            if abs(fal) < NEARZERO:
                break
            if abs(delal) < NEARZERO:
                break
            alold = al
            cosal = np.cos(al)
            sinal = np.sin(al)
            fal = cosal - (a * sinal) + b
            fpal = -sinal - (a * cosal)
            if kount <= 3:
                delal = fal / fpal
            else:
                fppal = b - fal
                delal = (2.0 * fal * fpal) / (
                    (2.0 * fpal * fpal) - (fal * fppal)
                )
            al = al - delal
            if al < almn:
                al = (alold + almn) / 2.0
            if al > almx:
                al = (alold + almx) / 2.0

    return al


@nb.jit(nopython=True)
def _snr_sst(
    coso,
    sino,
    sin_d,
    alt,
    almn,
    almx,
    azmn,
    azmx,
    azs,
    als,
    azrh,
    maxiter_sntemp,
):
    """Determine local solar sunrise/set azimuth, altitude, and hour angle.

    This is the snr_sst function from PRMS.

    Args:
        coso: Cosine of segment latitude
        sino: Sine of segment latitude
        sin_d: Sine of solar declination
        alt: Topographic altitude (radians)
        almn: Minimum solar altitude (radians)
        almx: Maximum solar altitude (radians)
        azmn: Minimum azimuth angle (radians)
        azmx: Maximum azimuth angle (radians)
        azs: Initial guess for sunrise/set azimuth (radians)
        als: Initial guess for sunrise/set altitude (radians)
        azrh: Stream azimuth angle (radians)
        maxiter_sntemp: Maximum iterations for convergence

    Returns:
        azs: Sunrise/set azimuth (radians)
        als: Sunrise/set altitude (radians)
        hrs: Sunrise/set hour angle (radians)
    """
    maxiter_sntemp = int(maxiter_sntemp)

    # Trig function for local altitude
    tanalt = np.tan(alt)
    tano = sino / coso
    f = 999999.0
    delazs = 9999999.0
    g = 99999999.0
    delals = 99999999.0

    # Begin Newton-Raphson solution
    for count in range(int(maxiter_sntemp)):
        if abs(delazs) < NEARZERO:
            break
        if abs(delals) < NEARZERO:
            break
        if abs(f) < NEARZERO:
            break
        if abs(g) < NEARZERO:
            break

        cosazs = np.cos(azs)
        sinazs = np.sin(azs)

        sinazr = abs(np.sin(azs - azrh))
        if ((azs - azrh) <= 0.0 and (azs - azrh) <= -PI) or (
            (azs - azrh) > 0.0 and (azs - azrh) <= PI
        ):
            cosazr = np.cos(azs - azrh)
        else:
            cosazr = -np.cos(azs - azrh)

        cosals = np.cos(als)
        if cosals < NEARZERO:
            cosals = NEARZERO
        sinals = np.sin(als)
        tanals = sinals / cosals

        # Functions of Azs & Als
        f = cosazs - (((sino * sinals) - sin_d) / (coso * cosals))
        g = tanals - (tanalt * sinazr)

        # First partials derivatives of f & g
        fazs = -sinazs
        fals = ((tanals * (sin_d / coso)) - (tano / cosals)) / cosals
        gazs = -tanalt * cosazr
        gals = 1.0 / (cosals * cosals)

        # Jacobian
        xjacob = (fals * gazs) - (fazs * gals)

        # Delta corrections
        delazs = ((f * gals) - (g * fals)) / xjacob
        delals = ((g * fazs) - (f * gazs)) / xjacob

        # New values of Azs & Als
        azs = azs + delazs
        als = als + delals

        # Check for limits
        if azs < (azmn + NEARZERO):
            azs = azmn + NEARZERO
        if azs > (azmx - NEARZERO):
            azs = azmx - NEARZERO
        if als < (almn + NEARZERO):
            als = almn + NEARZERO
        if als > (almx - NEARZERO):
            als = almx - NEARZERO

    # Ensure azimuth remains between -PI & PI
    if azs < -PI:
        azs = azs + PI
    elif azs > PI:
        azs = azs - PI

    # Determine local sunrise/set hour angle
    sinals = np.sin(als)
    temp = (sinals - (sino * sin_d)) / (coso * np.cos(np.arcsin(sin_d)))

    if temp > 1.0:
        temp = 1.0
    elif temp < -1.0:
        temp = -1.0

    if azs > 0.0:
        hrs = np.abs(np.arccos(temp))
    else:
        hrs = -(np.abs(np.arccos(temp)))

    return azs, als, hrs


@nb.jit(nopython=True)
def _rprnvg(
    hrsr,
    hrrs,
    hrss,
    sino,
    coso,
    sin_d,
    cosod,
    sinod,
    vce,
    voe,
    vhe,
    azrh,
    vdemx,
    vdemn,
    seg_width,
    summer_flag,
    vcw,
    vow,
    vhw,
    vdwmx,
    vdwmn,
):
    """Compute riparian vegetation shade.

    This is the rprnvg function from PRMS.

    Args:
        hrsr: Sunrise hour angle (radians)
        hrrs: Reach sunrise hour angle (radians)
        hrss: Sunset hour angle (radians)
        sino: Sine of segment latitude
        coso: Cosine of segment latitude
        sin_d: Sine of solar declination
        cosod: Product of cos(latitude) and cos(declination)
        sinod: Product of sin(latitude) and sin(declination)
        vce: East bank vegetation crown width (meters)
        voe: East bank vegetation offset (meters)
        vhe: East bank vegetation height (meters)
        azrh: Stream azimuth angle (radians)
        vdemx: Maximum east bank vegetation density
        vdemn: Minimum east bank vegetation density
        seg_width: Flow width for the segment (meters)
        summer_flag: 1 for summer, 0 for winter
        vcw: West bank vegetation crown width (meters)
        vow: West bank vegetation offset (meters)
        vhw: West bank vegetation height (meters)
        vdwmx: Maximum west bank vegetation density
        vdwmn: Minimum west bank vegetation density

    Returns:
        Total vegetation shade (sum of east and west bank contributions)
    """
    # Determine seasonal shade
    if hrsr == hrss:
        svri = 0.0
        svsi = 0.0
    else:
        svri = 0.0
        if hrsr < hrrs:
            vco = (vce / 2.0) - voe
            delhsr = hrrs - hrsr

            for n in range(15):
                hrs = hrsr + (EPSLON[n] * delhsr)
                coshrs = np.cos(hrs)
                # sinhrs = np.sin(hrs)
                temp = sinod + (cosod * coshrs)
                if temp > 1.0:
                    temp = 1.0
                als = np.arcsin(temp)
                cosals = np.cos(als)
                sinals = np.sin(als)
                if sinals == 0.0:
                    sinals = NEARZERO

                temp = ((sino * sinals) - sin_d) / (coso * cosals)

                if temp > 1.0:
                    temp = 1.0
                elif temp < -1.0:
                    temp = -1.0

                azs = np.arccos(temp)
                if azs < 0.0:
                    azs = HALF_PI - azs
                if hrs < 0.0:
                    azs = -azs

                bs = (
                    (vhe * (cosals / sinals)) * abs(np.sin(azs - azrh))
                ) + vco
                if bs < 0.0:
                    bs = 0.0
                if bs > seg_width:
                    bs = seg_width

                if summer_flag == 1:
                    svri += vdemx * bs * sinals * WEIGHT[n]
                else:
                    svri += vdemn * bs * sinals * WEIGHT[n]

            svri = svri * delhsr

        svsi = 0.0
        if hrss > hrrs:
            vco = (vcw / 2.0) - vow
            delhss = hrss - hrrs

            for n in range(15):
                hrs = hrrs + (EPSLON[n] * delhss)
                coshrs = np.cos(hrs)
                # sinhrs = np.sin(hrs)
                temp = sinod + (cosod * coshrs)
                if temp > 1.0:
                    temp = 1.0
                als = np.arcsin(temp)
                cosals = np.cos(als)
                sinals = np.sin(als)
                if sinals == 0.0:
                    sinals = NEARZERO

                temp = ((sino * sinals) - sin_d) / (coso * cosals)

                if temp > 1.0:
                    temp = 1.0
                elif temp < -1.0:
                    temp = -1.0

                azs = np.arccos(temp)
                if azs < 0.0:
                    azs = HALF_PI - azs
                if hrs < 0.0:
                    azs = -azs

                bs = (
                    (vhw * (cosals / sinals)) * abs(np.sin(azs - azrh))
                ) + vco
                if bs < 0.0:
                    bs = 0.0
                if bs > seg_width:
                    bs = seg_width

                if summer_flag == 1:
                    svsi += vdwmx * bs * sinals * WEIGHT[n]
                else:
                    svsi += vdwmn * bs * sinals * WEIGHT[n]

            svsi = svsi * delhss

    return svri + svsi
