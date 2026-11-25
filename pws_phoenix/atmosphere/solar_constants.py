"""
atmosphere/solar_constants.py
=============================
Solar-geometry constants, ported verbatim from
pywatershed/atmosphere/solar_constants.py (PRMS 6
c_solar_radiation.f90 heritage). All are fixed 366-day (ndoy) tables
or scalars, computed once at import.
"""

import math

import numpy as np

NDOY = 366  # pywatershed constants.ndoy

pi = math.pi
n_days_per_year_flt = 365.242
eccentricy = 0.01671
two_pi = 2 * pi
pi_12 = 12 / pi
rad_day = two_pi / n_days_per_year_flt

julian_days = np.arange(NDOY) + 1

obliquity = 1 - (eccentricy * np.cos((julian_days - 3) * rad_day))

# integer julian day values yield local noon in the sunrise equation
yy = (julian_days - 1) * rad_day
yy2 = yy * 2
yy3 = yy * 3
solar_declination = (
    0.006918
    - 0.399912 * np.cos(yy)
    + 0.070257 * np.sin(yy)
    - 0.006758 * np.cos(yy2)
    + 0.000907 * np.sin(yy2)
    - 0.002697 * np.cos(yy3)
    + 0.00148 * np.sin(yy3)
)

# Solar constant cal/cm2/min (r0 could also be 1.95, Drummond et al 1968)
r0 = 2.0
# Solar constant for 60 minutes
r1 = (60.0 * r0) / (obliquity**2)

# The degree-day solar fraction table (solf) lives in prms_atmosphere.py,
# the only module that reads it: its kernels freeze the table, and a
# cached kernel only notices edits to its OWN file (DEVELOPER.md "Numba
# cache").
