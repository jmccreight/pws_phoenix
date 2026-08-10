"""Unit tests for the Time clock (globals.py).

Time.year/.month match pywatershed utils/time_utils.py exactly; day_of_month,
doy, and dowy are ported from there. All fields are 1-based (calendar-natural,
matching pywatershed's zero_based=False default); time-varying parameters index
positionally with `field - 1` (see Upper.calculate).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from globals import Time

# (date, year, month, day_of_month, doy, dowy)  -- 2000 is a leap year
KNOWN = [
    ("2000-01-01", 2000, 1, 1, 1, 93),
    ("2000-03-15", 2000, 3, 15, 75, 167),
    ("2000-10-01", 2000, 10, 1, 275, 1),
    ("2000-12-31", 2000, 12, 31, 366, 92),
    ("2001-03-01", 2001, 3, 1, 60, 152),
]


@pytest.fixture
def time_obj():
    dates = np.array([row[0] for row in KNOWN], dtype="datetime64[D]")
    return Time(dates)


@pytest.mark.parametrize(
    ("idx", "date", "year", "month", "dom", "doy", "dowy"),
    [(ii, *row) for ii, row in enumerate(KNOWN)],
)
def test_calendar_fields(time_obj, idx, date, year, month, dom, doy, dowy):
    time_obj.set_index(idx)
    assert time_obj.current == np.datetime64(date)
    assert time_obj.year == year
    assert time_obj.month == month
    assert time_obj.day_of_month == dom
    assert time_obj.doy == doy
    assert time_obj.dowy == dowy


def test_n_time_and_set_index(time_obj):
    assert time_obj.n_time == len(KNOWN)
    time_obj.set_index(2)
    assert time_obj.current_index == 2


def test_accepts_dataarray():
    dates = np.array(["2000-06-15"], dtype="datetime64[D]")
    tt = Time(xr.DataArray(dates, dims=["time"]))
    assert tt.year == 2000
    assert tt.month == 6


def test_jsol():
    """Solar day of year (days since the most recent Dec 22, 1-based)
    matches pywatershed's datetime_jsol convention, including across
    the Dec 22 rollover and a leap year."""
    cases = [  # (date, jsol)
        ("2000-12-21", 366),  # leap: doy 356 + 10
        ("2000-12-22", 1),
        ("2000-12-31", 10),
        ("2001-01-01", 11),
        ("2001-06-21", 182),  # doy 172 + 10
        ("2001-12-21", 365),  # non-leap: doy 355 + 10
    ]
    dates = np.array([cc[0] for cc in cases], dtype="datetime64[D]")
    tt = Time(dates)
    for ii, (_date, jsol) in enumerate(cases):
        tt.set_index(ii)
        assert tt.jsol == jsol
