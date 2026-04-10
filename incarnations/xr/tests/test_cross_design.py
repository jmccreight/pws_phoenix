import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from base import Model
from base_attrs import ModelAttrs
from processes import Lower as LowerBase
from processes import Upper as UpperBase
from processes_attrs import Lower as LowerAttrs
from processes_attrs import Upper as UpperAttrs

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dimensions():
    n_years = 1
    n_space = 20
    start_year = 2000
    start_time = np.datetime64(f"{start_year}-01-01")
    end_time = np.datetime64(f"{start_year + n_years}-01-01") - np.timedelta64(
        1, "D"
    )
    time = np.arange(start_time, end_time, dtype="datetime64[D]")
    return {
        "n_space": n_space,
        "n_time": len(time),
        "time": time,
        "space": np.arange(n_space),
    }


@pytest.fixture
def parameters_memory(dimensions):
    return xr.Dataset(
        data_vars=dict(
            param_up_0=(
                ["space"],
                np.random.uniform(0.1, 1, size=dimensions["n_space"]),
            ),
            param_up_1=(
                ["time", "space"],
                np.random.uniform(
                    0.1, 1, size=(dimensions["n_time"], dimensions["n_space"])
                ),
            ),
            param_low_0=(
                ["space"],
                np.random.uniform(0.17, 0.23, size=dimensions["n_space"]),
            ),
            param_common=(
                ["space"],
                np.random.uniform(0, 0, size=dimensions["n_space"]),
            ),
        ),
        coords=dict(
            space_coord=("space", dimensions["space"]),
            time_coord=("time", dimensions["time"]),
        ),
    )


@pytest.fixture
def parameters_file(parameters_memory, tmp_path):
    path = tmp_path / "parameters.nc"
    parameters_memory.to_netcdf(path)
    return path


@pytest.fixture
def forcing_memory(dimensions):
    sin_data = np.sin(
        np.arange(
            0,
            2 * np.pi,
            2 * np.pi / dimensions["n_time"],
        )
    )
    shifts = np.random.uniform(10, 100, size=dimensions["n_space"])
    return xr.DataArray(
        data=sin_data[:, np.newaxis] + shifts[np.newaxis, :],
        dims=["time", "space"],
        coords=dict(
            space_coord=("space", dimensions["space"]),
            time_coord=("time", dimensions["time"]),
        ),
        name="forcing_0",
    )


@pytest.fixture
def forcing_file(forcing_memory, tmp_path):
    path = tmp_path / "forcing_0.nc"
    forcing_memory.to_netcdf(path)
    return path


@pytest.fixture
def forcing_common_memory(dimensions):
    return xr.DataArray(
        data=np.ones((dimensions["n_time"], dimensions["n_space"])),
        dims=["time", "space"],
        coords=dict(
            space_coord=("space", dimensions["space"]),
            time_coord=("time", dimensions["time"]),
        ),
        name="forcing_common",
    )


@pytest.fixture
def forcing_common_file(forcing_common_memory, tmp_path):
    path = tmp_path / "forcing_common.nc"
    forcing_common_memory.to_netcdf(path)
    return path


@pytest.fixture
def flow_ic_memory(dimensions):
    return xr.DataArray(
        data=np.random.uniform(100, 1000, size=dimensions["n_space"]),
        dims=["space"],
        coords=dict(space_coord=("space", dimensions["space"])),
        name="flow_ic",
    )


@pytest.fixture
def flow_ic_file(flow_ic_memory, tmp_path):
    path = tmp_path / "flow_ic.nc"
    flow_ic_memory.to_netcdf(path)
    return path


@pytest.fixture
def storage_ic_memory(dimensions):
    return xr.DataArray(
        data=np.random.uniform(100, 500, size=dimensions["n_space"]),
        dims=["space"],
        coords=dict(space_coord=("space", dimensions["space"])),
        name="storage_ic",
    )


@pytest.fixture
def storage_ic_file(storage_ic_memory, tmp_path):
    path = tmp_path / "storage_ic.nc"
    storage_ic_memory.to_netcdf(path)
    return path


@pytest.fixture(params=["memory", "file"])
def data(
    request,
    parameters_memory,
    parameters_file,
    forcing_memory,
    forcing_file,
    forcing_common_memory,
    forcing_common_file,
    flow_ic_memory,
    flow_ic_file,
    storage_ic_memory,
    storage_ic_file,
):
    if request.param == "memory":
        return {
            "parameters": parameters_memory,
            "forcing": forcing_memory,
            "forcing_common": forcing_common_memory,
            "flow_ic": flow_ic_memory,
            "storage_ic": storage_ic_memory,
        }
    else:
        return {
            "parameters": parameters_file,
            "forcing": forcing_file,
            "forcing_common": forcing_common_file,
            "flow_ic": flow_ic_file,
            "storage_ic": storage_ic_file,
        }


# ---------------------------------------------------------------------------
# Cross-design equivalence test
# ---------------------------------------------------------------------------


class TestCrossDesignEquivalence:
    """Direct comparison between base.py (Process) and base_attrs.py
    (@process) designs.  Both models run with identical inputs; outputs
    are compared directly without relying on a shared ground-truth fixture.
    """

    def test_outputs_match(self, dimensions, data):
        dt = np.float64(1.0)
        n_time = np.int32(dimensions["n_time"])

        # ---- old design ----
        old_process_dict = {
            "upper": {
                "class": UpperBase,
                "forcing_0": data["forcing"],
                "forcing_common": data["forcing_common"],
                "flow_initial": data["flow_ic"],
                "parameters": data["parameters"],
            },
            "lower": {
                "class": LowerBase,
                "forcing_common": data["forcing_common"],
                "storage_initial": data["storage_ic"],
                "parameters": data["parameters"],
            },
        }
        with Model(old_process_dict, {}) as old_model:
            old_model.run(dt, n_time)

        # ---- new design ----
        new_process_dict = {
            "upper": {
                "class": UpperAttrs,
                "forcing_0": data["forcing"],
                "forcing_common": data["forcing_common"],
                "flow_initial": data["flow_ic"],
                "parameters": data["parameters"],
            },
            "lower": {
                "class": LowerAttrs,
                "forcing_common": data["forcing_common"],
                "storage_initial": data["storage_ic"],
                "parameters": data["parameters"],
            },
        }
        with ModelAttrs(new_process_dict, {}) as new_model:
            new_model.run(dt, n_time)

        # ---- direct comparison ----
        for var in ("flow", "flow_previous"):
            np.testing.assert_allclose(
                old_model.model_dict["upper"][var].values,
                new_model.model_dict["upper"][var].values,
                rtol=1e-12,
                err_msg=f"upper[{var!r}] differs between designs",
            )

        for var in ("storage", "storage_previous"):
            np.testing.assert_allclose(
                old_model.model_dict["lower"][var].values,
                new_model.model_dict["lower"][var].values,
                rtol=1e-12,
                err_msg=f"lower[{var!r}] differs between designs",
            )
