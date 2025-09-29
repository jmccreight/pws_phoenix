import pathlib as pl
import sys
from typing import Any, Dict

import numpy as np
import xarray as xr

# Import from the toy model module
from toy_model_1_xr import Lower, Model, Upper

np.random.seed(42)

# Add parent directory to path for utils
parent_dir = (pl.Path("./") / __file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils import timer  # noqa

# Dimensions
n_years = 4
n_space = 20

start_year = 2000
end_year = start_year + n_years
start_time = np.datetime64(f"{start_year}-01-01")
end_time = np.datetime64(f"{end_year}-01-01") - np.timedelta64(1, "D")
time = np.arange(start_time, end_time, dtype="datetime64[D]")
n_time = len(time)

space = np.arange(n_space)

# Parameters
parameter_file = pl.Path("toy_model_1_data/parameters.nc")
dims_match = False
if parameter_file.exists():
    param_ds = xr.open_dataset(parameter_file)
    dims_match = (
        len(param_ds.space) == n_space and len(param_ds.time) == n_time
    )
    del param_ds

if not dims_match:
    parameter_file.unlink(missing_ok=True)

if not parameter_file.exists():
    print(f"Creating parameter file: {parameter_file}")
    param_ds = xr.Dataset(
        data_vars=dict(
            param_up_0=(
                ["space"],
                np.random.uniform(low=0.1, high=1, size=n_space),
            ),
            param_up_1=(
                ["space", "time"],
                np.random.uniform(low=0.1, high=1, size=(n_space, n_time)),
            ),
            param_low_0=(
                ["space"],
                np.random.uniform(low=0.17, high=0.23, size=(n_space)),
            ),
        ),
        coords=dict(
            space_coord=("space", space),
            time_coord=("time", time),
        ),
        attrs=dict(description="Flow parameters."),
    )
    param_ds.to_netcdf(parameter_file)
    del param_ds

# Forcing(s)
forcing_0_file = pl.Path("toy_model_1_data/forcing_0.nc")
if not dims_match:
    forcing_0_file.unlink(missing_ok=True)

if not forcing_0_file.exists():
    print(f"Creating forcing file: {forcing_0_file}")
    sin_data = np.sin(
        np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
    )
    # import matplotlib.pyplot as plt
    # plt.plot(time, sin_data)
    # plt.show()
    shifts = np.random.uniform(low=10, high=100, size=n_space)
    forcing_0_data = (
        np.broadcast_to(sin_data.transpose(), (n_space, n_time))
        + np.broadcast_to(shifts, (n_time, n_space)).transpose()
    )
    forcing_0 = xr.DataArray(
        data=forcing_0_data,
        dims=["space", "time"],
        coords=dict(
            space_coord=("space", space),
            time_coord=("time", time),
        ),
        attrs=dict(
            description="Primal forcing.",
            units="parsecs",
        ),
    )
    del sin_data, shifts, forcing_0_data
    forcing_0.to_netcdf(forcing_0_file)

# initial conditions
ic_files_dict = {
    "flow": pl.Path("toy_model_1_data/flow_ic.nc"),
    "storage": pl.Path("toy_model_1_data/storage_ic.nc"),
}
if not dims_match:
    for kk, vv in ic_files_dict.items():
        vv.unlink(missing_ok=True)

for kk, vv in ic_files_dict.items():
    if vv.exists():
        continue

    print(f"Creating initital conditon file: {vv}")
    ic_units_dict = {"flow": "cumecs", "storage": "quibits"}
    if kk == "flow":
        data = np.random.uniform(low=100, high=1000, size=n_space)
    elif kk == "storage":
        data = np.random.uniform(low=100, high=500, size=n_space)
    else:
        raise ValueError("?")

    da = xr.DataArray(
        data=data,
        dims=["space"],
        coords=dict(
            space_coord=("space", space),
        ),
        attrs=dict(
            description=f"Initial {kk}.",
            units=ic_units_dict[kk],
            time=str(time[0]),
        ),
    )
    da.to_netcdf(vv)
    del data, da

dt = np.float64(1.0)

# TODO: Is there a case where we pass vars/memory and not files?
process_dict: Dict[str, Dict[str, Any]] = {
    "upper": {
        "class": Upper,
        "forcing_0": forcing_0_file,
        "flow_initial": ic_files_dict["flow"],
        "parameters": parameter_file,
    },
    "lower": {
        "class": Lower,
        "storage_initial": ic_files_dict["storage"],
        "parameters": parameter_file,
    },
}

control = {
    "output_var_names": ["flow", "storage_previous"],
    "time_chunk_size": 10,
}


@timer
def init_model() -> None:
    global model
    model = Model(process_dict, control)


@timer
def run_model(n_steps: int = n_time, verbose: bool = False) -> None:
    global model
    model.run(dt, np.int32(n_steps), verbose=verbose)
    del model


if __name__ == "__main__":
    init_model()
    run_model(verbose=False)
