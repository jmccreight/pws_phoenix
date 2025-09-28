import pathlib as pl
from typing import Union

import numpy as np
import xarray as xr

from utils import timer

np.random.seed(42)

# TODO is there a fast index order in python, does it matter?
# TODO: pass through an option to load on open?


def open_xr(path: pl.Path) -> Union[xr.DataArray, xr.Dataset]:
    """Return appropriate DataArray or Dataset given a file.

    Args:
        path: The path of the file to open.
    """
    ds = xr.open_dataset(path)
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    else:
        return ds


class Parameters:
    def __init__(self, data: xr.Dataset):
        self.data = data
        pass

    @staticmethod
    def from_file(file_path: pl.Path) -> "Parameters":
        return Parameters(xr.open_dataset(file_path))


class Input:
    # TODO: needs a control to find the start time index as the first index
    def __init__(
        self, data_or_file: Union[xr.DataArray, pl.Path], read_only=False
    ) -> None:
        if isinstance(data_or_file, pl.Path):
            self.data = xr.open_dataarray(data_or_file)
            self._input_file = data_or_file
        else:
            self.data = data_or_file
            self._input_file = None
        # <
        if read_only:
            self.data.values.flags.writeable = True
            # self.data.values.flags.writeable = False  # restore
        # <
        self._current_index = np.int64(-1)
        self._current_values = np.nan * self.data[:, 0]
        return

    def advance(self) -> None:
        self._current_index += np.int64(1)
        self._current_values[:] = self.data[:, self._current_index]
        return

    @property
    def current_values(self):
        return self._current_values

    @staticmethod
    def from_file(
        file_path: pl.Path, var_name: Union[str, None] = None
    ) -> "Input":
        if var_name is None:
            return Input(xr.open_dataarray(file_path))
        else:
            return Input(xr.open_dataset(file_path)[var_name])


class Process:
    def __init__(self, parameters: xr.Dataset, **kwargs):
        import itertools

        # add parameters, inputs, input_outputs, and public variables
        # Note: It is somewhat strange that inputs of parameters are treated
        #       coming from a dataset while inputs and input_outputs are
        #       treated as coming from dataarrays. Conceptually, it makes some
        #       sense for the following reasons:
        #       1. parameters are not time varying
        #       2. parameters generally have somewhat less impact on output
        #       3. parameters are "endogenous" or internal, while inputs
        #          are external and might be coming from various other places/
        dim_names = [vv["dims"] for vv in self.get_variables().values()]
        dim_names = set(itertools.chain.from_iterable(dim_names))
        # Strange naming assumption of coords. Could use a mapping from
        # dims to coords somewhere
        coords = {}
        for dd in dim_names:
            coords[f"{dd}_coord"] = (dd, parameters[dd].values)
        # <
        self.data = xr.Dataset(coords=coords)
        # parameters
        for pp in self.get_parameters():
            # in this case we want read-only copies??
            self[pp] = parameters[pp]
            self[pp].values.flags.writeable = False
        for ii in self.get_inputs():
            # Input object set on self?4
            if isinstance(kwargs[ii], Input):
                self[ii] = kwargs[ii].current_values
                assert id(self[ii].values) == id(
                    kwargs[ii].current_values.values
                )
            else:
                self[ii] = kwargs[ii]
                assert id(self[ii].values) == id(kwargs[ii].values)

            # TODO write a test for the above references
        for oo in self.get_input_outputs():
            self[oo] = kwargs[oo].current_values
            assert id(self[oo].values) == id(kwargs[oo].current_values.values)
        for kk, vv in self.get_variables().items():
            self[kk] = self._var_from_metadata(vv, **kwargs)
        for kk, vv in self._get_private_variables().items():
            self[kk] = self._var_from_metadata(vv)
        # <
        return

    def _var_from_metadata(self, var_meta: dict, **kwargs) -> xr.DataArray:
        # TODO move this map to constants somewhere
        fill_value_map = {np.float64: np.nan}
        sizes = self.data.sizes
        dims = tuple([sizes[dd] for dd in var_meta["dims"]])
        da = xr.DataArray(
            data=np.full(
                dims,
                fill_value_map[var_meta["dtype"]],
                dtype=var_meta["dtype"],
            ),
            dims=var_meta["dims"],
            attrs=var_meta["metadata"],
        )
        if (
            "initial" in var_meta.keys()
            and var_meta["initial"] in kwargs.keys()
        ):
            da[:] = kwargs[var_meta["initial"]]
        # <
        return da

    def __getitem__(self, name: str) -> xr.DataArray:
        # TODO: may want to use getattr to get other properties?
        return self.data[name]

    def __setitem__(self, name: str, value: xr.DataArray) -> None:
        self.data[name] = value  # [name]
        return

    def __delitem__(self, name: str) -> None:
        del self.data[name]
        return

    @staticmethod
    def get_parameters() -> tuple:
        raise NotImplementedError()

    @staticmethod
    def get_inputs() -> tuple:
        raise NotImplementedError()

    @staticmethod
    def get_input_outputs() -> tuple:
        raise NotImplementedError()

    @staticmethod
    def get_variables() -> dict:
        raise NotImplementedError()

    @staticmethod
    def _get_private_variables() -> dict:
        raise NotImplementedError()


class Upper(Process):
    def __init__(
        self,
        parameters: xr.Dataset,
        forcing_0: xr.DataArray,
        flow_initial: xr.DataArray,
    ) -> None:
        super().__init__(
            parameters=parameters,
            forcing_0=forcing_0,
            flow_initial=flow_initial,
        )
        return

    @staticmethod
    def get_parameters() -> tuple:
        return ("param_up_0", "param_up_1")

    @staticmethod
    def get_inputs() -> tuple:
        return ("forcing_0",)

    @staticmethod
    def get_input_outputs() -> tuple:
        return ()

    @staticmethod
    def get_variables() -> dict:
        return {
            "flow": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "flowy"},
                "initial": "flow_initial",
            },
            "flow_previous": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "was flowy"},
            },
        }

    @staticmethod
    def _get_private_variables() -> dict:
        return {}

    def advance(self) -> None:
        self["flow_previous"][:] = self["flow"]
        return

    def calculate(self, dt: np.float64) -> None:
        for loc in self["space"]:
            self["flow"][loc] = (
                self["flow_previous"][loc] * np.float64(0.95)
                + self["forcing_0"][loc]
            )
        return


class Lower(Process):
    def __init__(
        self,
        parameters: xr.Dataset,
        flow: xr.DataArray,
        storage_initial: xr.DataArray,
    ) -> None:
        super().__init__(
            parameters=parameters,
            flow=flow,
            storage_initial=storage_initial,
        )
        return

    @staticmethod
    def get_parameters() -> tuple:
        return ("param_low_0",)

    @staticmethod
    def get_inputs() -> tuple:
        return ("flow",)

    @staticmethod
    def get_input_outputs() -> tuple:
        return ()

    @staticmethod
    def get_variables() -> dict:
        return {
            "storage": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "storagey"},
                "initial": "storage_initial",
            },
            "storage_previous": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "old storagey"},
            },
        }

    @staticmethod
    def _get_private_variables() -> dict:
        return {}

    def advance(self) -> None:
        self["storage_previous"][:] = self["storage"]
        return

    def calculate(self, dt: np.float64) -> None:
        for loc in self["space"]:
            self["storage"][loc] = (
                self["storage_previous"][loc] * np.float64(0.95)
            ) + (self["flow"][loc] * np.float64(0.12))
        return


class Model:
    """
    Principle: the model should transform all paths in to datasets for
    parameters and dataarrays for all other fields. No process should take
    paths.

    """

    def __init__(
        self,
        process_dict: dict,
    ) -> None:
        from copy import deepcopy

        self._passed_process_dict = process_dict
        self._process_dict = deepcopy(process_dict)

        self._paths_to_data_proc_dict()

        # wire up the model
        self.model_dict = {}
        self.inputs_dict = {}
        self._set_inputs_and_model_dicts()

        self._set_time()
        return

    def _paths_to_data_proc_dict(self):
        """All input paths to Dataset or DataArray without opening files twice.

        If inputs come as memory, we dont need to do anything (though that's
        potentially a source of user error)

        """
        repeated_paths = self._get_repeated_paths()
        for proc_name in self._process_dict.keys():
            proc = self._process_dict[proc_name]
            for input_key in proc.keys():
                input_val = proc[input_key]
                if isinstance(input_val, pl.Path):
                    if input_val in repeated_paths.keys():
                        proc[input_key] = repeated_paths[input_val]
                    else:
                        proc[input_key] = open_xr(input_val)

        # TODO: write test that repeated path has same memory id
        # assert id(self._process_dict["upper"]["parameters"]) == id(
        #     self._process_dict["lower"]["parameters"]
        # )
        return

    def _get_repeated_paths(
        self,
    ) -> dict[pl.Path, Union[xr.DataArray, xr.Dataset]]:
        """Open repeated paths in the process_dict once key against path.

        This is intended to work with parameter files or forcing files specified
        across multiple processes.
        """
        from collections import Counter

        flat_proc_dict_paths = []
        for outer_val in self._process_dict.values():
            for inner_val in outer_val.values():
                if isinstance(inner_val, pl.Path):
                    flat_proc_dict_paths.append(inner_val)

        repeated_paths_list = [
            kk for kk, vv in Counter(flat_proc_dict_paths).items() if vv > 1
        ]
        repeated_paths_data = {kk: open_xr(kk) for kk in repeated_paths_list}
        return repeated_paths_data

    def _set_inputs_and_model_dicts(self):
        """Initialize inputs and processes to processes above.

        The inputs are in self.inputs_dict and the wired model is in
        self.model_dict.
        """
        for kk, vv in self._process_dict.items():
            # get the process dict with the classes removed
            init_dict = {kkk: vvv for kkk, vvv in vv.items() if kkk != "class"}

            inputs_req = vv["class"].get_inputs()
            input_outputs_req = vv["class"].get_input_outputs()
            all_inputs = inputs_req + input_outputs_req
            for ii in all_inputs:
                if ii in init_dict.keys():
                    # combine all inputs/input_outputs into a dict that will be
                    # advanced
                    data_or_file = init_dict[ii]
                    if ii in inputs_req:
                        read_only = True
                    else:
                        raise ValueError("This should not happen from file.")
                    # <
                    init_dict[ii] = Input(data_or_file, read_only=read_only)
                    self.inputs_dict[ii] = init_dict[ii]
                    assert init_dict[ii].data is self.inputs_dict[ii].data
                    del data_or_file

                else:
                    # inputs not in init_dict need to be found elsewhere in
                    # the processes above
                    for pp in self.procs_above(kk):
                        if ii in self.model_dict[pp].get_variables():
                            init_dict[ii] = self.model_dict[pp][ii]

                # <<<
                self.model_dict[kk] = vv["class"](**init_dict)
                # TODO: test refs across processes

        return

    def _set_time(self) -> None:
        # TODO: would take start and end times and check for these
        # TODO: would check consistency of this across all inputs
        kk0 = list(self.inputs_dict.keys())[0]
        self.ntime = self.inputs_dict[kk0].data.sizes["time"]
        self.time_index = self.inputs_dict[kk0].data.time
        self.times = self.inputs_dict[kk0].data.time_coord
        return None

    def procs_above(self, proc_name) -> list:
        procs_above = []
        for pp in self._process_dict:
            if proc_name != pp:
                procs_above.append(pp)
            else:
                return procs_above

        raise ValueError("This should be unreachable.")
        return

    def advance(self) -> None:
        for vv in self.inputs_dict.values():
            vv.advance()
        for vv in self.model_dict.values():
            vv.advance()

    def calculate(self, dt: np.float64) -> None:
        for vv in self.model_dict.values():
            vv.calculate(dt)

    def run(self, dt: np.float64, n_steps: np.int32, verbose=False) -> None:
        for tt in range(n_steps):
            self.advance()
            self.calculate(dt=dt)

            if verbose:
                print(f"{tt=}")
                print(f"{self.inputs_dict['forcing_0'].current_values=}")
                print(f"{self.model_dict['upper']['forcing_0']=}")
                print(f"{self.model_dict['upper']['flow']=}")
                print(f"{self.model_dict['lower']['flow']=}")
                print(f"{self.model_dict['lower']['storage']=}")
        return


if __name__ == "__main__":
    import pathlib as pl

    import numpy as np
    import xarray as xr

    # Dimensions
    n_years = 4
    n_space = 2000

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
    process_dict = {
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

    @timer
    def init_model():
        global model
        model = Model(process_dict)

    @timer
    def run_model(n_steps=n_time, verbose=False):
        global model
        # for numba the args must be positional, not kw
        model.run(dt, np.int32(n_steps), verbose=verbose)
        del model

    init_model()
    run_model(verbose=False)
