import pathlib as pl
from typing import Literal, Union

import numpy as np
import xarray as xr

np.random.seed(42)


# TODO is there a fast index order in python, does it matter?
class Parameters:
    def __init__(self, data: xr.Dataset):
        self.data = data
        pass

    @staticmethod
    def from_file(file_path: pl.Path) -> "Parameters":
        return Parameters(xr.open_dataset(file_path))


class Input:
    # TODO: needs a control to find the start time index as the first index
    def __init__(self, data: xr.DataArray) -> None:
        self.data = data
        self.current_index = np.int64(-1)
        self.current_values = np.nan * self.data[0, :]
        return

    def advance(self) -> None:
        self.current_index += np.int64(1)
        self.current_values[:] = self.data[self.current_index, :]
        return

    @staticmethod
    def from_file(
        file_path: pl.Path, var_name: Union[str, None] = None
    ) -> "Input":
        if var_name is None:
            return Input(xr.open_dataarray(file_path))
        else:
            return Input(xr.open_dataset(file_path)[var_name])


class Upper:
    def __init__(
        self,
        forcing_0: xr.DataArray,
        flow_initial: xr.DataArray,
    ) -> None:
        self.forcing_0 = forcing_0
        self.flow = flow_initial.copy()
        self.flow_previous = self.flow.copy()
        self.calc_method = calc_method
        if self.calc_method == "numpy":
            self.calc = self.Calc(
                forcing_0=self.forcing_0,
                flow=self.flow,
                flow_previous=self.flow_previous,
            )

    @staticmethod
    def get_parameters() -> tuple:
        return ("param_up_0", "param_up_1")

    @staticmethod
    def get_inputs() -> tuple:
        return ("forcing_0",)

    @staticmethod
    def get_variables() -> dict:
        return {
            "flow": {"dims": ("space", "time"), "description": "flowy"},
            "flow_previous": {
                "dims": ("space", "time"),
                "description": "was flowy",
            },
        }

    class Calc:
        def __init__(
            self,
            forcing_0: np.ndarray,
            flow: np.ndarray,
            flow_previous: np.ndarray,
        ) -> None:
            self.forcing_0 = forcing_0
            self.flow = flow
            self.flow_previous = flow_previous
            self.n_loc = len(self.flow)
            return

        def advance(self) -> None:
            self.flow_previous[:] = self.flow
            return

        def calculate(self, dt: np.float64) -> None:
            for loc in range(self.n_loc):
                self.flow[loc] = (
                    self.flow_previous[loc] * np.float64(0.95)
                    + self.forcing_0[loc]
                )
            return


class Lower:
    def __init__(
        self,
        flow: np.ndarray,
        storage_initial: np.ndarray,
        calc_method: Literal["numpy", "numba"] = "numba",
    ) -> None:
        self.flow = flow
        self.storage = storage_initial.copy()
        self.storage_previous = self.storage.copy()
        self.calc_method = calc_method
        if self.calc_method == "numpy":
            self.calc = self.Calc(
                flow=self.flow,
                storage=self.storage,
                storage_previous=self.storage_previous,
            )
        elif self.calc_method == "numba":
            jit_class_spec = [
                ("flow", float64[:]),
                ("storage", float64[:]),
                ("storage_previous", float64[:]),
                ("n_loc", int64),
            ]
            self.calc = jitclass(self.Calc, jit_class_spec)(
                flow=self.flow,
                storage=self.storage,
                storage_previous=self.storage_previous,
            )
        else:
            raise ValueError(f"Invalid value: {calc_method=}.")

    @staticmethod
    def get_inputs() -> tuple:
        return ("flow",)

    @staticmethod
    def get_variables() -> tuple:
        return ("storage", "storage_previous")

    class Calc:
        def __init__(
            self,
            flow: np.ndarray,
            storage: np.ndarray,
            storage_previous: np.ndarray,
        ) -> None:
            self.flow = flow
            self.storage = storage
            self.storage_previous = storage_previous
            self.n_loc = len(self.flow)
            return

        def advance(self) -> None:
            self.storage_previous[:] = self.storage
            return

        def calculate(self, dt: np.float64) -> None:
            for loc in range(self.n_loc):
                self.storage[loc] = (
                    self.storage_previous[loc] * np.float64(0.95)
                ) + (self.flow[loc] * np.float64(0.12))
            return


class Model:
    def __init__(
        self,
        process_dict: dict,
        calc_method: Literal["numpy", "numba"] = "numpy",
    ) -> None:
        self.process_dict = process_dict
        self.calc_method = calc_method
        self.n_times = self.get_n_times()

        self.model_dict = {}
        self.inputs_dict = {}
        for kk, vv in process_dict.items():
            init_dict = {kkk: vvv for kkk, vvv in vv.items() if kkk != "class"}
            inputs_req = vv["class"].get_inputs()
            for ii in inputs_req:
                if ii in init_dict.keys():
                    self.inputs_dict[ii] = init_dict[ii]
                    init_dict[ii] = self.inputs_dict[ii].calc.current_values
                else:
                    for pp in self.procs_above(kk):
                        if ii in self.model_dict[pp].get_variables():
                            init_dict[ii] = getattr(self.model_dict[pp], ii)

            self.model_dict[kk] = vv["class"](**init_dict)

        if self.calc_method == "numpy":
            self.calc = self.CalcNumpy(
                inputs_dict=self.inputs_dict,
                model_dict=self.model_dict,
            )

        elif self.calc_method == "numba":
            n_max_classes = 20  # the number of args available in CalcNumba
            # default to Dummy for all classes
            inst_list_full = [Dummy() for ii in range(n_max_classes)]
            spec = [
                (f"a{ii + 1}", typeof(vv))
                for ii, vv in enumerate(inst_list_full)
            ]

            # fill the class_list and spec with inputs and model classes
            inst_list = list(self.inputs_dict.values()) + list(
                self.model_dict.values()
            )
            for ii, vv in enumerate(inst_list):
                inst_list_full[ii] = vv.calc
                spec[ii] = (f"a{ii + 1}", typeof(vv.calc))

            self.calc = jitclass(self.CalcNumba, spec)(*inst_list_full)

        else:
            raise ValueError(f"Invalid value: {calc_method=}.")

        return

    def get_n_times(self) -> np.int64:
        # would check consistency of this across all inputs
        proc_dict_0 = list(self.process_dict.values())[0]
        class_0 = proc_dict_0["class"]
        input_0 = class_0.get_inputs()[0]
        return np.int64(proc_dict_0[input_0].calc.data.shape[0])

    def procs_above(self, proc_name) -> list:
        procs_above = []
        for pp in self.process_dict:
            if proc_name != pp:
                procs_above.append(pp)
            else:
                return procs_above

        raise ValueError("This should be unreachable.")
        return

    class CalcNumpy:
        def __init__(self, inputs_dict: dict, model_dict: dict) -> None:
            self.inputs_dict = inputs_dict
            self.model_dict = model_dict
            return

        def advance(self) -> None:
            for vv in self.inputs_dict.values():
                vv.calc.advance()
            for vv in self.model_dict.values():
                vv.calc.advance()

        def calculate(self, dt: np.float64) -> None:
            for vv in self.model_dict.values():
                vv.calc.calculate(dt)

        def run(self, dt: np.float64, n_steps: np.int32) -> None:
            for tt in range(n_steps):
                self.advance()
                self.calculate(dt=dt)
                verbose = False
                if verbose:
                    print(f"{tt=}")
                    print(
                        f"{self.inputs_dict['forcing_0'].calc.current_values=}"
                    )
                    print(f"{self.model_dict['upper'].calc.forcing_0=}")
                    print(f"{self.model_dict['upper'].calc.flow=}")
                    print(f"{self.model_dict['lower'].calc.flow=}")
                    print(f"{self.model_dict['lower'].calc.storage=}")
            return


if __name__ == "__main__":
    import pathlib as pl

    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    # Dimensions
    n_time = 365 * 4
    n_space = 20000

    start_time = np.datetime64("2000-01-01")
    end_time = start_time + np.timedelta64(n_time, "D")
    time = np.arange(start_time, end_time, dtype="datetime64[D]")

    space = np.arange(n_space)

    # Parameters
    parameter_file = pl.Path("toy_model_1_data/parameters.nc")
    if not parameter_file.exists():
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
            ),
            coords=dict(
                space=space,
                time=time,
            ),
            attrs=dict(description="Flow parameters."),
        )
        param_ds.to_netcdf(parameter_file)

    # Forcing
    sin_data = np.sin(np.arange(0, 2 * np.pi * n_time / 365, 1))
    plt.plot(time, sin_data)
    plt.show()

    shifts = np.random.uniform(low=10, high=100, size=n_space)
    forcing_0_data = np.broadcast_to(
        sin_data, (n_space, n_time)
    ).transpose() + np.broadcast_to(shifts, (n_time, n_space))
    # write to file
    asdff

    # initial conditions
    # are these parameters or restarts?
    flow_initial_data = np.random.uniform(low=100, high=1000, size=n_space)
    storage_initial_data = np.random.uniform(low=100, high=500, size=n_space)

    # output arrays, just  final time to be stored to by the first calc_method
    output_flow = forcing_0_data[0, :] * np.nan
    output_storage = forcing_0_data[0, :] * np.nan

    dt = np.float64(1.0)

    calc_methods = ["numpy", "numba"]
    for calc_method in calc_methods:
        forcing_0 = Input(forcing_0_data.copy(), calc_method=calc_method)
        flow_initial = flow_initial_data.copy()
        storage_initial = storage_initial_data.copy()

        process_dict = {
            "upper": {
                "class": Upper,
                "forcing_0": forcing_0,
                "flow_initial": flow_initial,
                "calc_method": calc_method,
            },
            "lower": {
                "class": Lower,
                "storage_initial": storage_initial,
                "calc_method": calc_method,
            },
        }

        @timer
        def init_model():
            global model
            model = Model(process_dict, calc_method=calc_method)
            print(model.calc)

        @timer
        def run_model(n_steps=n_time):
            global model
            # for numba the args must be positional, not kw
            model.calc.run(dt, np.int32(n_steps))

        @timer
        def output_model():
            global model
            # check the output at the last timestep

            if calc_method == calc_methods[0]:
                output_flow[:] = model.model_dict["upper"].flow
                output_storage[:] = model.model_dict["lower"].storage
                print(f"{output_flow.mean()}")
                print(f"{output_storage.mean()}")

            else:
                np.testing.assert_equal(
                    output_flow, model.model_dict["upper"].flow
                )
                np.testing.assert_equal(
                    output_storage, model.model_dict["lower"].storage
                )

        print(f"{calc_method=}")
        init_model()
        run_model()
        output_model()
        del model
