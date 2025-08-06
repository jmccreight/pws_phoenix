from typing import Literal

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass


class Input:
    def __init__(
        self, data: np.ndarray, calc_method: Literal["numpy", "numba"] = "numba"
    ) -> None:
        # could dispatch float vs int types here
        if calc_method == "numpy":
            self.calc = self.Calc(data=data)
        elif calc_method == "numba":
            jit_class_spec = [
                ("data", float64[:, :]),
                ("current_index", int64),
                ("current_values", float64[:]),
            ]
            self.calc = jitclass(self.Calc, jit_class_spec)(data=data)
        else:
            raise ValueError(f"Invalid value: {calc_method=}.")
        return

    class Calc:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data
            self.current_index = np.int64(-1)
            self.current_values = np.nan * self.data[0, :]
            return

        def advance(self):
            self.current_index += np.int64(1)
            self.current_values[:] = self.data[self.current_index,]
            return


class Upper:
    def __init__(
        self,
        forcing_0: np.ndarray,
        flow_initial: np.ndarray,
        calc_method: Literal["numpy", "numba"] = "numba",
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
        elif self.calc_method == "numba":
            jit_class_spec = [
                ("forcing_0", float64[:]),
                ("flow", float64[:]),
                ("flow_previous", float64[:]),
                ("n_loc", int64),
            ]
            self.calc = jitclass(self.Calc, jit_class_spec)(
                forcing_0=self.forcing_0,
                flow=self.flow,
                flow_previous=self.flow_previous,
            )
        else:
            raise ValueError(f"Invalid value: {calc_method=}.")

    @staticmethod
    def get_inputs() -> tuple:
        return ("forcing_0",)

    @staticmethod
    def get_variables() -> tuple:
        return ("flow", "flow_previous")

    class Calc:
        def __init__(
            self, forcing_0: np.ndarray, flow: np.ndarray, flow_previous: np.ndarray
        ) -> None:
            self.forcing_0 = forcing_0
            self.flow = flow
            self.flow_previous = flow_previous
            self.n_loc = len(self.flow)
            return

        def advance(self) -> None:
            self.flow_previous[:] = self.flow
            return

        # is parallelization possible?
        # @njit(
        #     void(
        #         self,
        #         float64[:],  # seg_lateral_inflow
        #     ),
        #     fastmath=True,
        #     parallel=True,
        # )
        def calculate(self, dt: np.float64) -> None:
            for loc in range(self.n_loc):
                self.flow[loc] = self.flow_previous[loc] * 0.95 + self.forcing_0[loc]
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

        def advance(self) -> None:
            self.storage_previous[:] = self.storage
            return

        def calculate(self, dt: np.float64) -> None:
            for loc in range(self.n_loc):
                self.storage[loc] = (
                    self.storage_previous[loc] * 0.95 + self.flow[loc] * 0.12
                )
            return


class Model:
    def __init__(self, process_dict: dict) -> None:
        self.process_dict = process_dict

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

    def advance(self) -> None:
        for vv in self.inputs_dict.values():
            vv.calc.advance()
        for vv in self.model_dict.values():
            vv.calc.advance()

    def calculate(self, dt: np.float64) -> None:
        for vv in self.model_dict.values():
            vv.calc.calculate(dt)

    def run(self, dt: np.float64) -> None:
        for tt in range(self.n_times):
            self.advance()
            self.calculate(dt=dt)
            verbose = False
            if verbose:
                print(f"{tt=}")
                print(f"{self.inputs_dict['forcing_0'].calc.current_values=}")
                print(f"{self.model_dict['upper'].calc.forcing_0=}")
                print(f"{self.model_dict['upper'].calc.flow=}")
                print(f"{self.model_dict['lower'].calc.flow=}")
                print(f"{self.model_dict['lower'].calc.storage=}")
        return


def timer(func):
    """Use as a decorator to print the execution time of the passed function"""
    import functools
    from time import time

    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_func


if __name__ == "__main__":
    # construct input/forcing timeseries
    # do bi-weekly timestep with annual cylce
    n_time = 5000
    n_space = 20000
    sin_data = np.sin(np.arange(0, np.pi * 2, np.pi * 2 / n_time))
    shifts = np.random.uniform(low=10, high=100, size=n_space)
    forcing_0_data = np.broadcast_to(
        sin_data, (n_space, n_time)
    ).transpose() + np.broadcast_to(shifts, (n_time, n_space))

    # initial conditions
    flow_initial_data = np.random.uniform(low=100, high=1000, size=n_space)
    storage_initial_data = np.random.uniform(low=100, high=500, size=n_space)

    # output arrays, just to be stored to by the first calc_method
    output_flow = forcing_0_data[0, :] * np.nan
    output_storage = forcing_0_data[0, :] * np.nan

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

        dt = np.float64(1.0)

        @timer
        def init_model():
            global model
            model = Model(process_dict)

        @timer
        def run_model():
            global model
            model.run(dt=dt)

        @timer
        def output_model():
            global model
            # check the output at the last timestep

            if calc_method == calc_methods[0]:
                output_flow[:] = model.model_dict["upper"].flow
                output_storage[:] = model.model_dict["lower"].storage
            else:
                np.testing.assert_equal(output_flow, model.model_dict["upper"].flow)
                np.testing.assert_equal(
                    output_storage, model.model_dict["lower"].storage
                )

        print(f"{calc_method=}")
        init_model()
        run_model()
        output_model()
        del model
