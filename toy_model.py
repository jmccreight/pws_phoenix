import numpy as np


class Upper:
    def __init__(self, forcing_0: np.ndarray, flow_initial: np.ndarray) -> None:
        self.forcing_0 = forcing_0
        self.flow = flow_initial.copy()
        self.flow_previous = self.flow.copy()

    @staticmethod
    def get_inputs() -> tuple:
        return ("forcing_0",)

    @staticmethod
    def get_variables() -> tuple:
        return ("flow", "flow_previous")

    def advance(self) -> None:
        self.flow_previous[:] = self.flow
        return

    def calculate(self, dt: np.float64):
        self.flow[:] = self.flow_previous * 0.95 + self.forcing_0
        return


class Lower:
    def __init__(self, flow: np.ndarray, storage_initial: np.ndarray) -> None:
        self.flow = flow
        self.storage = storage_initial.copy()
        self.storage_previous = self.storage.copy()

    @staticmethod
    def get_inputs() -> tuple:
        return ("flow",)

    @staticmethod
    def get_variables() -> tuple:
        return ("storage", "storage_previous")

    def advance(self) -> None:
        self.storage_previous[:] = self.storage
        return

    def calculate(self, dt: np.float64) -> None:
        self.storage[:] = self.storage_previous * 0.95 + self.flow * 0.12
        return


class Model:
    def __init__(self, process_dict: dict, inputs: dict) -> None:
        self.process_dict = process_dict
        self.inputs = inputs

        self.n_times = self.get_n_times()

        self.model_dict = {}
        for kk, vv in process_dict.items():
            init_dict = {kkk: vvv for kkk, vvv in vv.items() if kkk != "class"}
            inputs_req = vv["class"].get_inputs()
            for ii in inputs_req:
                if ii in inputs.keys():
                    init_dict[ii] = inputs[ii].current_values
                else:
                    for pp in self.procs_above(kk):
                        if ii in self.model_dict[pp].get_variables():
                            init_dict[ii] = getattr(self.model_dict[pp], ii)

            self.model_dict[kk] = vv["class"](**init_dict)

        return

    def get_n_times(self) -> np.int64:
        # would check consistency of this across all inputs
        return np.int64(list(self.inputs.values())[0].data.shape[1])

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
        for vv in self.inputs.values():
            vv.advance()
        for vv in self.model_dict.values():
            vv.advance()

    def calculate(self, dt: np.float64) -> None:
        for vv in self.model_dict.values():
            vv.calculate(dt)

    def run(self, dt: np.float64) -> None:
        for tt in range(self.n_times):
            self.advance()
            self.calculate(dt=dt)
            verbose = True
            if verbose:
                print(f"{tt=}")
                print(f"{self.inputs['forcing_0'].current_values=}")
                print(f"{self.model_dict['upper'].forcing_0=}")
                print(f"{self.model_dict['upper'].flow=}")
                print(f"{self.model_dict['lower'].flow=}")
                print(f"{self.model_dict['lower'].storage=}")
        return


class Input:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data.copy()
        self.current_index = -1
        self.current_values = np.nan * self.data[:, 0]
        return

    def advance(self):
        self.current_index += 1
        self.current_values[:] = self.data[:, self.current_index]
        return


if __name__ == "__main__":
    # do bi-weekly timestep with annual cylce
    sin_data = np.sin(np.arange(0, np.pi * 2, np.pi * 2 / 26))
    forcing_0 = Input(np.stack((sin_data + 15, sin_data + 25), axis=0))

    input_dict = {"forcing_0": forcing_0}
    process_dict = {
        "upper": {"class": Upper, "flow_initial": np.array([100.0, 200.0])},
        "lower": {"class": Lower, "storage_initial": np.array([300.0, 700.0])},
    }

    dt = np.float64(1.0)
    model = Model(process_dict, input_dict)
    model.run(dt=dt)
    asdff
