from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from base import Process


class Upper(Process):
    def __init__(
        self,
        parameters: xr.Dataset,
        forcing_0: xr.DataArray,
        forcing_common: xr.DataArray,
        flow_initial: xr.DataArray,
    ) -> None:
        super().__init__(
            parameters=parameters,
            forcing_0=forcing_0,
            forcing_common=forcing_common,
            flow_initial=flow_initial,
        )
        return

    @staticmethod
    def get_parameters() -> Tuple[str, ...]:
        return ("param_up_0", "param_up_1", "param_common")

    @staticmethod
    def get_inputs() -> Tuple[str, ...]:
        return ("forcing_0", "forcing_common")

    @staticmethod
    def get_mutable_inputs() -> Tuple[str, ...]:
        return ()

    @staticmethod
    def get_variables() -> Dict[str, Dict[str, Any]]:
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
    def _get_private_variables() -> Dict[str, Dict[str, Any]]:
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
        forcing_common: xr.DataArray,
        flow: xr.DataArray,
        storage_initial: xr.DataArray,
    ) -> None:
        super().__init__(
            parameters=parameters,
            forcing_common=forcing_common,
            flow=flow,
            storage_initial=storage_initial,
        )
        return

    @staticmethod
    def get_parameters() -> Tuple[str, ...]:
        return ("param_low_0", "param_common")

    @staticmethod
    def get_inputs() -> Tuple[str, ...]:
        return ("flow", "forcing_common")

    @staticmethod
    def get_mutable_inputs() -> Tuple[str, ...]:
        return ()

    @staticmethod
    def get_variables() -> Dict[str, Dict[str, Any]]:
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
    def _get_private_variables() -> Dict[str, Dict[str, Any]]:
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
