"""Per-process lifecycle timings -- the primary benchmark set.

One timing per process per lifecycle method, each exactly ONE call
on the subprocess's shared model (``_common.get_model()``):

    time_init        type(proc)(proc._obj)       bind to the grid dataset
    time_initialize  proc.initialize()           parameter_internal derivation
    time_advance     proc.advance()              the *_previous copies
    time_calculate   proc.calculate(DT, time)    one step of physics

Build cost of a process = time_init + time_initialize; per-step cost
= time_advance + time_calculate. Repeated calls without an
``advance`` between them drift the state -- the wall clock is right,
the values are not a physics run (README section 4).

``ProcessVariants`` = the same four timings for the ledger's other
controls (``_common.VARIANT_ROWS``, "label/process"): only the slots
whose class the CONTROL model lacks, one shared model per control.

``ProcessKernel.time_kernel`` = the bare njit call below calculate():

    time_kernel      kernel(*args)   the ``_calculate`` njit dispatcher
                                     with the arguments calculate()
                                     handed it (``_common.get_kernel``)

so ``time_calculate - time_kernel`` is the Python/xarray share of a
step (the ``self._obj[name].values`` lookups and scalar prep). Only
the single-kernel processes (``_common.KERNEL_ROWS``).
"""

from benchmarks._common import (
    DT,
    KERNEL_ROWS,
    PROCESS_NAMES,
    TIMING_ATTRS,
    VARIANT_ROWS,
    get_kernel,
    get_model,
    split_row,
)


class ProcessLifecycle:
    params = PROCESS_NAMES
    param_names = ["process"]

    number = TIMING_ATTRS["number"]
    repeat = TIMING_ATTRS["repeat"]
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = TIMING_ATTRS["warmup_time"]
    timeout = TIMING_ATTRS["timeout"]

    def time_init(self, name):
        proc = get_model().model_dict[name]
        type(proc)(proc._obj)

    def time_initialize(self, name):
        get_model().model_dict[name].initialize()

    def time_advance(self, name):
        get_model().model_dict[name].advance()

    def time_calculate(self, name):
        model = get_model()
        model.model_dict[name].calculate(DT, model.time)


class ProcessVariants(ProcessLifecycle):
    params = VARIANT_ROWS
    param_names = ["control/process"]

    def time_init(self, row):
        label, name = split_row(row)
        proc = get_model(label).model_dict[name]
        type(proc)(proc._obj)

    def time_initialize(self, row):
        label, name = split_row(row)
        get_model(label).model_dict[name].initialize()

    def time_advance(self, row):
        label, name = split_row(row)
        get_model(label).model_dict[name].advance()

    def time_calculate(self, row):
        label, name = split_row(row)
        model = get_model(label)
        model.model_dict[name].calculate(DT, model.time)


class ProcessKernel:
    params = KERNEL_ROWS
    param_names = ["control/process"]

    number = TIMING_ATTRS["number"]
    repeat = TIMING_ATTRS["repeat"]
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = TIMING_ATTRS["warmup_time"]
    timeout = TIMING_ATTRS["timeout"]

    def time_kernel(self, row):
        kernel, args = get_kernel(row)
        kernel(*args)
