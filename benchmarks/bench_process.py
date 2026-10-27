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
"""

from benchmarks._common import DT, PROCESS_NAMES, TIMING_ATTRS, get_model


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
