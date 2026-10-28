"""Whole-model numbers: the full step, the build, and peak memory.

    time_chain_step      get_model().run(DT, 1)   all advance + calculate
                                                  + 13 Map applies + Input
                                                  stepping; number = 10 =
                                                  a 10-day sequence per sample
    time_model_build     build_model()            parameter loading, per-grid
                                                  dataset assembly, binding,
                                                  init hooks, Map resolution
    peakmem_imports      (nothing)                RSS after the imports --
                                                  the baseline for the others
    peakmem_model_build  build_model()            RSS after one build
    peakmem_chain_warm   get_model()              RSS after build + 1 step:
                                                  the numba compile of every
                                                  kernel is in, no sequence
    peakmem_chain_run    get_model().run(DT, 30)  RSS after build + 31 steps;
                                                  growth over chain_warm =
                                                  per-step allocation -- but
                                                  RSS noise is ~50 MB, so:
    track_step_alloc_bytes                        tracemalloc net bytes
                                                  retained over 30 steps on
                                                  the warm model, byte-exact;
                                                  the prime directive says ~0

time_chain_step minus the nine (time_advance + time_calculate) of
bench_process = the framework overhead (Maps, Inputs, loop). The
StepBreakdown rows split it, each on the warm model, number = 10,
none consuming the time axis:

    time_maps            the 13 mm.apply(...) as Model.calculate issues
                         them (no process calculate)
    time_inputs_advance  Input.set_index(current) + Input.advance() per
                         Input: the rewind makes it drift-free
    time_clock           Time.set_index + current_time[0] = time.current
                         + the five calendar properties once each
    time_advance_all     the nine pp.advance()
    time_calculate_all   the nine vv.calculate(dt, time), no Maps

time_chain_step - (the five) = the loop residual.

Step budget: the shared model cannot be rewound and drb_2yr has 731
steps. time_chain_step consumes number x (warm-up iterations + repeat)
steps -- with number = 10, warmup_time = 2 s at ~9 ms/step that is
roughly 10 x (24 + 10) = 340 steps. Keep it under 731 or Model.run
indexes past the end of the time axis.
"""

import tracemalloc

import numpy as np

from benchmarks._common import DT, TIMING_ATTRS, build_model, get_model


class ChainStep:
    number = 10
    repeat = TIMING_ATTRS["repeat"]
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = TIMING_ATTRS["warmup_time"]
    timeout = TIMING_ATTRS["timeout"]

    def time_chain_step(self):
        get_model().run(DT, np.int32(1))


class StepBreakdown:
    """The framework share of time_chain_step, piece by piece (module
    docstring). Every row runs on the warm shared model and leaves its
    time index where it found it."""

    number = 10
    repeat = TIMING_ATTRS["repeat"]
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = TIMING_ATTRS["warmup_time"]
    timeout = TIMING_ATTRS["timeout"]

    def time_maps(self):
        model = get_model()
        for proc_name in model.model_dict:
            for mm in model._proc_maps[proc_name]:
                mm.apply()

    def time_inputs_advance(self):
        model = get_model()
        idx = model.time.current_index
        for ii in model.inputs_dict.values():
            ii.set_index(idx)
            ii.advance()

    def time_clock(self):
        model = get_model()
        tt = model.time.current_index
        model.time.set_index(tt)
        model.current_time[0] = model.time.current
        time = model.time
        _ = (time.month, time.year, time.day_of_month, time.doy, time.dowy)

    def time_advance_all(self):
        for pp in get_model().model_dict.values():
            pp.advance()

    def time_calculate_all(self):
        model = get_model()
        for vv in model.model_dict.values():
            vv.calculate(DT, model.time)


class ModelBuild:
    # one build per sample, no warm-up: the first build IS the
    # measurement of interest (numba compile is not involved -- no
    # kernel runs at build)
    number = 1
    repeat = 5
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = 0
    timeout = TIMING_ATTRS["timeout"]

    def time_model_build(self):
        build_model()


class PeakMemory:
    timeout = TIMING_ATTRS["timeout"]

    def peakmem_imports(self):
        """The baseline: importing this module already imported
        pws_phoenix (+ numpy, xarray, numba, pyPRMS), so the subprocess's
        peak RSS at this point is the cost of the imports alone."""

    def peakmem_model_build(self):
        build_model()

    def peakmem_chain_warm(self):
        get_model()

    def peakmem_chain_run(self):
        get_model().run(DT, np.int32(30))


class StepAllocation:
    timeout = TIMING_ATTRS["timeout"]
    unit = "bytes"

    def track_step_alloc_bytes(self):
        """Net bytes still allocated after 30 steps of the warm model
        (numpy arrays included: numpy reports to tracemalloc). A value
        that grows with the step count is a per-step allocation.

        Warm-up is 5 MORE steps here, not get_model()'s one: kernel
        paths first reached after step 1 (season/flag-dependent
        branches, e.g. stream-temp shade) compile late, and that
        one-time numba work measured as 7.9 MB over steps 2-31 on the
        first run (Oct 2026); steps 5-35 then retained a few kB."""
        model = get_model()
        model.run(DT, np.int32(5))
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        model.run(DT, np.int32(30))
        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        return after - before
