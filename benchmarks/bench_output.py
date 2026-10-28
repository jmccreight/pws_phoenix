"""The serial zarr writer (data_io.Output) alone -- the IO half of the
compute-vs-IO split.

    time_output_zarr   Output(...) + initialize_variable_tracking +
                       collect_current_timestep x N_STEPS + finalize

with time_chunk_size == N_STEPS, so one call = N_STEPS buffer copies
+ ONE full-chunk append (which creates the store) + a finalize with
an empty tail. The variables are CONTROL's own output request (103
hru + 31 segment names, filtered to what the model has). Model state
is NOT advanced between collects: IO cost in isolation.

This is the one place a `setup`/`teardown` pair is used: a fresh
store directory per call (cheap, and ASV re-runs setup before every
timed call anyway -- README section 3). The MPI output path
(mpixarray's streamed NetCDF) is not timed; the MPI path is outside
the first cut.
"""

import pathlib as pl
import shutil
import tempfile

from pws_phoenix.data_io import Output

from benchmarks._common import TIMING_ATTRS, get_model, get_output_var_names

N_STEPS = 30


class OutputZarr:
    number = 1
    repeat = TIMING_ATTRS["repeat"]
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = TIMING_ATTRS["warmup_time"]
    timeout = TIMING_ATTRS["timeout"]

    def setup(self):
        self.tmp_dir = pl.Path(tempfile.mkdtemp(prefix="pws_phoenix_asv_"))

    def teardown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def time_output_zarr(self):
        model = get_model()
        output = Output(
            time_chunk_size=N_STEPS,
            variable_names=get_output_var_names(),
            output_store=self.tmp_dir / "out.zarr",
            time_values=model.times.values,
        )
        output.initialize_variable_tracking(model.model_dict)
        for tt in range(N_STEPS):
            output.collect_current_timestep(tt)
        output.finalize()
