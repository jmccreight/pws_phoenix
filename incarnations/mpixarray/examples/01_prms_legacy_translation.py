# %% [markdown]
# # Running a PRMS model from its legacy files
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# In this notebook we'll run the complete National Hydrologic Model
# (NHM) with stream temperature on the Delaware River Basin, starting
# from nothing but the original PRMS input files:
#
# - `nhm_stream_temp.control` -- the PRMS control file
# - `myparam.param`           -- the PRMS parameter file
# - `prcp/tmax/tmin/rhavg.cbh` -- the ASCII climate-by-HRU forcings
#
# First we'll do it in a SINGLE CALL. Then we'll look under the hood
# at how the translation works, step by step, in case you're curious
# or need to adapt it to a configuration the one-liner doesn't cover.
#
# ## Prerequisites
#
# The `pwpx` environment (see `pws_phoenix/environment.yaml`), which
# includes `pyPRMS` -- all the PRMS file decoding goes through it.
# The verification section at the end also wants the generated PRMS
# answers (`drb_2yr/output_stream_temp/`); it politely skips if they
# are not present.

# %%
import pathlib as pl
import sys

import numpy as np
import xarray as xr

# as a script __file__ locates us; in a Jupyter kernel the cwd is the
# notebook's directory (JupyterLab default)
try:
    _here = pl.Path(__file__).parent
except NameError:
    _here = pl.Path.cwd()
_pkg = _here.parent
assert (_pkg / "model.py").exists(), (
    f"expected to run from incarnations/mpixarray/examples, not {_here}"
)
sys.path.append(str(_pkg))

from prms_translate import model_from_control

MPIX_ROOT = _pkg.parents[2]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
CONTROL_FILE = DOMAIN_DIR / "nhm_stream_temp.control"

# %% [markdown]
# ## The one-call setup
#
# A PRMS model is fully described by its control file: it names the
# parameter file, the forcing files, the simulation window, and which
# process representations to run. `model_from_control` reads all of
# that and hands back a ready-to-run model.

# %%
model = model_from_control(CONTROL_FILE)
print(f"assembled: {len(model.model_dict)} processes, ntime={model.ntime}")

# %% [markdown]
# That's the whole setup -- and note it wrote NOTHING to disk: the
# inputs were read, the model lives in memory, and the run below
# keeps its results there too (which is what we'll inspect). To
# write output, supply a store:
# `model_from_control(CONTROL_FILE, control={"output_serial_zarr":
# "run.zarr", "time_chunk_size": 61})` -- the control file's own
# requested output variables (`nhruOutVar_names` +
# `nsegmentOutVar_names`) are then honored, filtered to what this
# model has (`kit.dropped_output_var_names` lists the rest), or pass
# your own `"output_var_names": [...]` to override. Where to write
# is always YOUR decision -- the PRMS control's output paths are
# deliberately not used.
#
# Let's run the first five days (feel free to run the full two years
# -- change `n_days` to `model.ntime`) and look at the simulated
# stream temperature. The time step is one day, in seconds.

# %%
n_days = 5
model.run(np.float64(60.0 * 60.0 * 24.0), np.int32(n_days))
seg_tave_water = model.model_dict["prms_stream_temp"]["seg_tave_water"]
finite = np.isfinite(seg_tave_water.values)
print(
    f"after {n_days} days: seg_tave_water finite on {finite.sum()} of "
    f"{finite.size} segments; mean = "
    f"{seg_tave_water.values[finite].mean():.3f} degC"
)

# %% [markdown]
# A mean around zero degrees makes sense -- it's early January in the
# Delaware basin. One segment is not finite: the basin has a single
# segment with no upstream HRUs anywhere, so it never has flow (or a
# temperature); PRMS writes its -99.9 "invalid" sentinel there, and
# pws_phoenix marks it NaN. Let's verify both points against the PRMS
# (Fortran) answers directly rather than trusting the story.

# %%
answers_file = DOMAIN_DIR / "output_stream_temp" / "seg_tave_water.nc"
if answers_file.exists():
    answers = xr.load_dataarray(answers_file)
    ans_day = answers.values[n_days - 1, :]
    ours = seg_tave_water.values
    both = finite & (np.abs(ans_day - (-99.9)) > 1e-6)
    assert both.sum() == finite.sum()
    np.testing.assert_allclose(
        ours[both], ans_day[both], rtol=5e-3, atol=5e-3
    )
    print(
        f"day-{n_days} means -- ours: {ours[both].mean():.3f}, "
        f"PRMS: {ans_day[both].mean():.3f} degC "
        "(elementwise within 5e-3, the stream-temperature tolerance)"
    )
    wh_sentinel = np.where(~both)[0]
    print(
        f"and the non-finite segment (index {wh_sentinel}) is exactly "
        f"where PRMS writes {ans_day[wh_sentinel]}"
    )
else:
    print("(PRMS answers not generated; verification skipped)")

# %% [markdown]
# ## When the one-liner doesn't fit: the assembly kit
#
# `model_from_control` is a thin wrapper around
# `assemble_from_control`, which returns everything the `Model`
# constructor takes -- the ASSEMBLY KIT. If your configuration needs
# a tweak (different output settings, a swapped process, a parameter
# edit), get the kit, modify it, then build:

# %%
from prms_translate import assemble_from_control

kit = assemble_from_control(CONTROL_FILE)
print("processes:", list(kit.process_dict))
print(f"maps: {len(kit.maps)}, grids: {sorted(kit.discretizations)}")
print(f"window: {kit.config.start_time} .. {kit.config.end_time}")

# %% [markdown]
# This is the point where you would edit the kit -- change an entry
# in `kit.control`, swap a process class, adjust a parameter dataset
# -- and then build with `kit.model()`. We'll leave ours untouched
# and come back to build and run it at the end of the notebook, once
# we've seen what's inside.
#
# Configurations the assembler doesn't cover yet (for example PRMS's
# agricultural soil zone, or stream temperature driven by per-segment
# humidity parameters) raise a clear error naming what's unsupported
# -- nothing is ever silently substituted. In those cases the rest of
# this notebook shows how the kit is built, so you can build your own.
#
# ## Under the hood
#
# Let's rebuild the kit's contents step by step with the same
# functions the assembler uses, and check as we go that we get
# exactly what it produced.
#
# ### 1. The control file
#
# The control file's module names and flags choose the process
# classes (for instance `srunoff_smidx` selects `PRMSRunoff`, and
# `dprst_flag` decides whether surface-depression storage is
# included), and its `*_day` entries point at the forcing files.

# %%
from prms_translate import from_control, load_control

cfg = from_control(load_control(CONTROL_FILE), CONTROL_FILE)
print(f"window: {cfg.start_time} .. {cfg.end_time}")
for slot, cls in cfg.classes.items():
    print(f"  {slot:24s} -> {cls.__name__}")
print("forcings:", {kk: vv.name for kk, vv in cfg.cbh_paths.items()})

# %% [markdown]
# ### 2. The parameter file
#
# `load_parameters` reads `myparam.param` into one flat dataset --
# every parameter at full float64 precision, monthly parameters
# arranged `(month, hru)` (see `prms_translate/readers.py` if you're
# curious about the conventions).

# %%
from prms_translate import load_parameters

params = load_parameters(DOMAIN_DIR / "myparam.param")
params

# %% [markdown]
# ### 3. The computed parameters
#
# Three declared parameters are computed rather than read: the solar
# tables (from each HRU's slope, aspect, and latitude), the routing
# order of the stream segments (walked from the `tosegment`
# connectivity by the segment grid's `Discretization`), and a
# units-conversion field derived from HRU areas (handled inside the
# packaging step below).

# %%
from atmosphere.prms_solar_geometry import compute_soltabs
from discretization import Discretization

discretizations = {
    "nhru": Discretization(["nhru"]),
    "nsegment": Discretization(
        ["nsegment"],
        parameters=xr.Dataset({"tosegment": params["tosegment"]}),
        topo_order={"segment_order": "tosegment"},
    ),
}
soltabs = compute_soltabs(
    params[["hru_slope", "hru_aspect", "hru_lat"]], hru_dim="nhru"
)
soltabs

# %% [markdown]
# ### 4. Splitting parameters by what each process declares
#
# Every process class declares the parameters it needs -- so the flat
# file splits into per-process datasets by those declarations, with
# no hand-maintained lists. Let's check one against the kit.

# %%
from prms_translate import package_parameters

packaged = package_parameters(
    params,
    cfg.classes,
    extra={str(nn): soltabs[nn] for nn in soltabs.data_vars},
)
xr.testing.assert_identical(
    packaged["prms_snow"],
    kit.process_dict["prms_snow"]["parameters"],
)
print(
    "prms_snow parameters:",
    sorted(str(nn) for nn in packaged["prms_snow"].data_vars),
)

# %% [markdown]
# ### 5. The forcings
#
# Each ASCII CBH file becomes a float64 `(time, nhru)` array at the
# model's input name, sliced to the control window. (The humidity
# file is the input to the stream-temperature process, which lives on
# the segment grid -- more on that next.)

# %%
from prms_translate import load_cbh

prcp = load_cbh(cfg.cbh_paths["prcp"], cfg.start_time, cfg.end_time)
xr.testing.assert_identical(
    prcp, kit.process_dict["prms_atmosphere"]["prcp"]
)
prcp

# %% [markdown]
# ### 6. The Maps: coupling the HRU and segment grids
#
# The land-surface processes live on 765 HRUs; routing and stream
# temperature live on 456 segments. Maps carry variables between the
# grids: three lateral-inflow volumes aggregate to segments using the
# `hru_segment` assignments, and stream temperature needs ten more
# HRU quantities aggregated the way PRMS does internally. The
# humidity forcing crosses grids too, so a small "carrier" process
# owns it on the HRU side (`prms_translate.HumidityCarrier` -- the
# kit inserted it for us; it shows up in the process list above).

# %%
from prms_translate import volume_map_weights

vol_weights = volume_map_weights(params)
np.testing.assert_array_equal(
    vol_weights, kit.maps["sroff_vol"].weights
)
print(
    f"volume weights: {vol_weights.shape} (segments x HRUs), "
    f"{int(vol_weights.sum())} HRUs assigned; "
    f"kit maps: {len(kit.maps)} total"
)

# %% [markdown]
# ### 7. Putting it together
#
# The kit is exactly these pieces -- a process dictionary (classes +
# their parameters, initial values, and forcings), the two grids, and
# the maps. `kit.model()` is just the `Model` constructor, which
# validates the whole assembly as it builds. Let's close the loop:
# build the model from the kit we've been checking against, run the
# same five days, and confirm the result is IDENTICAL to our
# one-liner run at the top -- bit for bit.

# %%
model_from_kit = kit.model()
model_from_kit.run(np.float64(60.0 * 60.0 * 24.0), np.int32(n_days))
np.testing.assert_array_equal(
    model_from_kit.model_dict["prms_stream_temp"][
        "seg_tave_water"
    ].values,
    seg_tave_water.values,
)
print(
    "kit.model() run matches the one-liner run bit-for-bit "
    "(and therefore the PRMS answers above)"
)

# %% [markdown]
# ## A parallel run from the same kit
#
# Everything so far lived in memory. A PARALLEL run cannot: the
# distributed (HRU) grid is STREAMED from one combined NetCDF file on
# disk, decomposed across the MPI ranks, while the segment grid is
# rebuilt identically on every rank. So going parallel introduces two
# disk requirements: the combined input file, and a NetCDF file for
# the streamed output. Both live in this notebook's data directory
# (the `01_prms_legacy_translation/` convention -- each notebook
# writes only beside itself).
#
# The kit writes the input file for us. We'll truncate it to our five
# days -- the stream defines the run length.

# %%
from prms_translate import write_mpi_input_file

data_dir = _here / "01_prms_legacy_translation"
data_dir.mkdir(exist_ok=True)
input_file = write_mpi_input_file(
    kit, data_dir / "mpi_input.nc", n_days=n_days
)
mb = input_file.stat().st_size / 1e6
print(f"wrote {input_file.name}: {mb:.1f} MB, {n_days} days")

# %% [markdown]
# An MPI model must be built UNDER mpirun -- every rank constructs it
# together -- so the parallel one-liner
# (`prms_translate.mpi_model_from_control`) is called from a small
# script rather than a notebook cell. Here is the entire script; we
# write it into the data directory and launch it on two ranks.
#
# Output is ROUTED BY THE GRID THAT OWNS EACH VARIABLE, and we'll
# demonstrate both routes: `seg_tave_water` (the stream temperature
# we verified above) lives on the replicated segment grid, so rank 0
# collects it into a zarr store -- everyone computes, one writes;
# `sroff` (surface runoff) lives on the distributed HRU grid, so it
# STREAMS to the parallel NetCDF, each rank writing its own block --
# the genuinely parallel IO path. (Today at most one distributed
# variable can stream; segment-grid output is not limited.)

# %%
import shutil
import subprocess

runner = f"""\
import sys

sys.path.append({str(_pkg)!r})
import numpy as np

from prms_translate import mpi_model_from_control

model = mpi_model_from_control(
    {str(CONTROL_FILE)!r},
    control={{
        "input_file": {str(input_file)!r},
        "output_parallel_netcdf": {str(data_dir / "mpi_output.nc")!r},
        "output_serial_zarr": {str(data_dir / "mpi_output.zarr")!r},
        "output_var_names": ["seg_tave_water", "sroff"],
        "time_chunk_size": {n_days},
        "mpi_grid": "nhru",
    }},
)
model.run(np.float64(60.0 * 60.0 * 24.0))
model.finalize()
"""
runner_path = data_dir / "run_mpi.py"
runner_path.write_text(runner)
print(runner)

# %%
mpirun = shutil.which("mpirun")
if mpirun is None:
    print("(mpirun not found; parallel demonstration skipped)")
else:
    result = subprocess.run(
        [mpirun, "-n", "2", sys.executable, str(runner_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("mpirun failed")
    print("parallel run complete")

# %% [markdown]
# Now read the parallel run's stream temperature back and compare it
# with our serial run. One nuance worth knowing: the HRU-to-segment
# aggregations sum each rank's partial contribution in a different
# order than the serial matrix product, and floating-point addition
# is order-sensitive -- so the parallel temperatures agree to within
# rounding (~1e-9 here), not bit-for-bit.

# %%
if mpirun is not None:
    with xr.open_zarr(
        data_dir / "mpi_output.zarr", consolidated=False
    ) as ds_mpi:
        seg_tave_mpi = ds_mpi["seg_tave_water"].values
    np.testing.assert_allclose(
        seg_tave_mpi[n_days - 1, :],
        seg_tave_water.values,
        rtol=1e-9,
        atol=1e-9,
    )
    print(
        f"parallel seg_tave_water (day {n_days}, all segments) "
        "matches the serial run to ~1e-9 -- the same field we "
        "verified against PRMS above"
    )

# %% [markdown]
# And the streamed route: the parallel NetCDF holds `sroff` at full
# extent, the ranks' blocks written side by side. Runoff is computed
# HRU-locally -- no communication touches it -- so here the match to
# the serial run IS bit-for-bit.

# %%
if mpirun is not None:
    with xr.open_dataset(data_dir / "mpi_output.nc") as ds_stream:
        sroff_mpi = ds_stream["sroff_out"].values  # (time, nhru)
    np.testing.assert_array_equal(
        sroff_mpi[n_days - 1, :],
        model.model_dict["prms_runoff"]["sroff"].values,
    )
    print(
        "parallel sroff (final day, all HRUs) matches the serial "
        "run bit-for-bit"
    )

# %% [markdown]
# ## Where to go next
#
# - `00_input_contract.py` (this directory) explores the INPUT CONTRACT
#   -- the machine-readable statement of everything a model
#   configuration requires -- which is what drives the parameter
#   splitting and supply you saw here.
# - For restart/warm-start capability, see `Model.write_restart` and
#   the `restart_read` control option (`help(Model)`).
# - Developers: the translation package's design notes and its
#   pyPRMS groundwork live in `prms_translate/`'s module docstrings
#   and `PORTS.md`.
