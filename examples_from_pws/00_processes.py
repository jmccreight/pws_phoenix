# %% [markdown]
# # Processes
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# The atomic unit of modeling in `pws_phoenix` is the process
# representation: `Process` is the class every physical process
# (groundwater, snow, canopy, ...) is built on. In this notebook we'll
# take one process, the PRMS groundwater reservoir, and run it alone
# on the Delaware River Basin for half a year. Along the way we'll see
# how a process handles
#
# * time
# * the discretization (the grid)
# * parameters
# * inputs
# * variables and their initial values
# * options
# * advancing and calculating
# * output to disk
# * mass balance (budget) tracking
# * finalization
#
# and then how a process *describes itself* in code: its parameters,
# inputs, variables, budget terms and restart state, all read from
# its declarations.
#
# This is a translation of pywatershed's notebook `00_processes.ipynb`;
# the last section checks our results against that notebook's output.
#
# ## Prerequisites
#
# The `pwpx` environment (see `pws_phoenix/environment.yaml`). The
# domain files are the small Delaware River Basin example that
# pywatershed ships inside its package, read from the pywatershed
# clone beside this repository. Plots need `matplotlib` (optional).

# %%
import inspect
import pathlib as pl
import pydoc
from pprint import pprint

import numpy as np
import xarray as xr

# as a script __file__ locates us; in a Jupyter kernel the cwd is the
# notebook's directory (JupyterLab default). The package itself is
# installed (`pip install -e .`); this only finds the data.
try:
    _here = pl.Path(__file__).parent
except NameError:
    _here = pl.Path.cwd()
_repo = _here.parent
assert (_repo / "pyproject.toml").exists(), (
    f"expected to run from pws_phoenix/examples_from_pws, not {_here}"
)

from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_groundwater import PRMSGroundwater
from pws_phoenix.model import Model

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None  # type: ignore[assignment]
    print("(matplotlib not installed; plots skipped)")

# %% [markdown]
# ## Where things are
#
# The domain files (the HRU discretization, the groundwater
# parameters and the three inflow time series) come from pywatershed's
# package data. Everything this notebook writes goes into its own data
# directory beside it.

# %%
PYWS_ROOT = _repo.parent / "pywatershed"
DOMAIN_DIR = PYWS_ROOT / "pywatershed" / "data" / "drb_2yr"
PYWS_NB_OUTPUT_DIR = PYWS_ROOT / "examples" / "00_processes"
if not DOMAIN_DIR.exists():
    raise FileNotFoundError(
        f"{DOMAIN_DIR} -- the pywatershed clone beside this repository "
        "supplies the drb_2yr domain files"
    )

nb_dir = _here / "00_processes"
nb_dir.mkdir(exist_ok=True)
print(f"domain: {DOMAIN_DIR}\noutputs: {nb_dir}")

# %% [markdown]
# ## How a Process works
#
# We'll pick a simple, conceptual groundwater representation from PRMS
# as a case study: one linear reservoir per hydrologic response unit
# (HRU), fed from the soil zone and draining to the stream network and
# to a sink.
#
# To get started, we can ask for `help()`. The full help is long, so
# we'll look at just the first lines.

# %%
# pydoc.plaintext: no bold over-striking (it is missing from the
# type stubs, hence the ignore)
prms_gw_help = pydoc.render_doc(
    PRMSGroundwater,
    "Help on %s",
    renderer=pydoc.plaintext,  # type: ignore[attr-defined]
)
print("\n".join(prms_gw_help.splitlines()[0:22]))

# %% [markdown]
# The first line names the module the class lives in; the next says
# that `PRMSGroundwater` extends `PRMSGroundwaterNoDprst` (the same
# reservoir without the surface-depression seepage inflow), which is
# itself a `Process`. Then comes the class's own description.
#
# What you won't find is a long constructor signature: a `Process` is
# never built by hand. You describe it in a dictionary entry (its
# class, its grid, its parameters and inputs) and hand that to a
# `Model`, which allocates the process's variables on the grid's
# shared dataset and drives it in time. Even for one process we build
# a one-process `Model`: it owns the clock, the output and the
# mass-budget check. We'll assemble that entry piece by piece.
#
# *Time:*
# Hydrologic processes are generally prognostic, meaning that the next
# state depends on the current state ($X_{t+1}(X_t)$). The clock
# (`pws_phoenix.clock.Time`) is owned by the `Model` and handed to each
# process at every step. There is nothing to construct: the time axis
# IS the time axis of the inputs we'll supply (daily, here the two
# years 1979-1980), and we choose how many of its steps to run. Like
# pywatershed's notebook we'll run the first half of 1979.

# %%
START = np.datetime64("1979-01-01")
END = np.datetime64("1979-07-01")
# pywatershed counts both end days
n_days = int((END - START) / np.timedelta64(1, "D")) + 1
dt = np.float64(24 * 60 * 60)  # the timestep length in seconds
print(f"{n_days} daily steps, {START} to {END}")

# %% [markdown]
# *Discretization:*
# The static data describing the spatial aspect of the process: the
# HRUs, their areas and ids. We'll load these from pywatershed's
# `parameters_dis_hru.nc` into a `Discretization`, whose one dimension
# we name `nhru`.

# %%
dis_file = DOMAIN_DIR / "parameters_dis_hru.nc"
assert dis_file.exists()
dis_hru = Discretization(["nhru"], parameters=dis_file)
dis_hru.parameters

# %% [markdown]
# *Parameters:*
# The static values the process uses beyond those in the
# discretization: here the routing and sink coefficients and the
# initial storage. A parameter set is just an `xarray.Dataset`; we
# load pywatershed's file for this process.

# %%
param_file = DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
assert param_file.exists()
params = xr.load_dataset(param_file)
params

# %% [markdown]
# *Inputs:*
# `soil_to_gw`, `ssr_to_gw` and `dprst_seep_hru` are the time-varying
# inflows to the reservoir, normally computed by the soil zone and
# runoff processes. Running groundwater alone, we read them from files
# (pywatershed wrote these from a full model run). One detail:
# pywatershed's output files put variables on a dimension named
# `nhm_id`; our grid's dimension is `nhru`, and the `Model` insists
# that an input's spatial dimension be the grid's, so we rename.

# %%
input_names = PRMSGroundwater.get_inputs()
print(input_names)
inputs = {
    nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
    for nn in input_names
}
inputs["soil_to_gw"]

# %% [markdown]
# *Options:*
# Run-wide settings live in a plain dictionary, the `control`. Three
# matter here:
#
# * `budget`: this process declares a mass budget (inputs, outputs and
#   storage changes that must balance every step). `"error"` raises if
#   the budget does not close; `"warn"` reports; `None` computes and
#   writes the terms without judging them. We'll elect to error.
# * `output_serial_zarr` and `output_var_names`: where output goes (a
#   zarr store) and which variables are written. We'll write all of the
#   process's variables plus the budget terms the model generates from
#   the budget declaration (`get_budget_var_names()`).
# * `time_chunk_size`: how many steps are buffered in memory before a
#   chunk is written to the store.
#
# (pywatershed's `calc_method` and `verbose` have no counterpart:
# numba is the one compute path.)

# %%
budget_var_names = list(PRMSGroundwater.get_budget_var_names())
print(budget_var_names)
control = {
    "budget": "error",
    "output_serial_zarr": nb_dir / "prms_groundwater.zarr",
    "output_var_names": list(PRMSGroundwater.get_var_names())
    + budget_var_names,
    "time_chunk_size": 61,
}

# %% [markdown]
# *Initial values:*
# Variables that carry state need a starting value. The process says
# which parameter supplies it (`gwres_stor` starts from `gwstor_init`,
# as in PRMS); we pass that array under the parameter's name.
#
# Putting it all together: the process entry names its class, its
# grid, its parameters, its initial values and its inputs.

# %%
process_dict = {
    "prms_groundwater": {
        "class": PRMSGroundwater,
        "discretization": "nhru",
        "parameters": params,
        "gwstor_init": params["gwstor_init"],
        **inputs,
    },
}

model = Model(process_dict, control, discretizations={"nhru": dis_hru})
prms_gw = model.model_dict["prms_groundwater"]
print(type(prms_gw).__name__, "on", model.ntime, "available time steps")

# %% [markdown]
# Now we have an instance of a `PRMSGroundwater`, bound to the `nhru`
# grid's dataset. That dataset holds every array the process touches:
# the discretization's variables, the parameters, one time slice of
# each input, and the process's variables, including the five budget
# variables the model generated from the budget declaration. Let's
# take a look.

# %%
model.discretizations["nhru"].dataset

# %% [markdown]
# No arrays were copied to build this: the parameter dataset you
# loaded above IS the model's working memory (which is why the model
# marks parameters read-only).
#
# Now we are ready to simulate. A timestep is `model.step(dt)`; here is
# exactly what one does.

# %%
print(inspect.getsource(Model.step))

# %% [markdown]
# So a step points the clock at the next time, *advances* every input
# and process (inputs serve their next slice; a process copies its
# current state to its `*_old`/`*_previous` variables), *calculates*
# every process in order (the physics, then the budget check), and
# collects output. Let's take one step and look at the state.

# %%
model.step(dt)
print("clock:", model.time.current)
print("gwres_stor_old[:3]:", prms_gw["gwres_stor_old"].values[:3])
print("gwres_stor[:3]:    ", prms_gw["gwres_stor"].values[:3])
print("gwres_flow[:3]:    ", prms_gw["gwres_flow"].values[:3])

# %% [markdown]
# Between steps the state is yours to inspect (or, carefully, to
# change: the arrays above are the live buffers). `model.run` is simply
# a loop of steps, continuing from wherever the model is; we run the
# remaining days.

# %%
model.run(dt, np.int32(n_days - 1))
print("clock:", model.time.current)

# %% [markdown]
# No errors arose, so the mass balance held for the entire run (every
# step was checked, because we chose `budget = "error"`). The budget's
# terms are ordinary variables on the grid dataset; the residual is
# `inputs - (outputs + storage_changes)` for the last step, and
# `residual_accum` its running sum.

# %%
for nn in budget_var_names:
    print(f"{nn:44s} max |.| = {np.abs(prms_gw[nn].values).max():.2e}")

# %% [markdown]
# We can see from the clock that we reached the end time, 1979-07-01.
#
# Now we finalize the model. This writes the last, partial chunk of
# output and closes the files; the in-memory state stays available.

# %%
model.finalize()

# %% [markdown]
# Let's take a look at the output that was written to the store: one
# zarr group with every requested variable as a `(time, nhru)` array.

# %%
# a half-year of 765 HRUs is small: read it all into memory
output_ds = xr.open_zarr(
    control["output_serial_zarr"], consolidated=False
).load()
pprint(sorted(output_ds.data_vars))
output_ds

# %% [markdown]
# The budget terms are in there too, so we can look at the balance
# through time: the inputs, the outputs and the storage change summed
# over the basin, and the residual (roundoff).

# %%
basin_sum = output_ds[budget_var_names[:3]].sum(dim="nhru").to_dataframe()
basin_sum.columns = ["inputs", "outputs", "storage_change"]
print(basin_sum.describe().loc[["mean", "max"]])
print(
    "max |residual| over the run:",
    f"{np.abs(output_ds['PRMSGroundwater_mass_residual']).max().item():.2e}",
)


# %%
def plot_series(series, ylabel, title, fname):
    """Line plot of {label: DataArray(time)}; saved beside the
    notebook, returned for inline display."""
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, da in series.items():
        ax.plot(da["time"].values, da.values, linewidth=1.5, label=label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(nb_dir / fname, dpi=120)
    print(f"saved {fname}")
    return fig


plot_series(
    {
        "inputs": output_ds[budget_var_names[0]].sum(dim="nhru"),
        "outputs": output_ds[budget_var_names[1]].sum(dim="nhru"),
        "storage change": output_ds[budget_var_names[2]].sum(dim="nhru"),
    },
    "inches (summed over HRUs)",
    "PRMSGroundwater mass budget terms",
    "mass_budget.png",
)

# %% [markdown]
# Let's take a look at the main output flux, `gwres_flow`, on a few
# HRUs. (pywatershed's notebook shows an interactive plot with an HRU
# slider; feel free to change the ids below.)

# %%
var_da = output_ds["gwres_flow"]
assert dis_hru.parameters is not None
hru_ids = dis_hru.parameters["nhm_id"].values
plot_series(
    {
        f"nhm_id {hru_ids[ii]}": var_da.isel(nhru=ii)
        for ii in (0, 100, 400, 700)
    },
    f"{var_da.name} [{PRMSGroundwater.get_variables()[var_da.name].units}]",
    "Groundwater discharge to the stream network",
    "gwres_flow.png",
)

# %% [markdown]
# pywatershed's notebook ends this section with a map of the mean
# `gwres_flow` on the HRU polygons. The GIS files and plotting helpers
# are not part of `pws_phoenix` yet; see that notebook for the map.
#
# ## Process design
#
# That's great! But you may have a few questions.
#
# * Why did we get those output variables?
# * How were the terms in the mass budget decided or known?
# * What are the parameters that were in that parameter file?
# * What are the units of the inputs?
#
# The answer is that a `Process` is self-describing in code: every
# parameter, input and variable is declared on the class with its
# kind, dimensions, type, description and units, and the budget is a
# declaration too. The class methods below read those declarations.

# %%
print("parameters:", PRMSGroundwater.get_parameters())
print("inputs:    ", PRMSGroundwater.get_inputs())
print("variables: ", PRMSGroundwater.get_var_names())

# %%
pprint(PRMSGroundwater.budgets)

# %% [markdown]
# This dictionary exactly specifies how the budget is calculated and
# is what the developer of any `Process` writes. (Compared with
# pywatershed's term list, `gwres_sink` is an output here: the kernel
# removes it from storage, so the budget closes to roundoff.)
#
# The variables carry their metadata:

# %%
for name, meta in PRMSGroundwater.get_variables().items():
    print(f"{name:18s} {meta.units or '':8s} {meta.description}")

# %% [markdown]
# These are the variables the process maintains, and by default none
# are written: `output_var_names` chooses. Only the current time (and,
# for prognostic variables, the previous time) is held in memory. The
# budget terms are kept apart (`kind="budget_variable"`), with the
# same metadata:

# %%
for name, meta in PRMSGroundwater.get_budget_variables().items():
    print(f"{name:42s} {meta.units or '':8s} {meta.description}")
#
# The declarations also say which variables are the prognostic state,
# the minimal set a restart file must hold:

# %%
print(PRMSGroundwater.get_restart_variables())

# %% [markdown]
# Put together, the declarations of a whole configuration are its
# *input contract*: what must be supplied to run it. `Model.input_spec`
# answers that from the process dictionary alone, without loading any
# data. For our one process:

# %%
pprint(Model.input_spec(process_dict)["required"]["nhru"])

# %% [markdown]
# The `examples/00_input_contract.py` notebook takes this further for
# the full NHM chain. These self-describing capabilities help users
# and programmers get answers about processes, and they let generic
# code connect multiple processes: the model checks every input is
# produced by another process, mapped from another grid, or supplied.
# Multiple-process models are explored in the next notebook.
#
# ## Checking against pywatershed
#
# pywatershed's notebook wrote its output beside itself; if it's
# there, we compare every state and flux variable with it over the
# common days (the budget terms are ours alone: pywatershed writes its
# budget in a different form). The groundwater reservoir is a linear store computed in the
# same order of operations, so the two agree to roundoff: the check
# below is at `1e-13`.

# %%
RTOL = ATOL = 1.0e-13
if not PYWS_NB_OUTPUT_DIR.exists():
    print(f"({PYWS_NB_OUTPUT_DIR} not found; verification skipped)")
else:
    for name in PRMSGroundwater.get_var_names():
        theirs = xr.load_dataarray(PYWS_NB_OUTPUT_DIR / f"{name}.nc")
        ours = output_ds[name]
        common = np.intersect1d(
            ours["time"].values.astype("datetime64[ns]"),
            theirs["time"].values.astype("datetime64[ns]"),
        )
        oo = ours.sel(time=common).values
        tt = theirs.sel(time=common).values
        assert oo.shape == tt.shape, (name, oo.shape, tt.shape)
        max_abs = np.abs(oo - tt).max()
        np.testing.assert_allclose(oo, tt, rtol=RTOL, atol=ATOL, err_msg=name)
        print(f"{name:18s} {len(common)} days, max |diff| = {max_abs:.1e}")

# %% [markdown]
# ## What just happened
#
# We described one process in a dictionary entry, built a one-process
# `Model` on an HRU discretization, stepped it once by hand, ran the
# remaining days, checked the mass budget every step, wrote the output
# to a zarr store and read it back -- and found the results identical
# to pywatershed's to roundoff. Everything the model needed to know
# about the process, it read from the process's own declarations.
#
# Where to go next: the next notebook connects several processes on
# one grid and across grids; `examples/00_input_contract.py` shows the
# input contract of the full NHM; `help(Model)` and `help(Process)`
# have the details.
