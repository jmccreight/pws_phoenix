# %% [markdown]
# # A STARFIT reservoir in the channel network: Big Sandy Reservoir
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# In this notebook we'll take the NHM channel network on the Flaming
# Gorge domain (the Green River in Wyoming) and insert a single new
# node into it: a STARFIT representation of the Big Sandy Reservoir.
# The channel network becomes a `FlowGraph` -- a routing network whose
# nodes can be of different types (Muskingum-Mann channel segments,
# reservoirs, pass-throughs, ...). We'll run the model three ways:
#
# 1. the NHM as-is (no reservoir), for a baseline;
# 2. the NHM with the reservoir inserted, as one model;
# 3. the `FlowGraph` alone, driven by the baseline run's output --
#    a "post-process" that lets you try reservoir designs without
#    rerunning everything upstream.
#
# This is a translation of pywatershed's notebook
# `06_flow_graph_starfit.ipynb`; the last section checks our results
# against that notebook's output.
#
# ## Prerequisites
#
# The `pwpx` environment (see `pws_phoenix/environment.yaml`). The
# Flaming Gorge domain is downloaded on first use (a ~zip from the
# pywatershed releases). The reservoir's STARFIT parameters and the
# verification targets are read from the pywatershed clone beside
# this repository; those cells skip politely if it is not there.
# Plots need `matplotlib` (optional).

# %%
import datetime
import pathlib as pl

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

from pws_phoenix.utils.addtl_domain_files import download_addtl_domains

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:  # plotting is optional
    HAVE_MATPLOTLIB = False
    print("(matplotlib not installed; plots skipped)")

S_PER_DAY = np.float64(60.0 * 60.0 * 24.0)

# %% [markdown]
# ## Where things are
#
# The domain files (PRMS control, parameters, climate forcings) come
# from the additional-domains asset; `download_addtl_domains` fetches
# and unpacks it once into `pws_phoenix/data/` and just returns the
# path on later calls. Everything this notebook writes goes into its
# own data directory beside it.

# %%
DOMAIN_DIR = download_addtl_domains() / "fgr_2yr"
CONTROL_FILE = DOMAIN_DIR / "nhm.control"

PYWS_ROOT = _repo.parent / "pywatershed"
BIG_SANDY_PARAM_FILE = (
    PYWS_ROOT / "pywatershed" / "data" / "big_sandy_starfit_parameters.nc"
)
PYWS_NB_OUTPUT_DIR = PYWS_ROOT / "examples" / "06_flow_graph_starfit"

nb_dir = _here / "06_flow_graph_starfit"
nb_dir.mkdir(exist_ok=True)
print(f"domain: {DOMAIN_DIR}\noutputs: {nb_dir}")

# %% [markdown]
# ## The Big Sandy Reservoir
#
# STARFIT (Turner et al., 2021) describes a reservoir's operations by
# seasonal rules fitted to observed storage and release: a "normal
# operating range" of storage and a release function around it. The
# parameters for Big Sandy (GRanD id 419) come from the ISTARF-CONUS
# data set; pywatershed ships this one reservoir's row as a small
# file, which we read here. Let's take a look.

# %%
if not BIG_SANDY_PARAM_FILE.exists():
    raise FileNotFoundError(
        f"{BIG_SANDY_PARAM_FILE} -- the pywatershed clone beside this "
        "repository supplies the reservoir parameters"
    )
big_sandy = xr.load_dataset(BIG_SANDY_PARAM_FILE).squeeze(drop=True)
big_sandy.to_pandas()

# %% [markdown]
# The reservoir capacity (`GRanD_CAP_MCM`) is 67.1 million cubic
# meters. On the map (see the pywatershed notebook for an interactive
# one) Big Sandy sits just above NHM segment 44426 and below segments
# 44434 and 44435: that is where we'll insert it.

# %%
NHM_SEG_BELOW_RESERVOIR = 44426

# %% [markdown]
# ## 1. The NHM as-is: no reservoir
#
# The NHM does not represent reservoirs. Let's first see how it
# simulates flow at segment 44426 without one. A PRMS model is fully
# described by its control file, so one call assembles the whole
# model (see `examples/01_prms_legacy_translation.py` for what
# happens inside). We ask for the segment outflow and -- for the
# post-process in section 3 -- the three lateral inflow volumes that
# feed the channel. `"budget": "error"` makes every process check
# its mass balance each day and stop on a violation.
#
# One wrinkle: this control file is PRMS's generic template, so its
# forcing-file entries name `precip.day`, `tmax.day`, `tmin.day` --
# while the domain ships `prcp.cbh`, `tmax.cbh`, `tmin.cbh`. (pywatershed
# never reads those entries; it looks for NetCDF forcings in
# `input_dir`.) `cbh_paths` points the assembler at the actual files.

# %%
from pws_phoenix.prms_translate import assemble_from_control

CBH_PATHS = {nn: DOMAIN_DIR / f"{nn}.cbh" for nn in ("prcp", "tmax", "tmin")}

nhm_store = nb_dir / "fgr_nhm.zarr"
kit_nhm = assemble_from_control(
    CONTROL_FILE,
    cbh_paths=CBH_PATHS,
    control={
        "budget": "error",
        "output_serial_zarr": nhm_store,
        "output_var_names": [
            "seg_outflow",
            "sroff_vol",
            "ssres_flow_vol",
            "gwres_flow_vol",
        ],
        "time_chunk_size": 365,
    },
)
print("processes:", list(kit_nhm.process_dict))
print(f"grids: {sorted(kit_nhm.discretizations)}, maps: {len(kit_nhm.maps)}")

# %% [markdown]
# The control file's window runs to 2022 but the forcing files cover
# 1979-01-01 to 1980-12-31, and the model's length follows the data.
# We'll run all of it. (Rerunning is skipped if the output store is
# already there -- delete it to rerun.)

# %%
if nhm_store.exists():
    print(f"{nhm_store.name} exists; not rerunning")
else:
    with kit_nhm.model() as model:
        n_days = model.ntime
        model.run(S_PER_DAY, np.int32(n_days))
    print(f"ran {n_days} days")

nhm_out = xr.open_zarr(nhm_store, consolidated=False).load()
nhm_out

# %% [markdown]
# The segment grid is labeled by position; the NHM segment ids live
# on the segment grid's parameters. Let's pull out the outflow at
# segment 44426 and plot it.

# %%
seg_dis = kit_nhm.discretizations["nsegment"].parameters
assert seg_dis is not None
nhm_seg = seg_dis["nhm_seg"].values
wh_44426 = int(np.where(nhm_seg == NHM_SEG_BELOW_RESERVOIR)[0][0])
outflow_nhm = nhm_out["seg_outflow"].isel(nsegment=wh_44426).rename("NHM")


def plot_series(series, ylabel, title, fname):
    """Line plot of {label: DataArray(time)}; saved beside the
    notebook, returned for inline display."""
    if not HAVE_MATPLOTLIB:
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
    {"NHM": outflow_nhm},
    "streamflow (cfs)",
    f"NHM outflow at segment {NHM_SEG_BELOW_RESERVOIR}",
    "nhm_outflow_44426.png",
)

# %% [markdown]
# How does the NHM do against observations here? The domain ships
# its streamflow gages (`sf_data`), and the parameter file says which
# segment each gage sits on (`poi_gage_segment`, a 1-based segment
# index). We'll compare the gage listed at segment index 184 -- the
# Green River below Big Sandy -- with the modeled outflow there. The
# PRMS files are read through pyPRMS, as the translation layer does.

# %%
import pyPRMS

from pws_phoenix.prms_translate import load_parameters

params = load_parameters(DOMAIN_DIR / "myparam.param")
POI_SEGMENT = 184
wh_poi = np.where(params["poi_gage_segment"].values == POI_SEGMENT)[0]
gage_id = str(params["poi_gage_id"].values[wh_poi][0])
gage_seg = int(nhm_seg[POI_SEGMENT - 1])
print(f"gage {gage_id} is on segment index {POI_SEGMENT} = nhm_seg {gage_seg}")

obs_all = pyPRMS.DataFile(
    str(DOMAIN_DIR / "sf_data"),
    metadata=pyPRMS.MetaData(verbose=False).metadata,
).data_by_variable("runoff")
obs = obs_all[f"runoff_{gage_id}"]
obs = obs.loc[str(nhm_out["time"].values[0]) : str(nhm_out["time"].values[-1])]
obs_da = xr.DataArray(
    obs.values, dims=("time",), coords={"time": obs.index.values}
)
modeled_gage = nhm_out["seg_outflow"].isel(nsegment=POI_SEGMENT - 1)
plot_series(
    {"modeled": modeled_gage, f"gage {gage_id}": obs_da},
    "streamflow (cfs)",
    f"NHM vs observed at nhm_seg {gage_seg}",
    "nhm_vs_gage.png",
)

# %% [markdown]
# ## 2. The NHM with a STARFIT reservoir, as one model
#
# Now we insert the reservoir. `insert_flow_graph` takes the assembled
# kit and a list of new nodes, and replaces the kit's `PRMSChannel`
# with a `FlowGraph`: the existing segments become channel nodes, each
# new node is spliced in above the segment it flows to, and the maps
# that carry runoff from the HRUs to the segments are extended (the
# new node receives no lateral inflow of its own). Every other process
# in the kit is untouched.
#
# A new node needs its type, where it flows to, an id, and its
# parameters. The reservoir's parameters are the STARFIT row we looked
# at, with two values pywatershed fills in for you at run time but
# that we supply up front here (parameters are frozen once the model
# is assembled):
#
# - `Obs_MEANFLOW_CUMECS`, the long-term mean inflow, defaults to the
#   `inflow_mean` in the file when missing;
# - `initial_storage` defaults to the midpoint of the normal operating
#   range on the day before the simulation starts
#   (`nor_midpoint_initial_storage`).
#
# The channel network routes in cubic feet per second, so storages are
# supplied in millions of cubic feet (`cms_to_cfs` converts; it equals
# the cubic-meters-to-cubic-feet factor).

# %%
from pws_phoenix.flow_graph_build import NewNode
from pws_phoenix.hydrology.starfit_daily_flow_node import StarfitDailyFlowNode
from pws_phoenix.hydrology.starfit_flow_node import (
    StarfitFlowNode,
    cms_to_cfs,
    nor_midpoint_initial_storage,
)
from pws_phoenix.prms_translate import insert_flow_graph

RESERVOIR_ID = 999
START_TIME = kit_nhm.config.start_time
# pywatershed's "initial time" is the day before the first time step
INIT_DATE = (START_TIME - np.timedelta64(1, "D")).astype("datetime64[s]")
INIT_DATE = INIT_DATE.astype(datetime.datetime)

PARAMETER_NAMES = [
    name
    for name, meta in StarfitFlowNode.fields.items()
    if meta.kind == "parameter"
]


def starfit_node_parameters(sf_ds, cap_mult=1.0):
    """The reservoir's STARFIT parameters as a NewNode parameter dict,
    capacity scaled by cap_mult, the two run-time defaults filled."""
    row = {nn: float(sf_ds[nn]) for nn in PARAMETER_NAMES if nn in sf_ds}
    row["GRanD_CAP_MCM"] *= cap_mult
    if np.isnan(row.get("Obs_MEANFLOW_CUMECS", np.nan)):
        row["Obs_MEANFLOW_CUMECS"] = float(sf_ds["inflow_mean"])
    if np.isnan(row.get("initial_storage", np.nan)):
        row["initial_storage"] = (
            nor_midpoint_initial_storage(row, INIT_DATE) * cms_to_cfs
        )
    missing = set(PARAMETER_NAMES) - set(row)
    assert not missing, f"no values for {sorted(missing)}"
    return row


big_sandy_params = starfit_node_parameters(big_sandy)
print(
    f"initial storage: {big_sandy_params['initial_storage']:.1f} million "
    f"cubic feet on {INIT_DATE:%Y-%m-%d} "
    f"(capacity {big_sandy['GRanD_CAP_MCM'].item() * cms_to_cfs:.1f})"
)

# %% [markdown]
# pywatershed's notebook runs this reservoir with `compute_daily=True`:
# the reservoir decides each day's release once, from the previous
# day's inflow, and holds it constant through the day, while the
# channel routes on hourly substeps around it. That is
# `StarfitDailyFlowNode` here (`StarfitFlowNode` is the hourly form,
# which we'll use in section 3).

# %%
starfit_store = nb_dir / "fgr_starfit.zarr"
kit_starfit = insert_flow_graph(
    assemble_from_control(
        CONTROL_FILE,
        cbh_paths=CBH_PATHS,
        control={
            "budget": "error",
            "output_serial_zarr": starfit_store,
            "output_var_names": [
                "node_outflows",
                "node_upstream_inflows",
                "node_storages",
            ],
            "time_chunk_size": 365,
        },
    ),
    [
        NewNode(
            StarfitDailyFlowNode,
            flows_to=NHM_SEG_BELOW_RESERVOIR,
            node_id=RESERVOIR_ID,
            parameters=big_sandy_params,
        )
    ],
)
print("processes:", list(kit_starfit.process_dict))
print(f"grids: {sorted(kit_starfit.discretizations)}")

# %% [markdown]
# The channel slot became `flow_graph`, on a new `nnodes` grid: the
# 298 segments plus our one reservoir. The grid's coordinate is the
# NHM segment id for channel nodes and the node id we chose for new
# ones, so we can select output by either. Let's run it.

# %%
if starfit_store.exists():
    print(f"{starfit_store.name} exists; not rerunning")
else:
    with kit_starfit.model() as model:
        model.run(S_PER_DAY, np.int32(model.ntime))
    print("done")

starfit_out = xr.open_zarr(starfit_store, consolidated=False).load()
outflow_starfit = (
    starfit_out["node_outflows"]
    .sel(nnodes=NHM_SEG_BELOW_RESERVOIR)
    .rename("STARFIT")
)
plot_series(
    {"NHM": outflow_nhm, "with STARFIT": outflow_starfit},
    "streamflow (cfs)",
    f"outflow at segment {NHM_SEG_BELOW_RESERVOIR}",
    "starfit_outflow_44426.png",
)

# %% [markdown]
# The reservoir flattens the spring peak and sustains flow through
# the rest of the year. Its storage tells the other half of the story
# (`node_storages` is defined on reservoir nodes only; channel nodes
# hold NaN there).

# %%
storage_starfit = (
    starfit_out["node_storages"].sel(nnodes=RESERVOIR_ID).rename("storage")
)
plot_series(
    {"Big Sandy storage": storage_starfit},
    "storage (million cubic feet)",
    "Big Sandy storage",
    "starfit_storage.png",
)

# %% [markdown]
# ## 3. The FlowGraph as a post-process
#
# Nothing in the NHM feeds back from the channel to the land surface,
# so the `FlowGraph` can run on its own, driven by the lateral inflows
# a previous run wrote out. That is how you'd explore reservoir
# designs: run the land surface once, then rerun only the routing.
#
# For illustration we'll also change the design: capacity scaled by
# 1.5, the hourly reservoir form, and -- to show that a graph can
# take any number of new nodes -- a pass-through node (no effect on
# flow by construction) above segment 44435.
#
# `splice_flow_graph` is the lower-level piece `insert_flow_graph`
# used above: given the segment network and the channel parameters it
# returns the composed graph class, its grid, its parameters, and a
# helper to extend the HRU-to-segment weights.

# %%
from pws_phoenix.flow_graph_build import splice_flow_graph
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode

CAP_MULT = 1.5
PASS_THROUGH_ID = 9999
NHM_SEG_BELOW_PASS_THROUGH = 44435

graph = splice_flow_graph(
    seg_dis,
    kit_nhm.process_dict["prms_channel"]["parameters"],
    [
        NewNode(
            StarfitFlowNode,
            flows_to=NHM_SEG_BELOW_RESERVOIR,
            node_id=RESERVOIR_ID,
            parameters=starfit_node_parameters(big_sandy, cap_mult=CAP_MULT),
        ),
        NewNode(
            PassThroughFlowNode,
            flows_to=NHM_SEG_BELOW_PASS_THROUGH,
            node_id=PASS_THROUGH_ID,
        ),
    ],
)
print(f"{graph.n_nodes} nodes = {graph.n_seg} segments + 2 new")

# %% [markdown]
# The graph's inputs are the three lateral inflow volumes PER NODE.
# In section 2 the model's maps aggregated them from the HRUs every
# day; here we do that aggregation once, up front, with the same
# weights (extended with zero rows for the two new nodes), on the
# volumes our baseline run wrote out.

# %%
node_inputs = {}
for name, mm in kit_nhm.maps.items():
    if mm.target_grid != "nsegment":
        continue
    weights = graph.pad_weights(mm.weights)  # (nnodes, nhru)
    hru_vol = nhm_out[mm.source_var]  # (time, nhru)
    node_inputs[f"node_{mm.source_var}"] = xr.DataArray(
        hru_vol.values @ weights.T,
        dims=("time", "nnodes"),
        coords={"time": hru_vol["time"].values},
    )
print("graph inputs:", sorted(node_inputs))

# %% [markdown]
# A one-process model: the graph, its grid, its parameters, and the
# inputs. We also ask for the reservoir's release and spill, which
# together make its outflow.

# %%
from pws_phoenix.model import Model

post_store = nb_dir / "fgr_starfit_post.zarr"
if post_store.exists():
    print(f"{post_store.name} exists; not rerunning")
else:
    model_post = Model(
        {
            "flow_graph": {
                "class": graph.graph_class,
                "discretization": "nnodes",
                "parameters": graph.parameters,
                **node_inputs,
            }
        },
        {
            "budget": "error",
            "output_serial_zarr": post_store,
            "output_var_names": [
                "node_outflows",
                "node_upstream_inflows",
                "node_storages",
                "lake_release",
                "lake_spill",
            ],
            "time_chunk_size": 365,
        },
        discretizations={"nnodes": graph.discretization},
    )
    with model_post:
        model_post.run(S_PER_DAY, np.int32(model_post.ntime))
    print("done")

post_out = xr.open_zarr(post_store, consolidated=False).load()
outflow_post = post_out["node_outflows"].sel(nnodes=NHM_SEG_BELOW_RESERVOIR)
storage_post = post_out["node_storages"].sel(nnodes=RESERVOIR_ID)

# %% [markdown]
# Three designs side by side at segment 44426, and the two reservoir
# storages:

# %%
plot_series(
    {
        "NHM": outflow_nhm,
        "STARFIT": outflow_starfit,
        f"STARFIT, capacity x{CAP_MULT}": outflow_post,
    },
    "streamflow (cfs)",
    f"outflow at segment {NHM_SEG_BELOW_RESERVOIR}",
    "post_outflow_44426.png",
)
plot_series(
    {
        "STARFIT": storage_starfit,
        f"STARFIT, capacity x{CAP_MULT}": storage_post,
    },
    "storage (million cubic feet)",
    "Big Sandy storage",
    "post_storage.png",
)

# %% [markdown]
# A reservoir's outflow is its release (by the operating rule) plus
# any spill (storage over capacity). From the outflow alone you can't
# tell them apart, which is why we asked for both: in the second
# summer there is a spill event that adds to the release.

# %%
plot_series(
    {
        "outflow": post_out["node_outflows"].sel(nnodes=RESERVOIR_ID),
        "release": post_out["lake_release"].sel(nnodes=RESERVOIR_ID),
        "spill": post_out["lake_spill"].sel(nnodes=RESERVOIR_ID),
    },
    "streamflow (cfs)",
    f"Big Sandy, capacity x{CAP_MULT}",
    "post_release_spill.png",
)

# %% [markdown]
# ## Checking against pywatershed
#
# pywatershed's notebook wrote its three runs' outputs beside itself;
# if they're present we compare ours with them. The two models read
# the same control file, the same parameters (pywatershed's
# `parameters_*.nc` hold the same float64 values we parse from
# `myparam.param`) and the same forcings, so any difference is
# arithmetic: pywatershed compiles its kernels with fast-math, which
# departs from strict IEEE arithmetic by about 1e-8 -- enough to flip
# a few snowpack decisions per season on a few HRUs, which routing
# then spreads downstream. The flows therefore agree to within 1 % on
# nearly every segment-day, with a small set of low-flow days where
# such a flip happened; the reservoir storage, which integrates the
# inflow, agrees to about 1e-6. (`docs/PORTS.md`, "The snow story",
# has the measurements behind this.)


# %%
def compare_with_pywatershed(label, ours, theirs, tol):
    """Fraction of values on the common days with |ours - theirs| >
    tol + tol |theirs| (NaN in both counts as agreement)."""
    common = np.intersect1d(
        ours["time"].values.astype("datetime64[ns]"),
        theirs["time"].values.astype("datetime64[ns]"),
    )
    oo = ours.sel(time=common).values
    tt = theirs.sel(time=common).values
    assert oo.shape == tt.shape, (oo.shape, tt.shape)
    both_nan = np.isnan(oo) & np.isnan(tt)
    fracs = {}
    for tt_ in tol:
        outside = (np.abs(oo - tt) > tt_ + tt_ * np.abs(tt)) & ~both_nan
        fracs[tt_] = outside.mean()
    print(
        f"{label}: {len(common)} days x {oo.shape[1]} columns; outside "
        + ", ".join(f"{tt_:.0e}: {ff:.1e}" for tt_, ff in fracs.items())
    )
    return fracs


# the outflows: nearly all within 1 %, and at most a few parts per
# thousand beyond 10 % (the knife-edge days)
FLOW_CHECK = ((1e-2, 1e-1), 1e-1, 5e-3)
# the reservoir storage: within 1e-5 everywhere
STORAGE_CHECK = ((1e-6, 1e-5), 1e-5, 0.0)
targets = {
    "fgr_nhm": {"seg_outflow": FLOW_CHECK},
    "fgr_starfit": {
        "node_outflows": FLOW_CHECK,
        "node_storages": STORAGE_CHECK,
    },
    "fgr_starfit_post": {
        "node_outflows": FLOW_CHECK,
        "node_storages": STORAGE_CHECK,
        "release": FLOW_CHECK,
        "spill": FLOW_CHECK,
    },
}
ours = {
    "fgr_nhm": nhm_out,
    "fgr_starfit": starfit_out,
    "fgr_starfit_post": post_out.rename(
        {"lake_release": "release", "lake_spill": "spill"}
    ),
}
if not PYWS_NB_OUTPUT_DIR.exists():
    print(f"({PYWS_NB_OUTPUT_DIR} not found; verification skipped)")
else:
    for run, checks in targets.items():
        for name, (tols, tol_assert, max_frac) in checks.items():
            fracs = compare_with_pywatershed(
                f"{run}/{name}",
                ours[run][name],
                xr.load_dataarray(PYWS_NB_OUTPUT_DIR / run / f"{name}.nc"),
                tols,
            )
            assert fracs[tol_assert] <= max_frac, (run, name, fracs)

# %% [markdown]
# ## What just happened, and where next
#
# - `insert_flow_graph(kit, [NewNode(...)])` turned a PRMS channel
#   network into a `FlowGraph` with a reservoir in it, inside an
#   otherwise unchanged NHM.
# - `splice_flow_graph` built the same graph on its own, which we
#   drove from a previous run's lateral inflows -- the pattern for
#   trying reservoir designs cheaply. The inputs can equally come from
#   pywatershed's own output files (they are `(time, nhm_id)` NetCDF;
#   rename the dimension to `nhru`).
# - Node types available: channel, pass-through, STARFIT (hourly and
#   daily), observed inflow, source/sink, and STARFIT with
#   source/sink -- see `pws_phoenix/hydrology/*_flow_node.py`.
#
# ## References
#
# Turner, S. W. D., Steyaert, J. C., Condon, L., and Voisin, N.
# (2021). Water storage and release policies for all large reservoirs
# of conterminous United States. Journal of Hydrology, 603, 126843.
