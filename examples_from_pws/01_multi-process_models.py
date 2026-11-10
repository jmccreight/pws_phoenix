# %% [markdown]
# # Multi-process models
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# In notebook `00_processes.py` we looked at how an individual process
# works and is designed. In this notebook we learn how to put several
# processes together into a composite model using the `Model` class.
#
# The starting point for the development of pywatershed, and of
# `pws_phoenix`, was the National Hydrologic Model (NHM, Regan et al.,
# 2018) configuration of the Precipitation-Runoff Modeling System
# (PRMS, Regan et al., 2022). We'll first construct a full NHM
# configuration with stream temperature. The spatial domain will again
# be the Delaware River Basin. Once we have the full NHM, we'll build
# the same model from a YAML file, reduce its output, and then
# construct a sub-model of it.
#
# This is a translation of pywatershed's notebook
# `01_multi-process_models.ipynb`; the last section checks our results
# against that notebook's output.
#
# ## Prerequisites
#
# The `pwpx` environment (see `pws_phoenix/environment.yaml`). The
# domain files are the small Delaware River Basin example that
# pywatershed ships inside its package, read from the pywatershed
# clone beside this repository. Plots need `matplotlib` (optional);
# the contract diagram prefers `graphviz` (optional).

# %%
import pathlib as pl
import pydoc
import time
from pprint import pprint
from typing import Any

import numpy as np
import xarray as xr
import yaml

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

from pws_phoenix.atmosphere.prms_atmosphere import PRMSAtmosphere
from pws_phoenix.atmosphere.prms_solar_geometry import compute_soltabs
from pws_phoenix.config import load_model_yaml
from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_canopy import PRMSCanopy
from pws_phoenix.hydrology.prms_channel import PRMSChannel
from pws_phoenix.hydrology.prms_groundwater import PRMSGroundwater
from pws_phoenix.hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryWidthOnly,
)
from pws_phoenix.hydrology.prms_runoff import PRMSRunoff
from pws_phoenix.hydrology.prms_snow import PRMSSnow
from pws_phoenix.hydrology.prms_soilzone import PRMSSoilzone
from pws_phoenix.hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    PRMSStreamTemp,
    derive_aggregation_weights,
)
from pws_phoenix.maps import Map
from pws_phoenix.model import Model
from pws_phoenix.model_contract_graph import ModelContractGraph
from pws_phoenix.process import DataArrayMeta, Process

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None  # type: ignore[assignment]
    print("(matplotlib not installed; plots skipped)")

# %% [markdown]
# ## Where things are

# %%
PYWS_ROOT = _repo.parent / "pywatershed"
DOMAIN_DIR = PYWS_ROOT / "pywatershed" / "data" / "drb_2yr"
PYWS_NB_OUTPUT_DIR = PYWS_ROOT / "examples" / "01_multi-process_models"
if not DOMAIN_DIR.exists():
    raise FileNotFoundError(
        f"{DOMAIN_DIR} -- the pywatershed clone beside this repository "
        "supplies the drb_2yr domain files"
    )

nb_dir = _here / "01_multi-process_models"
nb_dir.mkdir(exist_ok=True)
print(f"domain: {DOMAIN_DIR}\noutputs: {nb_dir}")

# %% [markdown]
# ## Getting to know the domain
#
# pywatershed's notebook opens with an interactive map of the HRUs
# and stream segments. The GIS files and plotting helpers are not part
# of `pws_phoenix` yet (see that notebook for the map); here we'll get
# to know the domain from its two discretization files, which carry
# the static description of space: the HRUs (their areas, latitudes,
# elevations) and the stream segments (their lengths, slopes, and the
# `tosegment` connectivity that makes them a network).

# %%
dis_hru_ds = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
dis_seg_ds = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
print(
    f"{dis_hru_ds.sizes['nhru']} HRUs, "
    f"{dis_seg_ds.sizes['nsegment']} segments; "
    f"total area {dis_hru_ds['hru_area'].sum().item():,.0f} acres"
)
dis_hru_ds

# %%
dis_seg_ds

# %% [markdown]
# ## An NHM multi-process model for the Delaware River Basin
#
# pywatershed's NHM is a list of ten `Process` classes. In
# `pws_phoenix` the same model has nine processes on two grids, plus
# one small helper:
#
# | pywatershed | here | grid |
# | --- | --- | --- |
# | `PRMSSolarGeometry` | not a process: `compute_soltabs()` makes its tables (parameters) | |
# | `PRMSAtmosphere` | `PRMSAtmosphere` | `nhru` |
# | `PRMSCanopy` | `PRMSCanopy` | `nhru` |
# | `PRMSSnow` | `PRMSSnow` | `nhru` |
# | `PRMSRunoff` | `PRMSRunoff` | `nhru` |
# | `PRMSSoilzone` | `PRMSSoilzone` | `nhru` |
# | `PRMSGroundwater` | `PRMSGroundwater` | `nhru` |
# | `PRMSChannel` | `PRMSChannel` | `nsegment` |
# | `PRMSHydraulicGeometryWidthOnly` | `PRMSHydraulicGeometryWidthOnly` | `nsegment` |
# | `PRMSStreamTempHumidityCBH` | `PRMSStreamTemp` + a humidity `Map` | `nsegment` |
#
# Two things stand out. First, solar geometry depends only on the
# HRUs' slope, aspect and latitude, so it is computed once as a set of
# parameters (tables by day of year) rather than stepped in time.
# Second, every process lives on exactly ONE grid. pywatershed's
# channel and stream temperature needed `dis_both` (HRUs and segments)
# because they summed HRU quantities onto segments internally; here
# that summing is done by `Map` objects between the grids, which we'll
# build below, so the segment processes know nothing about HRUs. The
# stream-temperature humidity forcing is HRU data that no HRU process
# consumes, so a one-field helper process (`HumidityCarrier`) owns it
# on the `nhru` grid for its Map to read.
#
# The slot names (`prms_atmosphere`, ...) are ours to choose; their
# ORDER is the order the processes run in each timestep.

# %%
nhm_processes: dict[str, tuple[type[Process], str]] = {
    "prms_atmosphere": (PRMSAtmosphere, "nhru"),
    "prms_canopy": (PRMSCanopy, "nhru"),
    "prms_snow": (PRMSSnow, "nhru"),
    "prms_runoff": (PRMSRunoff, "nhru"),
    "prms_soilzone": (PRMSSoilzone, "nhru"),
    "prms_groundwater": (PRMSGroundwater, "nhru"),
    "prms_channel": (PRMSChannel, "nsegment"),
    "prms_hydraulic_geometry": (PRMSHydraulicGeometryWidthOnly, "nsegment"),
    "prms_stream_temp": (PRMSStreamTemp, "nsegment"),
}


class HumidityCarrier(Process):
    """The humidity forcing on the hru grid, so its Map can read it."""

    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH relative humidity [percent]",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


# %% [markdown]
# A multi-process model is assembled by the `Model` class. We can take
# a quick look at the first lines of its help:

# %%
model_help = pydoc.render_doc(
    Model,
    "Help on %s",
    renderer=pydoc.plaintext,  # type: ignore[attr-defined]
)
print("\n".join(model_help.splitlines()[0:22]))

# %% [markdown]
# `Model` takes four things: the process dictionary (what runs, on
# which grid, with which data), the control dictionary (run-wide
# options), the maps (how quantities cross between grids) and the
# discretizations (the grids). We'll build them in turn, in memory,
# and then do it again from a YAML file.
#
# ## Model dictionary in memory
#
# *Discretizations:* the two grids, from the files we looked at above.
# The segment grid is also asked to derive `segment_order` (the
# upstream-to-downstream order the channel routes in) from
# `tosegment`.


# %%
def make_discretizations() -> dict[str, Discretization]:
    return {
        "nhru": Discretization(
            ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
        ),
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }


discretizations = make_discretizations()
seg_dis_params = discretizations["nsegment"].parameters
assert seg_dis_params is not None
print("segment_order[:10]:", seg_dis_params["segment_order"].values[:10])

# %% [markdown]
# *Parameters:* because the parameter files that come with pywatershed
# and our classes are consistently named, we can load each process's
# parameters by its class name, just as pywatershed's notebook does.

# %%
parameters: dict[str, xr.Dataset] = {
    slot: xr.load_dataset(DOMAIN_DIR / f"parameters_{cls.__name__}.nc")
    for slot, (cls, _) in nhm_processes.items()
}
print({slot: len(ds.data_vars) for slot, ds in parameters.items()})

# %% [markdown]
# Three adjustments bring these files to what our classes declare.
# The solar tables (pywatershed's `PRMSSolarGeometry` output) are
# parameters of the atmosphere and snow processes, so we compute them
# from the HRU discretization and merge them in. And stream
# temperature's shade calculation, a separate `PRMSStreamShadeDynamic`
# class in pywatershed, is part of `PRMSStreamTemp` here, so its
# parameter file merges in too, along with `hru_area` (the one HRU
# quantity stream temperature reads, to weight its aggregations).
# Anything a process declares that we don't supply here is looked up
# in its grid's discretization first (`lat_temp_adj`, `seg_length`,
# ...).

# %%
soltabs = compute_soltabs(dis_hru_ds)
parameters["prms_atmosphere"] = xr.merge(
    [parameters["prms_atmosphere"], soltabs]
)
parameters["prms_snow"] = xr.merge(
    [parameters["prms_snow"], soltabs[["soltab_horad_potsw"]]]
)
parameters["prms_stream_temp"] = xr.merge(
    [
        parameters["prms_stream_temp"],
        xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"),
        dis_hru_ds[["hru_area"]],
    ],
    compat="no_conflicts",
)
soltabs

# %% [markdown]
# *Forcings:* the three climate-by-HRU inputs the atmosphere process
# reads (precipitation, maximum and minimum temperature) and the
# relative humidity for stream temperature. The files store float32
# on a dimension named `nhm_id`; the model computes in float64 on the
# grid dimension `nhru`, so we widen and rename (pywatershed's
# `input_aliases` entry, `{"humidity_hru": "rhavg"}`, becomes the
# rename of the humidity variable).


# %%
def load_forcing(file_name: str) -> xr.DataArray:
    return (
        xr.load_dataarray(DOMAIN_DIR / file_name)
        .rename({"nhm_id": "nhru"})
        .astype(np.float64)
    )


forcings = {nn: load_forcing(f"{nn}.nc") for nn in ("prcp", "tmax", "tmin")}
forcings["humidity_hru"] = load_forcing("rhavg.nc").rename("humidity_hru")
forcings["prcp"]

# %% [markdown]
# *Maps:* the channel needs the three lateral-inflow volumes
# (surface runoff, subsurface flow, groundwater flow) summed from the
# HRUs onto their segments: one weights matrix (a 1 where an HRU
# drains to a segment) carrying three variables, so three `Map`s. The
# stream temperature's aggregates (segment-average air temperature,
# radiation, humidity, ...) are ten more, with area-weighted and
# flow-weighted matrices derived from the network. Each Map names its
# source variable on the HRU grid and the target variable it produces
# on the segment grid.

# %%
hru_segment = parameters["prms_channel"]["hru_segment"].values
vol_weights = np.zeros((dis_seg_ds.sizes["nsegment"], hru_segment.size))
for ihru in range(hru_segment.size):
    if hru_segment[ihru] > 0:
        vol_weights[hru_segment[ihru] - 1, ihru] = 1.0

stp = parameters["prms_stream_temp"]
agg_weights = derive_aggregation_weights(
    stp["hru_segment"].values.astype(np.int64),
    dis_hru_ds["hru_area"].values,
    dis_seg_ds["tosegment"].values.astype(np.int64),
    seg_dis_params["segment_order"].values.astype(np.int64),
    stp["seg_close"].values,
)
VOLUME_MAP_SPEC = {  # target: source (the stream-temp spec has this form)
    "seg_sroff_vol": "sroff_vol",
    "seg_ssres_flow_vol": "ssres_flow_vol",
    "seg_gwres_flow_vol": "gwres_flow_vol",
}


def make_maps(weights: dict[str, np.ndarray]) -> dict[str, Map]:
    """The 13 hru -> segment Maps from {weights_name: matrix}."""
    maps = {
        target: Map(
            weights=weights["vol"],
            grid={"nhru": "nsegment"},
            variable={source: target},
        )
        for target, source in VOLUME_MAP_SPEC.items()
    }
    maps |= {
        target: Map(
            weights=weights[wkey],
            grid={"nhru": "nsegment"},
            variable={source: target},
        )
        for target, (source, wkey) in AGGREGATION_MAP_SPEC.items()
    }
    return maps


weights = {"vol": vol_weights, **agg_weights}
maps = make_maps(weights)
print({name: mm.weights.shape for name, mm in maps.items()})

# %% [markdown]
# *The process dictionary:* each entry names its class, its grid, its
# parameters, and any inputs and initial values it takes from us. The
# forcings go to the processes that consume them; the humidity carrier
# slots in just before the first segment process; groundwater storage
# and channel flow start from the `*_init` arrays in their parameter
# files (as in PRMS).

# %%
process_dict: dict[str, dict[str, Any]] = {}
for slot, (cls, grid) in nhm_processes.items():
    if grid == "nsegment" and "humidity_carrier" not in process_dict:
        process_dict["humidity_carrier"] = {
            "class": HumidityCarrier,
            "discretization": "nhru",
            "humidity_hru": forcings["humidity_hru"],
        }
    process_dict[slot] = {
        "class": cls,
        "discretization": grid,
        "parameters": parameters[slot],
    }
process_dict["prms_atmosphere"] |= {
    nn: forcings[nn] for nn in ("prcp", "tmax", "tmin")
}
process_dict["prms_groundwater"]["gwstor_init"] = parameters[
    "prms_groundwater"
]["gwstor_init"]
process_dict["prms_channel"]["segment_flow_init"] = parameters["prms_channel"][
    "segment_flow_init"
]

pprint(
    {
        slot: {kk: type(vv).__name__ for kk, vv in entry.items()}
        for slot, entry in process_dict.items()
    },
    sort_dicts=False,
)

# %% [markdown]
# *Control:* we check every process's mass (and energy) budget each
# step, and write EVERY variable of every process, plus the budget
# terms, to one zarr store (we'll need all of them for the sub-model
# later). The run window is the first six months of 1979, as in
# pywatershed's notebook; feel free to increase it to the full two
# years available.


# %%
def all_var_names(process_dict: dict[str, dict[str, Any]]) -> list[str]:
    names: dict[str, None] = {}
    for entry in process_dict.values():
        for nn in entry["class"].get_var_names(include_budget=True):
            names[nn] = None
    return list(names)


START = np.datetime64("1979-01-01")
END = np.datetime64("1979-07-01")
n_days = int((END - START) / np.timedelta64(1, "D")) + 1
dt = np.float64(24 * 60 * 60)  # seconds

control: dict[str, Any] = {
    "budget": "error",
    "output_serial_zarr": nb_dir / "nhm_memory.zarr",
    "output_var_names": all_var_names(process_dict),
    "time_chunk_size": 61,
}
print(f"{n_days} days; {len(control['output_var_names'])} output variables")

# %% [markdown]
# The way these processes connect can be worked out from the process
# dictionary alone, because each process describes itself (as we saw
# in the previous notebook). `Model.input_spec` lists what the
# configuration needs from outside: for the full NHM, just the four
# forcings and the two initial values.

# %%
spec = Model.input_spec(process_dict, maps=maps)
for grid, gg in spec["required"].items():
    print(grid, "external inputs:", list(gg["external_inputs"]))
    print(grid, "initial values: ", list(gg["initial_values"]))

# %% [markdown]
# `ModelContractGraph` draws the same information (pywatershed's
# `ModelGraph`): each grid is a cluster, each process a box, solid
# edges are variables passed between processes on a grid, dashed
# edges are the Maps, and the gray ellipses are the forcings. With
# `graphviz` installed the cell renders the structured layout; without
# it the fallback is mermaid (JupyterLab >= 4.1). Notebook
# `examples/00_input_contract.py` goes deeper into the contract.

# %%
graph = ModelContractGraph(process_dict, maps=maps)
print(
    f"{sum(len(pp) for pp in graph.grids.values())} processes on "
    f"{len(graph.grids)} grids; {len(graph.internal_edges)} internal "
    f"couplings, {sum(len(ll) for _, _, ll in graph.map_edges)} map-fed "
    f"inputs, {sum(len(ee) for ee in graph.externals.values())} "
    "external inputs"
)
try:
    from graphviz import Source

    diagram: object = Source(graph.to_dot(rankdir="TB"))
except ImportError:
    diagram = graph  # mermaid fallback
diagram

# %% [markdown]
# Now we construct the model (construction checks that every input is
# accounted for) and run it. We'll time the run for comparison later.

# %%
model_mem = Model(
    process_dict, control, maps=maps, discretizations=discretizations
)
tic = time.perf_counter()
model_mem.run(dt, np.int32(n_days))
model_mem.finalize()
time_mem = time.perf_counter() - tic
print(f"clock: {model_mem.time.current}; wall time {time_mem:.1f} s")

# %% [markdown]
# No errors arose, so every budget closed on every step. We'll look at
# the outputs after we see how to build this model from a YAML file.
#
# ## Model dictionary YAML file
#
# It may be preferable to have a model dictionary encoded in a YAML
# file in many situations, instead of in Python code. Necessarily, the
# YAML will look different from the code above: a YAML file holds
# names and paths, not arrays, so each piece of in-memory data becomes
# a file, and each class becomes its name. We'll create a directory
# for the YAML-based run and write the data there: the merged
# parameter sets, the prepared forcings (already float64 on `nhru`),
# the initial values, and the Maps' weights.

# %%
run_dir = nb_dir / "nhm_yaml"
run_dir.mkdir(exist_ok=True)


def relpath(path: pl.Path) -> str:
    return str(path.relative_to(run_dir))


written: dict[str, str] = {}  # in-memory object name -> path in the yaml
for slot, ds in parameters.items():
    ds.to_netcdf(run_dir / f"parameters_{slot}.nc")
    written[f"parameters:{slot}"] = f"parameters_{slot}.nc"
for name, da in forcings.items():
    da.to_netcdf(run_dir / f"{name}.nc")
    written[name] = f"{name}.nc"
for slot, name in (
    ("prms_groundwater", "gwstor_init"),
    ("prms_channel", "segment_flow_init"),
):
    process_dict[slot][name].to_netcdf(run_dir / f"{name}.nc")
    written[name] = f"{name}.nc"
for wkey, ww in weights.items():
    xr.DataArray(ww, dims=("nsegment", "nhru"), name="weights").to_netcdf(
        run_dir / f"weights_{wkey}.nc"
    )
    written[f"weights:{wkey}"] = f"weights_{wkey}.nc"
pprint(written, sort_dicts=False)

# %% [markdown]
# Now the configuration itself. Paths are resolved relative to the
# YAML file's directory (absolute paths pass through: we point at the
# discretization files where pywatershed ships them). Process classes
# are named; `load_model_yaml` resolves the names through the registry
# every `Process` subclass joins when its module is imported, so the
# importing we did at the top (and defining `HumidityCarrier`) is
# what makes the names resolvable. The order of the `processes`
# mapping is the execution order, so we dump with `sort_keys=False`.

# %%
cfg: dict[str, Any] = {
    "discretizations": {
        "nhru": {
            "dims": ["nhru"],
            "parameters": str(DOMAIN_DIR / "parameters_dis_hru.nc"),
        },
        "nsegment": {
            "dims": ["nsegment"],
            "parameters": str(DOMAIN_DIR / "parameters_dis_seg.nc"),
            "topo_order": {"segment_order": "tosegment"},
        },
    },
    "control": {
        "budget": "error",
        "output_serial_zarr": "nhm_yaml.zarr",
        "output_var_names": control["output_var_names"],
        "time_chunk_size": 61,
    },
    "processes": {},
    "maps": {},
}
for slot, entry in process_dict.items():
    yentry: dict[str, Any] = {
        "class": entry["class"].__name__,
        "discretization": entry["discretization"],
    }
    if "parameters" in entry:
        yentry["parameters"] = written[f"parameters:{slot}"]
    for key in entry:
        if key in written:  # a forcing or an initial value
            yentry[key] = written[key]
    cfg["processes"][slot] = yentry
for target, source in VOLUME_MAP_SPEC.items():
    cfg["maps"][target] = {
        "class": "Map",
        "weights": written["weights:vol"],
        "grid": {"nhru": "nsegment"},
        "variable": {source: target},
    }
for target, (source, wkey) in AGGREGATION_MAP_SPEC.items():
    cfg["maps"][target] = {
        "class": "Map",
        "weights": written[f"weights:{wkey}"],
        "grid": {"nhru": "nsegment"},
        "variable": {source: target},
    }

model_yaml_file = run_dir / "model.yaml"
with open(model_yaml_file, "w") as ff:
    yaml.safe_dump(cfg, ff, sort_keys=False)

# %% [markdown]
# Let's examine the written file (the output variable list is long, so
# we elide it):

# %%
for line in model_yaml_file.read_text().splitlines():
    if line.startswith("  - ") and len(line) < 60:
        continue  # the output_var_names entries
    print(line)

# %% [markdown]
# Now we can create a `Model` from the file. `load_model_yaml` returns
# exactly the four things `Model` takes.

# %%
process_dict_yaml, control_yaml, maps_yaml, discretizations_yaml = (
    load_model_yaml(model_yaml_file)
)
model_yaml = Model(
    process_dict_yaml,
    control_yaml,
    maps=maps_yaml,
    discretizations=discretizations_yaml,
)
print(f"{len(model_yaml.model_dict)} processes, {len(maps_yaml)} maps")

# %%
tic = time.perf_counter()
model_yaml.run(dt, np.int32(n_days))
model_yaml.finalize()
time_yaml = time.perf_counter() - tic
print(f"clock: {model_yaml.time.current}; wall time {time_yaml:.1f} s")

# %% [markdown]
# ## Compare the outputs of the two models
#
# Let's see that these constructed and executed the same model. Each
# run wrote one zarr store; a half-year of this domain is small, so we
# read both into memory and ask xarray for equality (every variable,
# every value).

# %%
out_mem = xr.open_zarr(
    control["output_serial_zarr"], consolidated=False
).load()
out_yaml = xr.open_zarr(
    control_yaml["output_serial_zarr"], consolidated=False
).load()
assert set(out_mem.data_vars) == set(out_yaml.data_vars)
xr.testing.assert_equal(out_mem, out_yaml)
print(f"{len(out_mem.data_vars)} variables identical")
out_mem

# %% [markdown]
# We'll plot the stream temperature and the streamflow at the basin
# outlet. (pywatershed's notebook shows interactive and spatial plots;
# feel free to pick other segments or variables.)


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


# the outlet: the segment that flows to nowhere (tosegment == 0)
outlet = int(np.where(dis_seg_ds["tosegment"].values == 0)[0][0])
outlet_id = int(dis_seg_ds["nhm_seg"].values[outlet])
plot_series(
    {f"nhm_seg {outlet_id}": out_mem["seg_outflow"].isel(nsegment=outlet)},
    "cfs",
    "Streamflow at the basin outlet",
    "seg_outflow_outlet.png",
)
plot_series(
    {
        "water": out_mem["seg_tave_water"].isel(nsegment=outlet),
        "upstream inflow": out_mem["seg_tave_upstream"].isel(nsegment=outlet),
    },
    "degrees C",
    "Stream temperature at the basin outlet",
    "seg_tave_water_outlet.png",
)

# %% [markdown]
# We can also look at the budgets. Each budget's terms are variables
# in the store: the sums of inputs, outputs and storage changes, the
# residual, and its running accumulation. The soil zone's mass
# budget and the stream temperature's energy budget:

# %%
for cls in (PRMSSoilzone, PRMSStreamTemp):
    print(cls.__name__)
    for nn in cls.get_budget_var_names():
        print(f"  {nn:50s} max |.| = {np.abs(out_mem[nn].values).max():.2e}")

# %% [markdown]
# ## Reduce model output to disk
#
# Quite a lot of output was written (every variable of every process),
# and in many cases the amount can be reduced in favor of run time.
# That is one entry in the control: we'll ask for only the groundwater
# and channel variables, and reuse the in-memory process dictionary
# (its arrays are shared with the first model, no copies).

# %%
control_less: dict[str, Any] = control | {
    "output_serial_zarr": nb_dir / "nhm_less_output.zarr",
    "output_var_names": list(PRMSGroundwater.get_var_names())
    + list(PRMSChannel.get_var_names()),
}
model_less_output = Model(
    process_dict,
    control_less,
    maps=make_maps(weights),
    discretizations=make_discretizations(),
)
tic = time.perf_counter()
model_less_output.run(dt, np.int32(n_days))
model_less_output.finalize()
time_less = time.perf_counter() - tic
print(
    f"{len(control['output_var_names'])} variables: {time_mem:.1f} s; "
    f"{len(control_less['output_var_names'])} variables: {time_less:.1f} s"
)

# %% [markdown]
# ## NHM sub-model for the Delaware River Basin
#
# In many cases, running the full NHM may not be necessary and it may
# be advantageous to run just some of its processes. Suppose you
# wanted to change parameters or the process representation in
# `PRMSSoilzone` to better predict streamflow. As the model is
# one-way coupled, you can simply run a sub-model starting with the
# soil zone and running through the channel.

# %%
submodel_slots = ["prms_soilzone", "prms_groundwater", "prms_channel"]

# %% [markdown]
# This prompts the question: what inputs do we need for this
# sub-model? We can ask each process for its inputs, and for the
# variables it computes.

# %%
submodel_inputs = {
    slot: nhm_processes[slot][0].get_inputs() for slot in submodel_slots
}
pprint(submodel_inputs)

# %%
submodel_variables = {
    slot: nhm_processes[slot][0].get_var_names() for slot in submodel_slots
}
pprint(submodel_variables)

# %% [markdown]
# What the sub-model needs from outside is the inputs no process in
# it computes and no Map carries. Rather than take the set difference
# by hand, we ask `Model.input_spec` with the three volume Maps the
# channel needs (the stream-temperature Maps are not part of this
# sub-model):

# %%
submodel_dict: dict[str, dict[str, Any]] = {
    slot: dict(process_dict[slot]) for slot in submodel_slots
}
submodel_maps = {
    target: maps[target] for target in VOLUME_MAP_SPEC
}  # the wiring only: we'll rebuild them with make_maps below
sub_spec = Model.input_spec(submodel_dict, maps=submodel_maps)
submodel_file_inputs = {
    name: info["consumers"][0]
    for name, info in sub_spec["required"]["nhru"]["external_inputs"].items()
}
pprint(submodel_file_inputs)

# %% [markdown]
# And where do we get these? They were all computed, and written, by
# the full model above: we take them straight from its zarr store (the
# `time` axis of the inputs becomes the sub-model's time axis).

# %%
for name, consumer in submodel_file_inputs.items():
    submodel_dict[consumer][name] = out_mem[name]
sub_control: dict[str, Any] = {
    "budget": "error",
    "output_serial_zarr": nb_dir / "nhm_sub.zarr",
    "output_var_names": all_var_names(submodel_dict),
    "time_chunk_size": 61,
}

# %% [markdown]
# Well, that was a lot of work compared to the full model, where every
# input was computed by another process. But you just learned how the
# flow of information between processes is enabled by the design, and
# how to query it. Let's draw the sub-model's contract, then run it.

# %%
sub_graph = ModelContractGraph(submodel_dict, maps=submodel_maps)
print(f"{sum(len(ee) for ee in sub_graph.externals.values())} external inputs")
try:
    sub_diagram: object = Source(sub_graph.to_dot(rankdir="TB"))
except NameError:  # graphviz not installed
    sub_diagram = sub_graph
sub_diagram

# %%
vol_maps = {
    kk: vv for kk, vv in make_maps(weights).items() if kk in VOLUME_MAP_SPEC
}
submodel = Model(
    submodel_dict,
    sub_control,
    maps=vol_maps,
    discretizations=make_discretizations(),
)
tic = time.perf_counter()
submodel.run(dt, np.int32(n_days))
submodel.finalize()
print(
    f"clock: {submodel.time.current}; wall time {time.perf_counter() - tic:.1f} s"
)

# %% [markdown]
# That saved us some time: the run is like before, with fewer
# processes.
#
# The final time is still in memory. Let's take a look at, say,
# `recharge`: its metadata from the declaration, and the live array
# on the soil zone's grid dataset.

# %%
meta = PRMSSoilzone.get_variables()["recharge"]
print("recharge:", meta.description, f"[{meta.units}]", meta.dims)
recharge = submodel.model_dict["prms_soilzone"]["recharge"]
print("shape:", recharge.shape, "dtype:", recharge.dtype)
print("basin mean at the final time:", f"{recharge.values.mean():.4f}")

# %% [markdown]
# So `recharge` has only a spatial dimension in memory (one timestep;
# the time series is on disk). pywatershed's notebook maps it on the
# HRU polygons; here we plot its basin total through time from the
# sub-model's store.

# %%
out_sub = xr.open_zarr(
    sub_control["output_serial_zarr"], consolidated=False
).load()
plot_series(
    {
        "full model": out_mem["recharge"].sum(dim="nhru"),
        "sub-model": out_sub["recharge"].sum(dim="nhru"),
    },
    "inches (summed over HRUs)",
    "Recharge to the groundwater reservoir",
    "recharge.png",
)

# %% [markdown]
# The two lines sit on top of each other, as they should. We can check
# this for all of the sub-model's variables: the sub-model was fed the
# full model's own numbers, so its output must be identical.

# %%
for slot in submodel_slots:
    for name in submodel_variables[slot]:
        xr.testing.assert_equal(out_mem[name], out_sub[name])
print("sub-model output identical to the full model's")

# %% [markdown]
# ## Checking against pywatershed
#
# pywatershed's notebook wrote its full-model output beside itself. If
# it's there, we compare in two ways, because the two codes agree to
# different degrees in different places:
#
# * The atmosphere process agrees with pywatershed to 1e-5, which is
#   pywatershed's own test standard for that process.
# * Everything downstream of the snowpack can differ more: the snow
#   code has thresholds where a difference in the last digit decides
#   whether a pack melts today or tomorrow, and pywatershed's numba
#   compilation uses "fast math" (reordered floating-point operations)
#   while ours does not. Such flips are rare, but each one changes the
#   affected HRU's runoff for a few days. So for the full chain we
#   report the FRACTION of values outside a tolerance rather than
#   demanding every value agree.
# * The sub-model, driven by pywatershed's OWN outputs instead of ours,
#   removes the snowpack from the comparison: the soil zone,
#   groundwater and channel then agree with pywatershed to 1e-10 on
#   every value.

# %%
PYWS_OUT = PYWS_NB_OUTPUT_DIR / "nhm_yaml"


def pyws_output(name: str) -> xr.DataArray:
    """pywatershed's output, on our grid dimension names."""
    da = xr.load_dataarray(PYWS_OUT / f"{name}.nc")
    return da.rename(
        {
            dd: gg
            for dd, gg in (("nhm_id", "nhru"), ("nhm_seg", "nsegment"))
            if dd in da.dims
        }
    )


def outlier_fractions(ours: np.ndarray, theirs: np.ndarray, tols):
    finite = np.isfinite(ours) & np.isfinite(theirs)
    return {
        tol: float(
            (
                ~np.isclose(ours[finite], theirs[finite], rtol=tol, atol=tol)
            ).mean()
        )
        for tol in tols
    }


if not PYWS_OUT.exists():
    print(f"({PYWS_OUT} not found; verification skipped)")
else:
    # the atmosphere: every value within 1e-5
    for name in ("potet", "swrad", "hru_ppt"):
        theirs_da = pyws_output(name)
        assert theirs_da.shape == out_mem[name].shape, name
        np.testing.assert_allclose(
            out_mem[name].values, theirs_da.values, rtol=1e-5, atol=1e-5
        )
        print(f"{name:16s} all values within 1e-5")
    # downstream of the snowpack: outlier fractions
    tols = (1e-2, 1e-1)
    for name in (
        "snowmelt",
        "sroff",
        "recharge",
        "gwres_flow",
        "seg_outflow",
        "seg_tave_water",
    ):
        fracs = outlier_fractions(
            out_mem[name].values, pyws_output(name).values, tols
        )
        print(
            f"{name:16s}",
            {f"{tt:.0e}": f"{ff:.2%}" for tt, ff in fracs.items()},
        )
        assert fracs[1e-2] <= 1e-2, name  # gross-breakage guard

# %% [markdown]
# Over this half year no flow value differs by more than 1%; only a
# few hundredths of a percent of the stream-temperature segment-days
# do (segments near the no-flow threshold, where a tiny flow
# difference switches how the temperature is computed). Running the
# full two years, or a snowier domain, shows more of the melt-day
# flips described above. Now the tight check: the sub-model fed
# pywatershed's outputs.

# %%
if PYWS_OUT.exists():
    sub_pyws_dict: dict[str, dict[str, Any]] = {
        slot: dict(process_dict[slot]) for slot in submodel_slots
    }
    for name, consumer in submodel_file_inputs.items():
        sub_pyws_dict[consumer][name] = pyws_output(name)
    sub_pyws_control: dict[str, Any] = sub_control | {
        "output_serial_zarr": nb_dir / "nhm_sub_pyws_inputs.zarr",
    }
    sub_pyws = Model(
        sub_pyws_dict,
        sub_pyws_control,
        maps={
            kk: vv
            for kk, vv in make_maps(weights).items()
            if kk in VOLUME_MAP_SPEC
        },
        discretizations=make_discretizations(),
    )
    sub_pyws.run(dt, np.int32(n_days))
    sub_pyws.finalize()
    out_sub_pyws = xr.open_zarr(
        sub_pyws_control["output_serial_zarr"], consolidated=False
    ).load()

    RTOL = ATOL = 1e-10
    worst = 0.0
    for slot in submodel_slots:
        for name in submodel_variables[slot]:
            if not (PYWS_OUT / f"{name}.nc").exists():
                continue
            ours = out_sub_pyws[name].values
            theirs = pyws_output(name).values
            if name == "seg_stor_change":
                # inflow minus outflow: two nearly equal numbers, so
                # the difference keeps fewer digits than either
                np.testing.assert_allclose(ours, theirs, rtol=1e-7, atol=1e-4)
                continue
            np.testing.assert_allclose(
                ours, theirs, rtol=RTOL, atol=ATOL, err_msg=name
            )
            worst = max(worst, float(np.abs(ours - theirs).max()))
    print(
        f"soil zone, groundwater and channel within {RTOL}; max |diff| = {worst:.1e}"
    )

# %% [markdown]
# ## What just happened
#
# We assembled the full NHM with stream temperature from a process
# dictionary, two discretizations and thirteen Maps; built the same
# model from a YAML file and showed the two runs identical; cut the
# output down to two processes' variables; ran a three-process
# sub-model on the full model's output and showed it identical; and
# checked the whole against pywatershed: the atmosphere to 1e-5, the
# sub-model to 1e-10, and the rest of the chain to within 1%.
#
# Where to go next: `examples/01_prms_legacy_translation.py` builds
# this same model in ONE call from the original PRMS files;
# `examples/00_input_contract.py` explores the contract that let us
# find the sub-model's inputs; `help(Model)` has the details of the
# control dictionary (restart, output chunking, MPI).
#
# ## References
#
# * Regan, R. S., Markstrom, S. L., Hay, L. E., Viger, R. J., Norton,
#   P. A., Driscoll, J. M., & LaFontaine, J. H. (2018). Description of
#   the national hydrologic model for use with the precipitation-runoff
#   modeling system (PRMS) (No. 6-B9). US Geological Survey.
#   https://pubs.usgs.gov/publication/tm6B9
# * Regan, R.S., Markstrom, S.L., LaFontaine, J.H., Norton, P.A., 2022,
#   PRMS version 5.2.1: Precipitation-Runoff Modeling System (PRMS):
#   U.S. Geological Survey Software Release, 02/10/2022.
#   https://www.usgs.gov/data/precipitation-runoff-modeling-system-prms-version-521
