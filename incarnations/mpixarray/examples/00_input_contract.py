# %% [markdown]
# # The input contract
#
# **To open this file as a notebook** in JupyterLab: right-click it
# in the file browser and choose **Open With -> Notebook** (requires
# the `jupytext` extension in the JupyterLab environment). It also
# runs top-to-bottom as a plain Python script.
#
# In this notebook we ask a model configuration what must be
# SUPPLIED to run it. The answer comes live from
# `Model.input_spec()`, which derives it from the process classes'
# own field declarations -- nothing below is maintained by hand, so
# what you see IS the real contract.
#
# We'll render the full NHM chain through stream temperature (13
# Maps -- the model of `tests/test_prms_stream_temp_full_chain.py`);
# change the `process_dict`/`maps` cells to render any other
# configuration. At the end we bring it full circle: the contract is
# enough to BUILD the model, and we do, and verify a short run
# against PRMS itself.
#
# The returned spec has two halves (required first -- the input
# contract proper; the informational half must be asked for):
#
# 1. **required** -> grid -> external inputs, parameters (authored
#    vs DERIVABLE -- required either way, but derivable ones state
#    how to generate them), initial values (the `initial=` seams),
#    and initial state (the restartable warm-start surface)
# 2. **maps** -> each Map in the configuration implies ONE weights
#    matrix (required supply; kept beside "required" because that
#    dict's keys are grids)
# 3. **optional** (`include_optional=True`) -> grid -> internal
#    inputs, derived parameters, map-fed inputs

# %% [markdown]
# ## Reading notes
#
# - **External inputs** are the inputs proper: time-varying data
#   served in model-time lockstep (forcings/boundary data). A name
#   consumed by several processes is supplied ONCE and shared
#   structurally.
# - **Parameter packaging is not part of the contract.** At assembly
#   each declared parameter is looked up FIRST in the grid's
#   `Discretization` parameter dataset (the `parameters_dis_*.nc`
#   style files), THEN in the process entry's `parameters` dataset
#   (`model.py _add_process_fields`, "sourced DIS-FIRST"). So the
#   contract lists WHAT is needed; deciding which file carries each
#   name is a data-preparation (translation-layer) decision. The
#   source scan at the bottom shows where each name lives in drb_2yr
#   TODAY.
# - **Initial conditions come in two forms.** (a) The `initial=`
#   seams (spec "initial_values"): direct supplies of a state
#   variable's start values via a process_dict entry. Only TWO exist
#   (`gwstor_init`, `segment_flow_init`). (b) PRMS-convention
#   `*_init*` PARAMETERS (soilzone's `soil_moist_init_frac`, runoff's
#   `dprst_frac_init`, snow's `snowpack_init`, ...): ordinary
#   supplied parameters, typically consumed by `initialize()` --
#   listed by name in a cell below (the convention over-matches:
#   snow's `den_init` is the density of new-fallen snow, a physics
#   parameter, not a state IC). All other state initializes to fixed
#   values in `initialize()`.
# - **Three kinds of parameter, patently.** AUTHORED (your data and
#   calibration), DERIVABLE (required at assembly like any parameter,
#   but the contract states the factory/formula that generates them
#   -- the solar tables, `segment_order`, `hru_in_to_cf`), and
#   INTERNAL (`kind="parameter_internal"`: computed by
#   `initialize()`, never supplied, reported only in the optional
#   half). "Derivable" and "internal" are deliberately distinct
#   words for distinct lifecycles.
# - **Restart is the third source of initial state**, beyond the
#   `initial=` seams and the `*_init*` parameters:
#   `Model.write_restart(dir)` saves the prognostic state and
#   `control["restart_read"] = dir` warm-starts a new model from it.
#   Conveniently, a warm start takes the SAME (superset) input files
#   as the original run -- so no restart-specific input preparation
#   exists in this contract. See `help(Model)` for the details.
# - **CBH forcing files store float32; the model computes float64.**
#   The widening happens at input preparation (it is exact -- every
#   float32 value is exactly representable in float64) and matters
#   for matching PRMS results precisely. The translation layer
#   handles it for you (`prms_translate.load_cbh`).

# %%
import pathlib as pl
import sys
from pprint import pprint
from typing import Any

import numpy as np
import xarray as xr

# as a script __file__ locates us; in a Jupyter kernel there is no
# __file__ -- the kernel's cwd is the notebook's directory (JupyterLab
# default). Either way we are in examples/, one level inside the
# incarnation package dir.
try:
    _here = pl.Path(__file__).parent
except NameError:
    _here = pl.Path.cwd()
_pkg = _here.parent
assert (_pkg / "model.py").exists(), (
    f"expected to run from incarnations/mpixarray/examples, not {_here}"
)
sys.path.append(str(_pkg))
from atmosphere.prms_atmosphere import PRMSAtmosphere
from hydrology.prms_canopy import PRMSCanopy
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import PRMSGroundwater
from hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryWidthOnly,
)
from hydrology.prms_runoff import PRMSRunoff
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone
from hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    PRMSStreamTemp,
)
from map import Map
from model import Model
from process import DataArrayMeta, Process

MPIX_ROOT = _pkg.parents[2]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"

# %% [markdown]
# ## The configuration
#
# The full chain's `process_dict`: classes + grids only -- that is
# all `input_spec()` reads.
#
# **Why a carrier for humidity but not the other CBH forcings?**
# `prcp`/`tmax`/`tmin` are DECLARED inputs of `PRMSAtmosphere`, so
# they enter the contract through it. `humidity_hru` is consumed by
# NO process in this model -- only the humidity aggregation Map reads
# it off the hru-grid dataset -- and only processes declare fields,
# so a 1-variable compute-free carrier declares it. (A Map that could
# declare its own source-variable need would remove the carrier; a
# possible future Map extension.)


# %%
class HumidityCarrier(Process):
    """The humidity CBH forcing on the hru grid (external data)."""

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


process_dict: dict[str, dict[str, Any]] = {
    "prms_atmosphere": {"class": PRMSAtmosphere, "discretization": "nhru"},
    "prms_canopy": {"class": PRMSCanopy, "discretization": "nhru"},
    "prms_snow": {"class": PRMSSnow, "discretization": "nhru"},
    "prms_runoff": {"class": PRMSRunoff, "discretization": "nhru"},
    "prms_soilzone": {"class": PRMSSoilzone, "discretization": "nhru"},
    "prms_groundwater": {
        "class": PRMSGroundwater,
        "discretization": "nhru",
    },
    "humidity_carrier": {
        "class": HumidityCarrier,
        "discretization": "nhru",
    },
    "prms_channel": {"class": PRMSChannel, "discretization": "nsegment"},
    "prms_hydraulic_geometry": {
        "class": PRMSHydraulicGeometryWidthOnly,
        "discretization": "nsegment",
    },
    "prms_stream_temp": {
        "class": PRMSStreamTemp,
        "discretization": "nsegment",
    },
}

# %% [markdown]
# The 13 Maps' wiring. These are DUMMY Maps: `input_spec()` reads
# only each Map's grid/variable wiring, so the weights are a
# meaningless `np.zeros((1, 1))` placeholder (a real model derives
# them -- `derive_aggregation_weights()` for the ten aggregations,
# the 0/1 `hru_segment` matrix for the three volumes).


# %%
def dummy_wiring_map(source, target):
    return Map(
        weights=np.zeros((1, 1)),  # placeholder; wiring only
        grid={"nhru": "nsegment"},
        variable={source: target},
    )


maps = {
    "sroff_vol": dummy_wiring_map("sroff_vol", "seg_sroff_vol"),
    "ssres_vol": dummy_wiring_map("ssres_flow_vol", "seg_ssres_flow_vol"),
    "gw_vol": dummy_wiring_map("gwres_flow_vol", "seg_gwres_flow_vol"),
    **{
        target: dummy_wiring_map(source, target)
        for target, (source, _) in AGGREGATION_MAP_SPEC.items()
    },
}

# %% [markdown]
# ## The contract

# %%
spec = Model.input_spec(process_dict, maps=maps, include_optional=True)
pprint(spec["required"], sort_dicts=False)

# %% [markdown]
# ## The optional (informational) half

# %%
pprint(spec["optional"], sort_dicts=False)

# %% [markdown]
# ## The Maps' requirement: weights
#
# Each Map carries one variable between the grids and requires ONE
# weights matrix (rows map into the target grid). The `derivation`
# field records how the matrix is obtained when the map's author
# knows -- our wiring maps here carry none (`None` = you supply it),
# but the translation layer records its derivations (the
# `hru_segment` assignment matrix for the flow volumes, the
# basis-probed aggregation weights for stream temperature), as the
# full-circle section below demonstrates with REAL weights.

# %%
pprint(spec["maps"], sort_dicts=False)

# %% [markdown]
# ## The contract, drawn
#
# The same information makes a picture: `ModelContractGraph` renders
# the configuration from the declarations alone. Each grid is a
# cluster; each process is a TABLE -- a white header (the process,
# which the model computes) over GRAY sections for everything YOU
# supply: its parameters (static | time-varying), any initial
# values, and its restartable initial state (every prognostic
# variable can be given a starting value -- a warm start supplies
# all of them at once; see `help(Model)` on restart). Solid labeled
# edges are on-grid couplings, dashed edges are the Maps carrying
# variables between grids, gray ellipses are the external forcings,
# and Time (the model clock) reaches every process. Gray = supplied,
# white = computed -- that is the contract at a glance.
#
# With the `graphviz` package installed you get the STRUCTURED
# layout -- data flows top to bottom in schedule order (pass
# `rankdir="LR"` for landscape, `size=` to cap inches). Without it,
# the cell falls back to mermaid (renders in JupyterLab >= 4.1, but
# scatters more); `print(graph.to_dot())` always works for pasting
# into any dot viewer.

# %%
from model_contract_graph import ModelContractGraph

graph = ModelContractGraph(process_dict, maps=maps)
print(
    f"{sum(len(pp) for pp in graph.grids.values())} processes on "
    f"{len(graph.grids)} grids; {len(graph.internal_edges)} internal "
    f"couplings, "
    f"{sum(len(ll) for _, _, ll in graph.map_edges)} map-fed inputs, "
    f"{sum(len(ee) for ee in graph.externals.values())} external inputs"
)
try:
    from graphviz import Source

    diagram: object = Source(graph.to_dot(rankdir="TB"))
except ImportError:
    diagram = graph  # mermaid fallback (JupyterLab >= 4.1)
diagram

# %% [markdown]
# The section headers carry the counts; `show_params=True` expands
# every NAME inside its section -- the complete contract as one
# (tall) reference figure.

# %%
graph_full = ModelContractGraph(process_dict, maps=maps, show_params=True)
try:
    diagram_full: object = Source(graph_full.to_dot(rankdir="TB"))
except NameError:  # graphviz not installed
    diagram_full = graph_full
diagram_full

# %% [markdown]
# ## Parameters matching the PRMS `*_init*` naming convention
#
# (usually initial conditions consumed by `initialize()`; see the
# reading notes)

# %%
pprint(
    {
        grid: [nn for nn in gg["parameters"] if "_init" in nn]
        for grid, gg in spec["required"].items()
    },
    sort_dicts=False,
)

# %% [markdown]
# ## Where each required name lives in drb_2yr today
#
# A mechanical scan of the domain's `parameters_*.nc` + the CBH
# files: purely informational (the current packaging, not the
# contract). The DERIVABLE names annotate THEMSELVES now -- their
# declared derivations come straight from the contract -- and names
# that ride under a different native-PRMS name are annotated by hand.

# %%
DERIVABLE = {
    name: f"derivable: {info['derivation']}"
    for gg in spec["required"].values()
    for name, info in gg["derivable_parameters"].items()
}
KNOWN_RENAMED = {
    "humidity_hru": (
        "cbh.nc:rhavg (native PRMS name; humidity_hru is pywatershed's rename)"
    ),
}

sources: dict[str, list[str]] = {}
for ff in sorted(DOMAIN_DIR.glob("parameters_*.nc")):
    with xr.open_dataset(ff) as ds:
        for name in ds.data_vars:
            sources.setdefault(str(name), []).append(ff.name)
for nn in ("prcp", "tmax", "tmin"):
    if (DOMAIN_DIR / f"{nn}.nc").exists():
        sources.setdefault(nn, []).append(f"{nn}.nc (float32 CBH)")


def source_of(name):
    if name in DERIVABLE:
        return DERIVABLE[name]
    if name in KNOWN_RENAMED:
        return KNOWN_RENAMED[name]
    found = sources.get(name)
    return ", ".join(found) if found else "NOT IN ANY DOMAIN FILE"


for grid, gg in spec["required"].items():
    supplied = (
        list(gg["external_inputs"])
        + list(gg["parameters"])
        + list(gg["derivable_parameters"])
        + list(gg["initial_values"])
    )
    print(f"\n=== grid '{grid}' ===")
    pprint({nn: source_of(nn) for nn in supplied}, sort_dicts=False)

# %% [markdown]
# ## Full circle: supply the contract and assemble the Model
#
# The spec + the source scan are enough to BUILD the model, so let's
# do it. The cells below fill `process_dict` generically FROM THE
# CONTRACT: every required parameter is loaded by name from the
# first file the scan found it in (duplicated names carry identical
# NHM values), the `initial=` seams likewise, the computed names
# come from their factories (the soltabs; `segment_order` rides the
# Discretization), and the external inputs are loaded with the
# preparation steps done by hand -- float32 -> float64 widening, the
# `nhm_id -> nhru` dim rename, and `humidity_hru` straight from its
# native source, `cbh.nc:rhavg`. The dummy Maps are rebuilt with
# REAL weights. Constructing the `Model` then runs the full assembly
# validation (every input resolved, map wiring and shapes, the
# initialize hooks) -- the proof the contract is complete -- and a
# short run brings it home.
#
# This hand-rolled supply loop is exactly the job the PRMS
# translation layer owns: see `01_prms_legacy_translation.py` (this
# directory) for the SAME model built from the legacy PRMS files in
# one call.

# %%
from atmosphere.prms_solar_geometry import compute_soltabs
from discretization import Discretization
from hydrology.prms_stream_temp import derive_aggregation_weights

_file_cache: dict[str, xr.Dataset] = {}


def domain_ds(fname):
    if fname not in _file_cache:
        _file_cache[fname] = xr.load_dataset(DOMAIN_DIR / fname)
    return _file_cache[fname]


dis_hru = domain_ds("parameters_dis_hru.nc")
dis_seg = domain_ds("parameters_dis_seg.nc")
soltabs = compute_soltabs(dis_hru)

# %% [markdown]
# ### Parameters + initial values, from the contract

# %%
for entry in process_dict.values():
    das = {}
    for name in entry["class"].get_parameters():
        # derivable names route to their factories (soltabs) or the
        # Discretization (segment_order); hru_in_to_cf, also
        # derivable, happens to ride in this domain's dis files and
        # loads like anything else
        if name.startswith("soltab_"):
            das[name] = soltabs[name]
            continue
        if name == "segment_order":
            continue  # the Discretization's topo_order derives it
        das[name] = domain_ds(sources[name][0])[name]
    if das:
        entry["parameters"] = xr.Dataset(das)

for grid, gg in spec["required"].items():
    for init_name, info in gg["initial_values"].items():
        process_dict[info["process"]][init_name] = domain_ds(
            sources[init_name][0]
        )[init_name]

# %% [markdown]
# ### External inputs, from their native sources


# %%
def load_external(name):
    if name == "humidity_hru":  # the rhavg RENAME (translation layer)
        cbh = xr.load_dataset(DOMAIN_DIR / "cbh.nc")
        return (
            cbh["rhavg"]
            .astype(np.float64)
            .rename({"hru": "nhru"})
            .assign_coords(time=("time", cbh["datetime"].values))
        )
    return (  # float32 CBH -> f64 (exact); output-file dim -> grid dim
        xr.load_dataarray(DOMAIN_DIR / f"{name}.nc")
        .rename({"nhm_id": "nhru"})
        .astype(np.float64)
    )


for grid, gg in spec["required"].items():
    for name, info in gg["external_inputs"].items():
        process_dict[info["consumers"][0]][name] = load_external(name)

# %% [markdown]
# ### Discretizations + the Maps with REAL weights

# %%
discretizations = {
    "nhru": Discretization(
        ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
    ),
    "nsegment": Discretization(
        ["nsegment"],
        parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
        topo_order={"segment_order": "tosegment"},
    ),
}
seg_dis_params = discretizations["nsegment"].parameters
assert seg_dis_params is not None

stp = domain_ds("parameters_PRMSStreamTemp.nc")
agg_weights = derive_aggregation_weights(
    stp["hru_segment"].values.astype(np.int64),
    dis_hru["hru_area"].values,
    dis_seg["tosegment"].values.astype(np.int64),
    seg_dis_params["segment_order"].values.astype(np.int64),
    stp["seg_close"].values,
)
hru_segment = domain_ds("parameters_PRMSChannel.nc")["hru_segment"].values
vol_weights = np.zeros((dis_seg.sizes["nsegment"], hru_segment.shape[0]))
for ihru in range(hru_segment.shape[0]):
    if hru_segment[ihru] > 0:
        vol_weights[hru_segment[ihru] - 1, ihru] = 1.0


def real_map(source, target, ww):
    return Map(
        weights=ww, grid={"nhru": "nsegment"}, variable={source: target}
    )


maps = {
    "sroff_vol": real_map("sroff_vol", "seg_sroff_vol", vol_weights),
    "ssres_vol": real_map("ssres_flow_vol", "seg_ssres_flow_vol", vol_weights),
    "gw_vol": real_map("gwres_flow_vol", "seg_gwres_flow_vol", vol_weights),
    **{
        target: real_map(source, target, agg_weights[wkey])
        for target, (source, wkey) in AGGREGATION_MAP_SPEC.items()
    },
}

# %% [markdown]
# ### Assemble (construction IS the validation) and run a few days

# %%
model = Model(process_dict, {}, maps=maps, discretizations=discretizations)
print(f"assembled: {len(model.model_dict)} processes, ntime={model.ntime}")

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
# ### Verify against the PRMS (Fortran) answers
#
# Two things worth confirming directly rather than trusting:
#
# - **The near-zero mean**: it is early January in the Delaware
#   basin -- PRMS's own mean over the same segments and day should
#   match at the stream-temperature tolerance (5e-3; the full
#   731-day validation lives in
#   `tests/test_prms_stream_temp_full_chain.py`).
# - **The one non-finite segment**: the basin's single never-has-flow
#   segment (no HRUs anywhere upstream). We mark it NaN; Fortran has
#   no NaN convention and writes its -99.9 "invalid" sentinel there
#   for the entire run -- shown below.

# %%
answers_file = DOMAIN_DIR / "output_stream_temp" / "seg_tave_water.nc"
if answers_file.exists():
    answers = xr.load_dataarray(answers_file)
    ans_day = answers.values[n_days - 1, :]
    ours = seg_tave_water.values
    print(f"day {n_days} mean, PRMS: {ans_day[finite].mean():.3f} degC")
    print(f"day {n_days} mean, ours: {ours[finite].mean():.3f} degC")
    np.testing.assert_allclose(
        ours[finite], ans_day[finite], rtol=5.0e-3, atol=5.0e-3
    )
    print("elementwise match on finite segments at 5e-3: OK")
    never = np.where(~finite)[0]
    print(
        f"non-finite segment index: {never}; PRMS value there over "
        f"the whole run: {np.unique(answers.values[:, never])} "
        "(the constant never-flow sentinel)"
    )
else:
    print(f"answers not generated ({answers_file}); skipping")

# %% [markdown]
# ## Where to go next
#
# We asked a configuration for its contract, saw where every
# required name lives in this domain, and proved the contract
# complete by building and running the model from it.
#
# - `01_prms_legacy_translation.py` (this directory) does all of the
#   above in ONE CALL, starting from the legacy PRMS files -- the
#   translation layer whose job this notebook specifies.
# - `Model.input_spec()` renders the contract for ANY configuration:
#   edit the `process_dict`/`maps` cells above and re-run.
# - The contract is pinned data-free in `tests/test_input_spec.py`.
