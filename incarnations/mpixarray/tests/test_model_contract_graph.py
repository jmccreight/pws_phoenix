"""ModelContractGraph: data-free pins (declarations only -- no data,
no model construction, mirroring Model.input_spec's dry-run nature)."""

import pathlib as pl
import sys

import numpy as np
import pytest

sys.path.append(str(pl.Path(__file__).parent.parent))
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
from model_contract_graph import ModelContractGraph
from process import DataArrayMeta, Process
from processes_concrete import Lower, Upper


class _Carrier(Process):
    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="carrier",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


def _wiring_map(source, target):
    return Map(
        weights=np.zeros((1, 1)),
        grid={"nhru": "nsegment"},
        variable={source: target},
    )


@pytest.fixture()
def nhm_graph():
    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphere,
            "discretization": "nhru",
        },
        "prms_canopy": {"class": PRMSCanopy, "discretization": "nhru"},
        "prms_snow": {"class": PRMSSnow, "discretization": "nhru"},
        "prms_runoff": {"class": PRMSRunoff, "discretization": "nhru"},
        "prms_soilzone": {
            "class": PRMSSoilzone,
            "discretization": "nhru",
        },
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
        },
        "humidity_carrier": {"class": _Carrier, "discretization": "nhru"},
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
    maps = {
        "sroff_vol": _wiring_map("sroff_vol", "seg_sroff_vol"),
        "ssres_vol": _wiring_map("ssres_flow_vol", "seg_ssres_flow_vol"),
        "gw_vol": _wiring_map("gwres_flow_vol", "seg_gwres_flow_vol"),
        **{
            target: _wiring_map(source, target)
            for target, (source, _) in AGGREGATION_MAP_SPEC.items()
        },
    }
    return ModelContractGraph(process_dict, maps=maps)


def test_two_grid_toy():
    process_dict = {
        "upper": {"class": Upper, "discretization": "hru"},
        "lower": {"class": Lower, "discretization": "segment"},
    }
    maps = {
        "hru_to_seg": Map(
            weights=np.zeros((1, 1)),
            grid={"hru": "segment"},
            variable={"flow": "flow"},
        )
    }
    graph = ModelContractGraph(process_dict, maps=maps)
    mermaid = graph.to_mermaid()
    assert 'subgraph hru["grid: hru"]' in mermaid
    assert 'subgraph segment["grid: segment"]' in mermaid
    assert '    upper -. "flow" .-> lower' in mermaid
    assert "    ext_forcing_up --> upper" in mermaid
    assert "    ext_forcing_low --> lower" in mermaid
    assert mermaid.count("time -.->") == 2


def test_nhm_shape(nhm_graph):
    assert sum(len(pp) for pp in nhm_graph.grids.values()) == 10
    assert sorted(nhm_graph.grids) == ["nhru", "nsegment"]
    # 13 mapped variables aggregated onto 9 producer->consumer pairs
    assert len(nhm_graph.map_edges) == 9
    assert sum(len(labels) for _, _, labels in nhm_graph.map_edges) == 13
    # the prior-step back-edge appears as an ordinary internal edge
    pairs = {(src, dst) for src, dst, _ in nhm_graph.internal_edges}
    assert ("prms_snow", "prms_canopy") in pairs
    # externals: the four CBH forcings
    assert sorted(nhm_graph.externals["nhru"]) == [
        "humidity_hru",
        "prcp",
        "tmax",
        "tmin",
    ]
    assert nhm_graph.externals["nsegment"] == {}
    # the Maps' weights requirement, summarized per grid pair
    ww = nhm_graph.map_weights[("nhru", "nsegment")]
    assert ww["count"] == 13
    assert ww["derived"] == 0  # wiring maps carry no derivation
    assert ww["shape"] == ("n_nsegment", "n_nhru")
    dot = nhm_graph.to_dot()
    assert "Map weights: 13 matrices (n_nsegment x n_nhru)" in dot
    assert "Map weights: 13 (n_nsegment x n_nhru)" in (nhm_graph.to_mermaid())


def test_label_elision_and_params(nhm_graph):
    mermaid = nhm_graph.to_mermaid()
    assert " +" in mermaid  # aggregated edge labels elide long lists
    # mermaid (the fallback) carries the one-line parameter summary
    # in the bubble; dot uses the sectioned table (test_supply_side)
    assert (
        "prms_snow<br/>PRMSSnow<br/>params: 21 static + 3 tv + 1 derivable"
        in mermaid
    )


def test_markdown_fence(nhm_graph):
    md = nhm_graph.to_markdown()
    assert md.startswith("```mermaid\n") and md.endswith("\n```")
    assert nhm_graph._repr_markdown_() == md


def test_dot(nhm_graph):
    dot = nhm_graph.to_dot()
    assert "rankdir=LR;" in dot
    assert "subgraph cluster_nhru {" in dot
    assert "subgraph cluster_nsegment {" in dot
    # the prior-step back-edge is drawn but excluded from ranking
    snow_canopy = [
        line
        for line in dot.splitlines()
        if line.strip().startswith("prms_snow -> prms_canopy")
    ]
    assert len(snow_canopy) == 1
    assert "constraint=false" in snow_canopy[0]
    # forward internal edges DO rank
    atmos_canopy = [
        line
        for line in dot.splitlines()
        if line.strip().startswith("prms_atmosphere -> prms_canopy")
    ]
    assert "constraint=false" not in atmos_canopy[0]
    # 13 dashed map EDGES (the Time node is also style=dashed)
    dashed_edges = [
        ll for ll in dot.splitlines() if "->" in ll and "style=dashed" in ll
    ]
    assert len(dashed_edges) == 9  # aggregated pairs (13 variables)
    assert dot.strip().endswith("}")


def test_supply_side(nhm_graph):
    """The contract's supply half: parameter classification, the
    initial-value seams, and the restartable state -- all sectioned
    INSIDE the process node."""
    # classification: dims for static/tv; a declared derivation
    # makes its own class
    atmos = nhm_graph.parameters["prms_atmosphere"]
    assert "tmax_cbh_adj" in atmos["cyclic"]  # (nmonth, space)
    assert "soltab_potsw" in atmos["derivable"]  # declared derivation
    assert "temp_units" in atmos["static"]
    assert "hru_in_to_cf" in nhm_graph.parameters["prms_runoff"]["derivable"]
    # initial-value seams attach to their processes
    assert nhm_graph.initial_values == {
        "prms_groundwater": ["gwstor_init"],
        "prms_channel": ["segment_flow_init"],
    }
    # every restart=True variable is a settable initial condition
    assert "pkwater_equiv" in nhm_graph.restart_vars["prms_snow"]
    assert nhm_graph.restart_vars["prms_groundwater"] == ["gwres_stor"]
    # default: sectioned counts, no names
    dot = nhm_graph.to_dot()
    assert "<B>prms_snow</B>" in dot
    assert "parameters: 21 static" in dot
    assert "initial values: 1" in dot
    n_snow_state = len(nhm_graph.restart_vars["prms_snow"])
    assert f"initial state (restartable): {n_snow_state}" in dot
    assert "albset_rna" not in dot  # names only with show_params
    # show_params expands the names inside the sections
    full = ModelContractGraph(
        {
            "prms_atmosphere": {
                "class": PRMSAtmosphere,
                "discretization": "nhru",
            },
        },
        show_params=True,
    ).to_dot()
    assert "parameters: 15 time-varying" in full  # soltabs -> derivable
    assert "parameters: 2 derivable" in full
    assert 'tmax_cbh_adj<BR ALIGN="LEFT"/>' in full
    assert 'soltab_potsw<BR ALIGN="LEFT"/>' in full  # in its section
    assert "tmax_sum" in full  # restartable state, by name


def test_dot_orientation_and_size(nhm_graph):
    assert "rankdir=TB;" in nhm_graph.to_dot(rankdir="TB")
    assert 'size="10";' in nhm_graph.to_dot(size=10)
    assert 'size="8,11";' in nhm_graph.to_dot(size="8,11")
    assert '    size="' not in nhm_graph.to_dot()
    assert nhm_graph.to_mermaid(direction="LR").startswith("flowchart LR")
