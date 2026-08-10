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
    # 13 maps, each with exactly one consumer in this configuration
    assert len(nhm_graph.map_edges) == 13
    # the prior-step back-edge appears as an ordinary internal edge
    pairs = {
        (src, dst) for src, dst, _ in nhm_graph.internal_edges
    }
    assert ("prms_snow", "prms_canopy") in pairs
    # externals: the four CBH forcings
    assert sorted(nhm_graph.externals["nhru"]) == [
        "humidity_hru",
        "prcp",
        "tmax",
        "tmin",
    ]
    assert nhm_graph.externals["nsegment"] == {}


def test_label_elision_and_params(nhm_graph):
    mermaid = nhm_graph.to_mermaid()
    assert " +" in mermaid  # aggregated edge labels elide long lists
    assert "parameters" not in mermaid
    graph = ModelContractGraph(
        {
            "upper": {"class": Upper, "discretization": "hru"},
        },
        show_params=True,
    )
    assert "parameters)" in graph.to_mermaid()


def test_markdown_fence(nhm_graph):
    md = nhm_graph.to_markdown()
    assert md.startswith("```mermaid\n") and md.endswith("\n```")
    assert nhm_graph._repr_markdown_() == md
