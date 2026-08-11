"""Model.input_spec(): the declaration-derived input contract.

Dry-run resolution from classes + grid assignments + map wiring alone
-- no data files, so these tests run everywhere (no skips). Pinned
against the two real configurations whose supply lists the test suite
already exercises with data:

- PRMSStreamTemp standalone: the externals must be EXACTLY the 13
  disk inputs test_prms_stream_temp.py feeds.
- The full NHM chain through stream temperature (13 Maps): the hru
  externals must be exactly CBH + humidity_hru, and the segment grid
  must need NOTHING external -- everything internal or map-fed
  (test_prms_stream_temp_full_chain.py supplies exactly that).

The contract is the single source of truth for the PRMS translation
layer; if a declaration change alters what a model consumes, these
pins move too.
"""

import pathlib as pl
import sys

import numpy as np
import pytest

sys.path.append(str(pl.Path(__file__).parent.parent))
from atmosphere.prms_atmosphere import PRMSAtmosphere
from hydrology.prms_canopy import PRMSCanopy
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import PRMSGroundwater
from hydrology.prms_hydraulic_geometry import PRMSHydraulicGeometryWidthOnly
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


class _HumidityCarrier(Process):
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


def _dummy_map(source, target):
    """Wiring only -- input_spec never touches the weights."""
    return Map(
        weights=np.zeros((1, 1)),
        grid={"nhru": "nsegment"},
        variable={source: target},
    )


def test_stream_temp_standalone_externals():
    process_dict = {
        "prms_stream_temp": {
            "class": PRMSStreamTemp,
            "discretization": "nsegment",
        }
    }
    spec = Model.input_spec(process_dict)
    # the optional (informational) half must be actively requested
    assert set(spec) == {"required"}
    assert set(spec["required"]) == {"nsegment"}
    seg = spec["required"]["nsegment"]
    assert set(seg) == {"external_inputs", "parameters", "initial_values"}
    # exactly what test_prms_stream_temp.py supplies from disk
    assert set(seg["external_inputs"]) == {
        "seg_outflow",
        "seg_lateral_inflow",
        "seg_flow_width",
        "seg_tave_air",
        "seg_humid",
        "seg_ccov",
        "seg_melt",
        "seg_rain",
        "seg_potet",
        "seginc_swrad",
        "seginc_sroff",
        "seginc_ssflow",
        "seginc_gwflow",
    }
    # derived parameters are internal, never in the supply set
    assert "_seg_slope" not in seg["parameters"]

    optional = Model.input_spec(process_dict, include_optional=True)[
        "optional"
    ]["nsegment"]
    assert not optional["internal_inputs"]
    assert not optional["map_fed_inputs"]
    assert "_seg_slope" in optional["derived_parameters"]


def test_full_chain_contract():
    """The complete NHM through stream temperature: only CBH +
    humidity_hru are external, and the segment grid needs nothing."""
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
        "humidity_carrier": {
            "class": _HumidityCarrier,
            "discretization": "nhru",
        },
        "prms_channel": {
            "class": PRMSChannel,
            "discretization": "nsegment",
        },
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
        "sroff_vol": _dummy_map("sroff_vol", "seg_sroff_vol"),
        "ssres_vol": _dummy_map("ssres_flow_vol", "seg_ssres_flow_vol"),
        "gw_vol": _dummy_map("gwres_flow_vol", "seg_gwres_flow_vol"),
        **{
            target: _dummy_map(source, target)
            for target, (source, _) in AGGREGATION_MAP_SPEC.items()
        },
    }
    spec = Model.input_spec(process_dict, maps=maps, include_optional=True)
    assert set(spec) == {"required", "maps", "optional"}
    assert set(spec["required"]) == {"nhru", "nsegment"}

    # each Map implies ONE weights matrix (required supply)
    assert len(spec["maps"]) == 13
    sroff = spec["maps"]["sroff_vol"]
    assert sroff["source_grid"] == "nhru"
    assert sroff["target_grid"] == "nsegment"
    assert sroff["target_var"] == "seg_sroff_vol"
    assert sroff["weights_shape"] == ("n_nsegment", "n_nhru")
    assert sroff["derivation"] is None  # wiring map: modeler supplies
    hru, seg = spec["required"]["nhru"], spec["required"]["nsegment"]
    hru_opt = spec["optional"]["nhru"]
    seg_opt = spec["optional"]["nsegment"]

    # hru grid: the raw CBH files + the humidity forcing, nothing else
    assert set(hru["external_inputs"]) == {
        "prcp",
        "tmax",
        "tmin",
        "humidity_hru",
    }
    # the back-edges and mutable inputs resolve internally
    for name, producer in (
        ("pptmix", "prms_atmosphere"),  # canopy edits it in place
        ("sroff", "prms_runoff"),  # soilzone's dunnian mutable input
        ("pk_ice_prev", "prms_snow"),  # canopy's prior-step back-edge
        ("soil_lower_prev", "prms_soilzone"),  # runoff's back-edge
    ):
        assert hru_opt["internal_inputs"][name]["producer"] == producer

    # segment grid: NOTHING external -- channel/hydraulic geometry
    # feed stream temp structurally, the Maps feed the rest
    assert not seg["external_inputs"]
    assert set(seg_opt["map_fed_inputs"]) == {
        "seg_sroff_vol",
        "seg_ssres_flow_vol",
        "seg_gwres_flow_vol",
        *AGGREGATION_MAP_SPEC,
    }
    for name, producer in (
        ("seg_outflow", "prms_channel"),
        ("seg_lateral_inflow", "prms_channel"),
        ("seg_flow_width", "prms_hydraulic_geometry"),
    ):
        assert seg_opt["internal_inputs"][name]["producer"] == producer

    # the initial= seams surface (the *_init PARAMETERS -- soilzone's
    # soil_moist_init_frac etc. -- are ordinary supplied parameters
    # and ride the "parameters" list; restart is NOT implemented)
    assert seg["initial_values"]["segment_flow_init"] == {
        "variable": "seg_outflow",
        "process": "prms_channel",
    }
    assert "gwstor_init" in hru["initial_values"]
    assert "soil_moist_init_frac" in hru["parameters"]

    # the categories partition the declared input names
    for req, opt in ((hru, hru_opt), (seg, seg_opt)):
        cats = (
            set(req["external_inputs"])
            | set(opt["internal_inputs"])
            | set(opt["map_fed_inputs"])
        )
        assert len(cats) == (
            len(req["external_inputs"])
            + len(opt["internal_inputs"])
            + len(opt["map_fed_inputs"])
        )


def test_ignores_data_entries_and_default_grid():
    """Extra entry keys (paths, arrays) are ignored; a missing
    'discretization' falls back to the class default / 'space'."""

    class _Toy(Process):
        pp = DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="",
        )
        ff = DataArrayMeta(
            kind="input", dims=("space",), dtype=np.float64, description=""
        )
        vv = DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="",
        )

        def advance(self) -> None:
            pass

        def calculate(self, dt, time) -> None:
            pass

    spec = Model.input_spec(
        {
            "toy": {
                "class": _Toy,
                "parameters": pl.Path("/does/not/exist.nc"),
                "ff": object(),
            }
        }
    )
    assert set(spec["required"]) == {"space"}
    assert set(spec["required"]["space"]["external_inputs"]) == {"ff"}
    assert set(spec["required"]["space"]["parameters"]) == {"pp"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
