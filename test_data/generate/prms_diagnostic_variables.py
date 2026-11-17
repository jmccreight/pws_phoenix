"""Derive the answer variables PRMS does not write, from the PRMS
answers in one output directory.

Ported from pywatershed's test_data/generate/prms_diagnostic_variables.py
(feat_oracle, Nov 2026). The arithmetic and its order of operations are
pywatershed's; what changed:

- PRMS files are read with pws_phoenix.prms_translate (pyPRMS), not
  pywatershed.
- The first time of every *_prev file is the STARTING STATE of the
  pws_phoenix model the control file translates to (model_from_control:
  the init hooks, then the supplied state from prms_initial_state, e.g.
  gwres_stor <- gwstor_init). pywatershed built one pywatershed process
  per variable instead, chosen by dprst_flag / iter_aet_flag tables;
  resolve_classes makes that choice here.
- The control file and parameters are read once per output directory,
  not once per variable.

Every file is written into the output directory it is derived from
(data_dir == output_dir, as pywatershed's oracle_populate.py calls it).
Names and dims are pywatershed's: the files are read by tests that
read pywatershed's tree.
"""

import pathlib as pl

import numpy as np
import xarray as xr

from pws_phoenix.prms_translate import model_from_control
from pws_phoenix.prms_translate.control import from_control
from pws_phoenix.prms_translate.dyn_param import (
    forward_fill,
    load_dynamic_parameter,
)
from pws_phoenix.prms_translate.parameters import derive_hru_in_to_cf
from pws_phoenix.prms_translate.readers import (
    load_control,
    load_parameters,
)

S_PER_DAY = 86400.0
ZERO = 0.0
NEARZERO = 1.0e-6
# pws_phoenix's snow kernel (prms_snow.py _DNEARZERO). pywatershed's
# dnearzero is np.finfo(np.float64).eps = 2.220446049250313e-16: only a
# pkwater_equiv in [2.2204e-16, 2.23e-16) is classified differently.
DNEARZERO = 2.23e-16

# storages whose *_prev and *_change files are derived (pywatershed's
# previous_vars keys); each is derived only where its PRMS file exists
PREVIOUS_VARS = (
    "gwres_stor",
    "hru_intcpstor",
    "freeh2o",
    "pk_ice",
    "dprst_stor_hru",
    "hru_impervstor",
    "ag_soil_moist",
    "ag_soil_rechr",
    "pref_flow_stor",
    "slow_stor",
    "soil_lower",
    "soil_rechr",
    "ssres_stor",
)
# the PRMS legacy names of some previous-step files (default: *_prev)
PREV_RENAME = {
    "gwres_stor": "gwres_stor_old",
    "hru_impervstor": "hru_impervstor_old",
    "hru_intcpstor": "hru_intcpstor_old",
    "dprst_stor_hru": "dprst_stor_hru_old",
}
# hru depths (inches) -> volumes (cubic feet)
VOLUME_VARS = ("sroff", "ssres_flow", "gwres_flow")
THROUGH_RAIN_INPUTS = (
    "net_ppt",
    "pptmix_nopack",
    "snowmelt",
    "pkwater_equiv",
    "pk_ice_change",
    "freeh2o_change",
    "snow_evap",
    "net_snow",
    "net_rain",
)
CHANNEL_INPUTS = (
    "sroff_vol",
    "ssres_flow_vol",
    "gwres_flow_vol",
    "seg_outflow",
    "seg_inflow",
)


def _control_value(ctl, name: str, default=None):
    """The first value of a control entry, or default when absent."""
    if not ctl.exists(name):
        return default
    return np.atleast_1d(ctl.get(name).values)[0]


def _write(obj: xr.DataArray | xr.Dataset, path: pl.Path) -> None:
    """Write a derived file, first removing whatever is at path: a
    symlink there points into the oracle, and a write through it would
    change the oracle's answers."""
    if path.is_symlink() or path.exists():
        path.unlink()
    obj.to_netcdf(path)


def starting_state(control_file: pl.Path, names: list[str]) -> dict:
    """The starting value of each hru storage in `names` in the
    pws_phoenix model built from control_file (nothing is run)."""
    with model_from_control(control_file) as model:
        hru_ds = model.discretizations["nhru"].dataset
        missing = [nn for nn in names if nn not in hru_ds]
        if missing:
            raise KeyError(
                f"{control_file.name}: PRMS wrote {missing}, but the "
                "pws_phoenix model it translates to has no such hru "
                "variables."
            )
        # copies: the model's arrays are released with the model
        return {nn: hru_ds[nn].values.copy() for nn in names}


def derive_previous(out_dir: pl.Path, var_name: str, start: np.ndarray):
    """{var}_prev (or its PRMS legacy name) and {var}_change: the
    answers shifted one step later, the first step from `start`."""
    current = xr.load_dataarray(out_dir / f"{var_name}.nc")
    prev = current * np.nan
    prev[:] = np.roll(current.values, 1, axis=0)
    # the final value (-1) was wrapped to the first position
    prev[0, :] = start
    change = current - prev

    prev_name = PREV_RENAME.get(var_name, f"{var_name}_prev")
    _write(prev.rename(prev_name), out_dir / f"{prev_name}.nc")
    change_name = f"{var_name}_change"
    _write(change.rename(change_name), out_dir / f"{change_name}.nc")


def derive_volume(
    out_dir: pl.Path, var_name: str, hru_in_to_cf: np.ndarray, gsflow: bool
):
    """{var}_vol = {var} * hru_in_to_cf (cubic feet)."""
    depth = xr.load_dataarray(out_dir / f"{var_name}.nc")
    if gsflow:
        # Zero out tiny values before multiplying by large conversion
        # factor
        depth = depth.where(np.abs(depth) > 1e-11, 0.0)
    vol_name = f"{var_name}_vol"
    vol = (depth * hru_in_to_cf).rename(vol_name)
    _write(vol.to_dataset(), out_dir / f"{vol_name}.nc")


def derive_through_rain(out_dir: pl.Path) -> None:
    """through_rain: the part of net_rain that passes through the snow
    pack, from the snow answers (requires the *_change files)."""
    data = {
        vv: xr.load_dataarray(out_dir / f"{vv}.nc")
        for vv in THROUGH_RAIN_INPUTS
    }
    cond1 = data["net_ppt"] > ZERO
    cond2 = data["pptmix_nopack"] != 0
    cond3 = data["snowmelt"] < NEARZERO
    cond4 = data["pkwater_equiv"] < DNEARZERO
    cond5 = data["snow_evap"] < NEARZERO
    cond6 = data["net_snow"] < NEARZERO
    cond7 = data["snow_evap"] > -1 * (
        data["pk_ice_change"] + data["freeh2o_change"]
    )

    through_rain = data["net_rain"] * ZERO
    # these are in reverse order
    through_rain[:] = np.where(
        cond1 & cond3 & cond4 & cond6, data["net_rain"], ZERO
    )
    through_rain[:] = np.where(
        cond1 & cond3 & cond4 & cond5, data["net_ppt"], through_rain
    )
    through_rain[:] = np.where(cond1 & cond2, data["net_rain"], through_rain)
    # This condition does not exist in PRMS (pywatershed's comment) but
    # is necessary for mass balance: rain on snow (no new snow), then
    # snow_evap consumes the pack during the timestep.
    through_rain[:] = np.where(cond1 & cond6 & cond7, ZERO, through_rain)
    _write(
        through_rain.to_dataset(name="through_rain"),
        out_dir / "through_rain.nc",
    )


def derive_infil_hru(
    out_dir: pl.Path, params: xr.Dataset, dprst: bool, ag_frac
) -> None:
    """infil_hru: infil (pervious area) as an hru-area depth, plus
    infil_ag over the ag area when `ag_frac` is not None."""
    imperv_frac = params["hru_percent_imperv"].values
    dprst_frac = params["dprst_frac"].values if dprst else ZERO
    perv_frac = 1.0 - imperv_frac - dprst_frac
    infil = xr.load_dataarray(out_dir / "infil.nc").rename("infil_hru")
    if ag_frac is None:
        infil_hru = infil * perv_frac
    else:
        perv_frac = 1.0 - imperv_frac - dprst_frac - ag_frac
        infil_ag = xr.load_dataarray(out_dir / "infil_ag.nc")
        infil_hru = infil * perv_frac + (infil_ag * ag_frac)
    # infil_hru + infil_ag drops the name
    _write(infil_hru.to_dataset(name="infil_hru"), out_dir / "infil_hru.nc")


def derive_channel(out_dir: pl.Path, params: xr.Dataset) -> None:
    """The muskingum_mann channel's lateral inflow and volumes:
    channel_{sroff,ssres_flow,gwres_flow}_vol (hru), seg_lateral_inflow,
    seg_stor_change, channel_outflow_vol (segment)."""
    data = {
        vv: xr.load_dataarray(out_dir / f"{vv}.nc") for vv in CHANNEL_INPUTS
    }
    ntimes = data["seg_outflow"].shape[0]
    nseg = params.sizes["nsegment"]
    nhru = params.sizes["nhru"]
    hru_segment = params["hru_segment"].values - 1
    to_segment = (params["tosegment"].values - 1).astype("int64")
    outflow_mask = to_segment < 0

    seg_stor_change = (data["seg_inflow"] - data["seg_outflow"]) * S_PER_DAY
    channel_outflow_vol = (
        np.where(outflow_mask, data["seg_outflow"], ZERO)
    ) * S_PER_DAY

    seg_lateral_inflow = np.zeros((ntimes, nseg))
    channel_sroff_vol = np.zeros((ntimes, nhru))
    channel_ssres_flow_vol = np.zeros((ntimes, nhru))
    channel_gwres_flow_vol = np.zeros((ntimes, nhru))
    # hru by hru, in hru order: the order of the sums into each segment
    for ihru in range(nhru):
        iseg = hru_segment[ihru]
        if iseg < 0:
            # an hru with no segment: its volumes are not routed
            # (pywatershed's comment calls this discarded mass)
            continue
        channel_sroff_vol[:, ihru] = data["sroff_vol"].values[:, ihru]
        channel_ssres_flow_vol[:, ihru] = data["ssres_flow_vol"].values[
            :, ihru
        ]
        channel_gwres_flow_vol[:, ihru] = data["gwres_flow_vol"].values[
            :, ihru
        ]
        # cubic feet to cfs
        lateral_inflow = (
            channel_sroff_vol[:, ihru]
            + channel_ssres_flow_vol[:, ihru]
            + channel_gwres_flow_vol[:, ihru]
        ) / S_PER_DAY
        seg_lateral_inflow[:, iseg] += lateral_inflow

    seg_template = data["seg_outflow"]
    hru_template = data["sroff_vol"]
    results = {
        "channel_sroff_vol": (hru_template, channel_sroff_vol),
        "channel_ssres_flow_vol": (hru_template, channel_ssres_flow_vol),
        "channel_gwres_flow_vol": (hru_template, channel_gwres_flow_vol),
        "seg_lateral_inflow": (seg_template, seg_lateral_inflow),
        "seg_stor_change": (seg_template, seg_stor_change),
        "channel_outflow_vol": (seg_template, channel_outflow_vol),
    }
    for name, (template, values) in results.items():
        out = template.rename(name) * np.nan
        out[:, :] = values
        _write(out, out_dir / f"{name}.nc")


def derive_all(control_file: pl.Path, out_dir: pl.Path) -> None:
    """Every derived variable of the answers in out_dir, which
    control_file produced. The order matters: through_rain reads the
    snow *_change files, the channel reads the *_vol files."""
    ctl = load_control(control_file)
    cfg = from_control(ctl, control_file)
    params = load_parameters(cfg.param_files)
    modules = dict(ctl.modules)
    dprst = bool(int(_control_value(ctl, "dprst_flag", 0)))
    exe_desc = str(_control_value(ctl, "executable_desc", "prms"))
    gsflow = "gsflow" in exe_desc.lower()

    present = [
        vv for vv in PREVIOUS_VARS if (out_dir / f"{vv}.nc").exists()
    ]
    if present:
        start = starting_state(control_file, present)
        for vv in present:
            derive_previous(out_dir, vv, start[vv])

    hru_in_to_cf = derive_hru_in_to_cf(params).values
    for vv in VOLUME_VARS:
        if (out_dir / f"{vv}.nc").exists():
            derive_volume(out_dir, vv, hru_in_to_cf, gsflow)

    derive_through_rain(out_dir)

    ag_frac = None
    if modules.get("soilzone_module") == "soilzone_ag":
        if "ag_frac" in cfg.dyn_param_paths:
            times = xr.load_dataarray(out_dir / "infil.nc")["time"].values
            dyn = load_dynamic_parameter(cfg.dyn_param_paths["ag_frac"])
            ag_frac = forward_fill(dyn, times).values
        else:
            ag_frac = params["ag_frac"].values
    derive_infil_hru(out_dir, params, dprst, ag_frac)

    if modules.get("strmflow_module") == "muskingum_mann":
        # a run that wrote no channel answers (fgr_ag_2yr: no
        # seg_inflow) has nothing to derive them from; pywatershed
        # skipped it the same way
        missing = [
            vv for vv in CHANNEL_INPUTS if not (out_dir / f"{vv}.nc").exists()
        ]
        if missing:
            print(f"{out_dir}: channel variables not derived, no {missing}")
        else:
            derive_channel(out_dir, params)
