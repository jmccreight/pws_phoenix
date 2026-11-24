"""Shared fixtures for the pws_phoenix regression tests.

The serial (test_up_low_regression.py) and MPI
(test_up_low_regression_mpi.py) regressions build the SAME toy
Upper/Lower model from ONE input dataset (`make_toy_input`) and validate against
the SAME vectorized answers (`compute_answers`). Differences between the two are
intentional and explicit:

  - serial brings the input into memory (or round-trips it through per-input
    files); MPI writes ONE combined file and streams it.
  - `time` and `space` are real dim-coordinates (required by mpixarray's
    `as_iterable`/`parallelize`; the serial path reads them the same way).
  - both serial and MPI now use `param_up_1`, a cyclic-monthly time-varying
    parameter `(month, space)`, indexed each step via `time.month` (it stays
    resident through `as_iterable`, since `month` is not the streaming dim).
  - output backend differs (serial zarr store vs MPI streamed NetCDF).
"""

import os
import pathlib as pl
import platform
import sys
import tomllib

import numpy as np
import pytest
import xarray as xr
import yaml

# The numba on-disk kernel cache is OFF for every test run (bare pytest
# and run_tests.sh alike; each MPI rank runs its own conftest). A cached
# kernel keeps stale copies of jitted callees/constants from OTHER files
# (DEVELOPER.md "Numba cache") -- never let that reach a test. Must
# precede the first pws_phoenix import (the switch is read at import).
os.environ["PWS_NUMBA_CACHE"] = "0"

# ---- the pywatershed source clone at the mpix_meta root ----
# Not a data root: the parity tests import pywatershed from it via
# sys.path, and the Ag tests read its utils/prms_dyn_param.py. Goes
# away when those tests do.
PYWS_ROOT = pl.Path(__file__).parents[2] / "pywatershed"
# A test_data tree populated from the pws-prms-oracle AND derived
# (test_data/generate/oracle_populate.py), plus starfit/
# (test_data/generate/starfit_fetch.py). Default: this repo's
# test_data/; PWS_TEST_DATA overrides (e.g. a fetched copy in CI).
PWS_TEST_DATA = pl.Path(
    os.environ.get("PWS_TEST_DATA", pl.Path(__file__).parents[1] / "test_data")
)
PYWS_INPUT_VOL_NAMES = ("sroff_vol", "ssres_flow_vol", "gwres_flow_vol")

# the pywatershed STARFIT autotest reservoir subset (its comment says
# 117; there are 115): active over the full 1995-2001 reference window.
# Shared by test_starfit_flow_node.py (all 115) and
# test_starfit_daily_parity.py (a leading slice).
# fmt: off
STARFIT_INDS_TEST = [
    0,   1,   2,   3,   4,   5,   6,   8,   9,   10,  11,  12,  13,
    15,  16,  18,  20,  21,  22,  23,  24,  25,  26,  28,  29,  30,
    31,  32,  33,  36,  37,  38,  40,  43,  44,  47,  48,  49,  51,
    52,  53,  55,  56,  59,  62,  63,  64,  65,  67,  68,  69,  70,
    71,  72,  74,  75,  76,  77,  86,  87,  89,  90,  91,  92,  93,
    94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106,
    107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    122, 123, 130, 134, 137, 139, 140, 141, 145, 148, 149, 152, 154,
    155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166,
]
# fmt: on


def pytest_report_header(config):
    state = "found" if PWS_TEST_DATA.is_dir() else "MISSING"
    return f"PWS_TEST_DATA: {PWS_TEST_DATA} ({state})"


# ---- oracle answer pins (tests/prms_oracle.toml) ----
ORACLE_PINS = pl.Path(__file__).parent / "prms_oracle.toml"


def oracle_platform_tag() -> str:
    """The oracle's platform tag (prms_oracle/build.py platform_tag,
    replicated -- the oracle is not a dependency): linux-x86_64 /
    macos-arm64 / windows-x86_64."""
    system = {"linux": "linux", "darwin": "macos", "win32": "windows"}[
        sys.platform
    ]
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "arm64"}.get(
        machine, machine
    )
    return f"{system}-{arch}"


def _marker(path: pl.Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


def check_oracle_pins(root: pl.Path = PWS_TEST_DATA) -> list[str]:
    """Mismatches between the pinned keys and the markers in root: a
    domain's .oracle_ikey, each pinned answer directory's .oracle_okey
    (this platform's pin) and, where the answers are links, the
    oracle's own marker beside them, and its .derived_okey
    (oracle_populate.py derived the variables from THESE answers). A
    pinned directory that does not exist is not a mismatch -- the test
    modules' own skipifs report missing data."""
    pins = tomllib.loads(ORACLE_PINS.read_text())
    tag = oracle_platform_tag()
    problems = []
    for domain, ikey in pins["inputs"].items():
        found = _marker(root / domain / ".oracle_ikey")
        if found is not None and found != ikey:
            problems.append(f"{domain}/.oracle_ikey is {found}, pin {ikey}")
    for dir_name, by_platform in pins["outputs"].items():
        out_dir = root / dir_name
        okey = _marker(out_dir / ".oracle_okey")
        if okey is None:
            continue
        if tag not in by_platform:
            problems.append(f"{dir_name}: no pin for platform {tag}")
            continue
        if okey != by_platform[tag]:
            problems.append(
                f"{dir_name}/.oracle_okey is {okey}, pin {by_platform[tag]}"
            )
        # the answers are symlinks into the oracle and the marker is a
        # COPY: a regenerated oracle changes the answers under the
        # links while the copy keeps the old key
        linked = next(
            (pp for pp in sorted(out_dir.glob("*.nc")) if pp.is_symlink()),
            None,
        )
        if linked is not None:
            oracle_okey = _marker(linked.resolve().parent / ".oracle_okey")
            if oracle_okey is not None and oracle_okey != okey:
                problems.append(
                    f"{dir_name}: the linked answers are now key "
                    f"{oracle_okey}, the copied marker says {okey} "
                    "(rerun oracle_populate.py)"
                )
        derived = _marker(out_dir / ".derived_okey")
        if derived != okey:
            problems.append(
                f"{dir_name}: derived variables are from {derived}, "
                f"the answers are {okey} (rerun oracle_populate.py)"
            )
    return problems


def pytest_sessionstart(session):
    # The hru kernels are compiled threaded; a Model holds numba at
    # control["numba_n_threads"] (default 1). A kernel called without
    # a Model (tests/kernel_units.py) would otherwise run on every
    # core. Not NUMBA_NUM_THREADS=1: that caps numba's maximum, and
    # tests/test_numba_threads.py runs a model on several threads.
    import numba

    numba.set_num_threads(1)
    if not PWS_TEST_DATA.is_dir():
        return
    problems = check_oracle_pins()
    if problems:
        raise pytest.UsageError(
            f"oracle answers in {PWS_TEST_DATA} do not match the pins in "
            f"{ORACLE_PINS}:\n  " + "\n  ".join(problems)
        )


def pyws_domain_files(domain):
    """The files a channel FlowGraph test needs from a GENERATED
    pywatershed domain (for module-level skipifs)."""
    ddir = PWS_TEST_DATA / domain
    gen_dir = ddir / "output"
    return [
        ddir / "parameters_PRMSChannel.nc",
        ddir / "parameters_dis_seg.nc",
        gen_dir / "seg_outflow.nc",
        *[gen_dir / f"{nn}.nc" for nn in PYWS_INPUT_VOL_NAMES],
    ]


def cold_start_state(control, **state):
    """Supplied initial state for a process_dict entry -- or nothing
    when `control` restarts: a model takes its starting state from ONE
    source (supplied state + restart_read raises). Usage:
    `{"class": ..., **cold_start_state(control, flow=...)}`."""
    return {} if control.get("restart_read") is not None else state


@pytest.fixture(scope="session")
def pyws_domain():
    """Factory: pywatershed domain name -> the channel FlowGraph data
    (params/dis datasets, hru->segment weights, seg_outflow answers,
    and a node-volume-input builder), cached per session."""
    cache: dict = {}

    def _get(domain):
        if domain in cache:
            return cache[domain]
        ddir = PWS_TEST_DATA / domain
        gen_dir = ddir / "output"
        channel_params = xr.load_dataset(ddir / "parameters_PRMSChannel.nc")
        dis_seg = xr.load_dataset(ddir / "parameters_dis_seg.nc")

        # 0/1 hru->segment aggregation weights from hru_segment
        hru_segment = channel_params["hru_segment"].values
        n_seg = channel_params.sizes["nsegment"]
        weights = np.zeros((n_seg, hru_segment.shape[0]))
        for ihru in range(hru_segment.shape[0]):
            if hru_segment[ihru] > 0:
                weights[hru_segment[ihru] - 1, ihru] = 1.0

        def node_vol_input(name, node_name, n_extra):
            """hru volumes PRE-AGGREGATED to graph nodes (volumes @
            weights.T -- identical math and float order to a per-step
            Map apply; Map/MapMPI wiring is proven by the PRMSChannel
            submodel tests). Inserted nodes get ZERO columns (no
            lateral inflow)."""
            hru_da = xr.load_dataarray(gen_dir / f"{name}.nc")
            node_vals = hru_da.values @ weights.T  # (time, n_seg)
            if n_extra:
                zeros = np.zeros((node_vals.shape[0], n_extra))
                node_vals = np.concatenate([node_vals, zeros], axis=1)
            return xr.DataArray(
                node_vals,
                dims=("time", "nnodes"),
                coords={"time": hru_da["time"].values},
                name=node_name,
            )

        cache[domain] = {
            "channel_params_ds": channel_params,
            "dis_seg_ds": dis_seg,
            "weights": weights,
            "seg_outflow": xr.load_dataarray(gen_dir / "seg_outflow.nc"),
            "node_vol_input": node_vol_input,
        }
        return cache[domain]

    return _get


@pytest.fixture(scope="session")
def dimensions():
    """Toy model dimensions shared by both regressions."""
    n_years = 1
    n_space = 20
    start_year = 2000
    start_time = np.datetime64(f"{start_year}-01-01")
    end_time = np.datetime64(f"{start_year + n_years}-01-01") - np.timedelta64(
        1, "D"
    )
    time = np.arange(start_time, end_time, dtype="datetime64[D]")
    space = np.arange(n_space)
    return {
        "n_years": n_years,
        "n_space": n_space,
        "n_time": len(time),
        "time": time,
        "space": space,
    }


@pytest.fixture(scope="session")
def make_toy_input():
    """Factory returning the unified toy input Dataset.

    `time`/`space`/`month` are dim-coordinates. Vars: forcing_up, forcing_low
    (time, space; independent forcings for Upper/Lower); param_up_1 (month,
    space, cyclic-monthly); param_up_0, param_low_0, param_shared_name,
    flow_initial, storage_initial (space). Deterministic per seed.
    """

    def _make(dimensions: dict, seed: int = 42) -> xr.Dataset:
        n_years = dimensions["n_years"]
        n_time = dimensions["n_time"]
        n_space = dimensions["n_space"]
        rng = np.random.default_rng(seed)
        sin_data = np.sin(
            np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
        )
        shifts = rng.uniform(10, 100, n_space)
        forcing_up = sin_data[:, np.newaxis] + shifts[np.newaxis, :]
        # Lower's own forcing -- deterministic and unrelated to forcing_up (cos
        # + a fixed per-space shift), so it doesn't perturb the rng draws.
        cos_data = np.cos(
            np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
        )
        shifts_low = np.linspace(1.0, 5.0, n_space)
        forcing_low = cos_data[:, np.newaxis] + shifts_low[np.newaxis, :]
        return xr.Dataset(
            data_vars=dict(
                forcing_up=(["time", "space"], forcing_up),
                forcing_low=(["time", "space"], forcing_low),
                param_up_0=(["space"], rng.uniform(0.1, 1, n_space)),
                param_up_1=(
                    ["month", "space"],
                    rng.uniform(0.1, 1, (12, n_space)),
                ),
                param_low_0=(["space"], rng.uniform(0.17, 0.23, n_space)),
                param_shared_name=(["space"], np.zeros(n_space)),
                flow_initial=(["space"], rng.uniform(100, 1000, n_space)),
                storage_initial=(["space"], rng.uniform(100, 500, n_space)),
            ),
            coords=dict(
                time=("time", dimensions["time"]),
                space=("space", dimensions["space"]),
                month=("month", np.arange(1, 13)),
            ),
        )

    return _make


# ---------------------------------------------------------------------------
# Two-grid toy (Upper on "hru" -> Map -> Lower on "segment"): shared by the
# serial (test_two_grid.py) and MPI Step B (test_two_grid_mpi.py) tests.
# Deterministic per seed, so every MPI rank rebuilds identical data (the
# replication of the serial "segment" grid relies on this).
# ---------------------------------------------------------------------------

# N_HRU divides evenly over 2 and 4 ranks (uneven decomposition is a
# separate concern from the Step B cross-grid comm under test).
N_HRU = 8
N_SEG = 4


@pytest.fixture(scope="session")
def two_grid_weights():
    """(N_SEG, N_HRU): each segment aggregates two HRUs by averaging."""
    ww = np.zeros((N_SEG, N_HRU))
    for seg in range(N_SEG):
        ww[seg, 2 * seg] = 0.5
        ww[seg, 2 * seg + 1] = 0.5
    return ww


def two_grid_mpi_paths(data_dir: pl.Path) -> dict[str, pl.Path]:
    """The files of the two-grid toy under MPI
    (write_two_grid_mpi_files)."""
    return {
        "data_dir": data_dir,
        "input_file": data_dir / "hru_input.nc",
        "low_params": data_dir / "low_params.nc",
        "forcing_low": data_dir / "forcing_low.nc",
        "storage_initial": data_dir / "storage_initial.nc",
        "flow_initial": data_dir / "flow_initial.nc",
        "weights": data_dir / "weights.nc",
        "yaml_file": data_dir / "model.yaml",
    }


def write_two_grid_mpi_files(
    data_dir: pl.Path, toy: dict, two_grid_weights: np.ndarray
) -> dict[str, pl.Path]:
    """Write ALL on-disk inputs of the two-grid toy under MPI -- the
    hru input file, the segment input files, the Map weights, and
    model.yaml (with RELATIVE path strings, exercising config's
    resolve-relative-to-the-yaml rule). No MPI here: ONE process
    calls it (rank 0, or a test that launches the ranks itself)."""
    paths = two_grid_mpi_paths(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # The distributed grid's kitchen-sink input file: forcing +
    # time-varying param as (time|month, hru); static params as
    # (hru,). Segment data and initial state never enter it.
    xr.Dataset(
        data_vars=dict(
            forcing_up=toy["forcing_up"],
            param_up_0=toy["up_params"]["param_up_0"],
            param_up_1=toy["up_params"]["param_up_1"],
            param_shared_name=toy["up_params"]["param_shared_name"],
        ),
    ).to_netcdf(paths["input_file"])
    # Segment inputs + weights, one file each (the bare DataArrays
    # need a name for NetCDF).
    toy["low_params"].to_netcdf(paths["low_params"])
    toy["forcing_low"].rename("forcing_low").to_netcdf(paths["forcing_low"])
    toy["low_storage_initial"].rename("storage_initial").to_netcdf(
        paths["storage_initial"]
    )
    # the distributed grid's initial state: full extent, its own
    # file, each rank takes its block
    toy["up_flow_initial"].rename("flow_initial").to_netcdf(
        paths["flow_initial"]
    )
    xr.DataArray(
        two_grid_weights, dims=["segment", "hru"], name="weights"
    ).to_netcdf(paths["weights"])
    # The fully-serialized configuration (relative path strings).
    cfg = {
        "control": {
            "input_file": "hru_input.nc",
            "output_parallel_netcdf": "hru_output_yaml.nc",
            "output_serial_zarr": "segment_output_yaml.zarr",
            "output_var_names": ["flow", "storage"],
            "time_chunk_size": 10,
            "mpi_grid": "hru",
        },
        "processes": {
            "upper": {
                "class": "Upper",
                "discretization": "hru",
                "flow": "flow_initial.nc",
            },
            "lower": {
                "class": "Lower",
                "discretization": "segment",
                "parameters": "low_params.nc",
                "forcing_low": "forcing_low.nc",
                "storage": "storage_initial.nc",
            },
        },
        "maps": {
            "hru_to_seg": {
                "class": "MapMPI",
                "weights": "weights.nc",
                "grid": {"hru": "segment"},
                "variable": {"flow": "flow"},
            }
        },
    }
    # sort_keys=False is LOAD-BEARING: the processes mapping order IS
    # the execution schedule, and safe_dump alphabetizes by default
    # (which would put lower before upper -> the one-pass order
    # validation rightly rejects it).
    with open(paths["yaml_file"], "w") as ff:
        yaml.safe_dump(cfg, ff, sort_keys=False)
    return paths


@pytest.fixture(scope="session")
def make_two_grid_toy():
    """Factory for the two-grid toy inputs: Upper's on dim "hru" (N_HRU),
    Lower's on "segment" (N_SEG) -- each grid's key is its real dim;
    separate datasets. `flow` is NOT given to Lower (the Map feeds it).
    `hru` and `time` are real dim-coordinates (mpixarray's `parallelize`/
    `as_iterable` need them on the distributed grid)."""

    def _make(dimensions: dict, seed: int = 11) -> dict:
        n_time = dimensions["n_time"]
        rng = np.random.default_rng(seed)
        sin = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / n_time))
        pu1 = rng.uniform(0.1, 1, (12, N_HRU))
        hru_coord = np.arange(N_HRU)

        # -- grid "hru" (Upper) --
        up_params = xr.Dataset(
            dict(
                param_up_0=(["hru"], rng.uniform(0.1, 1, N_HRU)),
                param_up_1=(["month", "hru"], pu1),
                param_shared_name=(["hru"], np.zeros(N_HRU)),
            ),
            coords=dict(
                month=("month", np.arange(1, 13)),
                hru=("hru", hru_coord),
            ),
        )
        forcing_up = xr.DataArray(
            sin[:, None] + rng.uniform(10, 100, N_HRU)[None, :],
            dims=["time", "hru"],
            coords={"time": dimensions["time"], "hru": hru_coord},
        )
        up_flow_initial = xr.DataArray(
            rng.uniform(100, 1000, N_HRU),
            dims=["hru"],
            coords={"hru": hru_coord},
        )

        # -- grid "segment" (Lower) --
        low_params = xr.Dataset(
            dict(
                param_low_0=(["segment"], rng.uniform(0.17, 0.23, N_SEG)),
                param_shared_name=(["segment"], np.zeros(N_SEG)),
            )
        )
        forcing_low = xr.DataArray(
            rng.uniform(1, 3, (n_time, N_SEG)),
            dims=["time", "segment"],
            coords={"time": dimensions["time"]},
        )
        low_storage_initial = xr.DataArray(
            rng.uniform(100, 500, N_SEG), dims=["segment"]
        )
        return dict(
            up_params=up_params,
            forcing_up=forcing_up,
            up_flow_initial=up_flow_initial,
            low_params=low_params,
            forcing_low=forcing_low,
            low_storage_initial=low_storage_initial,
        )

    return _make


@pytest.fixture(scope="session")
def compute_two_grid_answers():
    """Factory for the two-grid ground truth: Upper.flow (hru) ->
    W @ flow -> Lower.storage (segment)."""

    def _compute(toy: dict, weights, dimensions: dict) -> dict:
        n_time = dimensions["n_time"]
        time = np.asarray(dimensions["time"])
        months = time.astype("datetime64[M]").astype(int) % 12
        f0 = toy["forcing_up"].values
        pu1 = toy["up_params"]["param_up_1"].values
        flow_init = toy["up_flow_initial"].values
        fl = toy["forcing_low"].values
        stor_init = toy["low_storage_initial"].values

        flow = np.zeros((n_time, N_HRU))
        flow_prev = np.zeros((n_time, N_HRU))
        for tt in range(n_time):
            flow_prev[tt] = flow_init if tt == 0 else flow[tt - 1]
            flow[tt] = flow_prev[tt] * 0.95 + f0[tt] * pu1[months[tt]]

        storage = np.zeros((n_time, N_SEG))
        storage_prev = np.zeros((n_time, N_SEG))
        for tt in range(n_time):
            flow_seg = weights @ flow[tt]
            storage_prev[tt] = stor_init if tt == 0 else storage[tt - 1]
            storage[tt] = (
                storage_prev[tt] * 0.95 + flow_seg * 0.12 + fl[tt] * 0.10
            )
        return dict(flow=flow, storage=storage)

    return _compute


@pytest.fixture(scope="session")
def compute_answers():
    """Factory for the vectorized ground-truth Upper/Lower solution."""

    def _compute(
        forcing_up,
        flow_initial,
        storage_initial,
        n_time,
        param_up_1,
        time,
        forcing_low,
    ) -> dict:
        forcing_up = np.asarray(forcing_up)
        flow_initial = np.asarray(flow_initial)
        storage_initial = np.asarray(storage_initial)
        param_up_1 = np.asarray(param_up_1)
        forcing_low = np.asarray(forcing_low)
        n_space = forcing_up.shape[1]

        # day -> month index (0-11), matching Time.month - 1
        months = np.asarray(time).astype("datetime64[M]").astype(int) % 12

        expected_flow = np.zeros((n_time, n_space))
        expected_flow_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_flow_prev[tt, :] = (
                flow_initial if tt == 0 else expected_flow[tt - 1, :]
            )
            expected_flow[tt, :] = (
                expected_flow_prev[tt, :] * 0.95
                + forcing_up[tt, :] * param_up_1[months[tt], :]
            )

        expected_storage = np.zeros((n_time, n_space))
        expected_storage_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_storage_prev[tt, :] = (
                storage_initial if tt == 0 else expected_storage[tt - 1, :]
            )
            expected_storage[tt, :] = (
                expected_storage_prev[tt, :] * 0.95
                + expected_flow[tt, :] * 0.12
                + forcing_low[tt, :] * 0.10
            )

        return {
            "expected_flow": expected_flow,
            "expected_flow_prev": expected_flow_prev,
            "expected_storage": expected_storage,
            "expected_storage_prev": expected_storage_prev,
        }

    return _compute
