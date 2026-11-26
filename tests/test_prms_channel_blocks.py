"""The pieces of PRMSChannelMPI (hydrology/prms_channel_mpi.py) that
need no MPI: the cut of the network into blocks, the block kernel,
and ModelMPI's checks of a divided process.

The reference is PRMSChannel in a serial Model of the channel alone,
on the real segment parameters of drb_2yr (456 segments) and ucb_2yr
(1,942), fed three made-up lateral volumes that change every day. The
blocks are driven here as the ranks would drive them: each block of
subtrees, its outlets' hourly outflow handed to the main stem, then
the main stem. Every variable of every segment must equal the
reference on every day as 64-bit patterns, for 1 to 33 blocks.

A THIRD NETWORK, "ucb_repointed": ucb with 600 segments made to drain
one segment further down, so that hundreds of segments have three or
more segments draining into them. Only there does the ORDER of the
adds into a segment reach the last bit, and
test_the_order_of_the_given_rows_matters shows that this network sees
a wrong order: the main stem with its given rows first in
segment_order, in reverse, does not equal the reference. (The given
rows first in their own order is not enough: it changes no sum on
this network.)

Under real MPI the class is tested in
tests/test_prms_channel_divided_mpi.py.
"""

import dataclasses
import types
from typing import Any

import numpy as np
import pytest
import xarray as xr

from conftest import PWS_TEST_DATA
from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_channel import PRMSChannel
from pws_phoenix.hydrology.prms_channel_mpi import (
    _INPUTS,
    _KERNEL_ARGS,
    _WRITTEN,
    PRMSChannelMPI,
    _Block,
    make_block,
    sent_columns,
    subtree_blocks,
)
from pws_phoenix.model import Model, ModelMPI

DT = np.float64(86400.0)
N_DAYS = 4
N_REPOINTED = 600
NETWORKS = ("drb_2yr", "ucb_2yr", "ucb_repointed")
BLOCK_COUNTS = (1, 2, 3, 4, 8, 16, 33)

_needed = [
    PWS_TEST_DATA / domain / name
    for domain in ("drb_2yr", "ucb_2yr")
    for name in ("parameters_dis_seg.nc", "parameters_PRMSChannel.nc")
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason="segment parameter files missing: " + ", ".join(_missing[:3]),
)


def _same_bits(aa: np.ndarray, bb: np.ndarray) -> bool:
    """Equal as 64-bit patterns: nan equals nan, -0.0 is not 0.0."""
    return aa.shape == bb.shape and bool(
        (aa.view(np.int64) == bb.view(np.int64)).all()
    )


def _repointed(tosegment: np.ndarray) -> np.ndarray:
    """`tosegment` (1-based, 0 = outlet) with N_REPOINTED segments
    made to drain into the segment below the one they drained into.
    Still without a cycle: each still drains downstream."""
    below = np.where(tosegment > 0, tosegment[tosegment - 1], 0)
    can_move = np.flatnonzero(below > 0)
    rng = np.random.default_rng(7)
    moved = rng.choice(can_move, size=N_REPOINTED, replace=False)
    out = tosegment.copy()
    out[moved] = below[moved]
    return out


def _channel_model(network: str, cls: type) -> tuple[Model, np.ndarray]:
    """A serial Model of the channel alone on `network`, and the
    (N_DAYS, 3, nsegment) lateral volumes it is fed."""
    domain = PWS_TEST_DATA / network.replace("ucb_repointed", "ucb_2yr")
    dis = xr.load_dataset(domain / "parameters_dis_seg.nc")
    if network == "ucb_repointed":
        dis["tosegment"] = (
            dis["tosegment"].dims,
            _repointed(dis["tosegment"].values),
        )
    n_seg = dis.sizes["nsegment"]
    rng = np.random.default_rng(11)
    # cubic feet per day; no two segments or days alike
    volumes = rng.uniform(1.0e3, 1.0e6, size=(N_DAYS, 3, n_seg))
    time = np.datetime64("1979-01-01") + np.arange(N_DAYS)
    process_dict = {
        "prms_channel": {
            "class": cls,
            "discretization": "nsegment",
            "parameters": xr.load_dataset(
                domain / "parameters_PRMSChannel.nc"
            ),
        }
        | {
            name: xr.DataArray(
                volumes[:, ii, :],
                dims=("time", "nsegment"),
                coords={"time": time.astype("datetime64[ns]")},
            )
            for ii, name in enumerate(_INPUTS)
        }
    }
    model = Model(
        process_dict,
        {"collect_var_names": list(_WRITTEN)},
        discretizations={
            "nsegment": Discretization(
                ["nsegment"],
                parameters=dis,
                topo_order={"segment_order": "tosegment"},
            )
        },
    )
    return model, volumes


def _run_model(network: str, cls: type) -> dict[str, np.ndarray]:
    """name -> (N_DAYS, nsegment), every written array of every day."""
    model, _volumes = _channel_model(network, cls)
    with model:
        model.run(DT, np.int32(N_DAYS))
        assert model.results is not None
        return {name: model.results.values[name].copy() for name in _WRITTEN}


@pytest.fixture(scope="module", params=NETWORKS)
def network(request) -> dict[str, Any]:
    """One network: the reference of every day, and the kernel's
    arrays as they are before the first step."""
    name = request.param
    model, volumes = _channel_model(name, PRMSChannel)
    with model:
        refs = model.model_dict["prms_channel"]._refs
        start = {nn: refs[nn].copy() for nn in _KERNEL_ARGS}
    return {
        "name": name,
        "reference": _run_model(name, PRMSChannel),
        "start": start,
        "volumes": volumes,
    }


def _cut(start: dict[str, np.ndarray], n_blocks: int) -> list[_Block]:
    """The n_blocks blocks of subtrees, then the main stem."""
    block_of = subtree_blocks(
        start["tosegment0"], start["segment_order"], n_blocks
    )
    column = sent_columns(start["tosegment0"], block_of, n_blocks)
    return [
        make_block(start, block_of, column, n_blocks, block)
        for block in range(n_blocks + 1)
    ]


def _run_blocks(
    blocks: list[_Block], volumes: np.ndarray
) -> dict[str, np.ndarray]:
    """Drive the blocks as PRMSChannelMPI drives the ranks; name ->
    (N_DAYS, nsegment) put back in the whole network's order."""
    main = blocks[-1]
    n_seg = volumes.shape[2]
    found = {name: np.full((N_DAYS, n_seg), np.nan) for name in _WRITTEN}
    for day in range(N_DAYS):
        inputs = {name: volumes[day, ii, :] for ii, name in enumerate(_INPUTS)}
        stop = 0
        for block in blocks:
            # PRMSChannelMPI.advance
            block.arrays["inflow_ts_prev"][:] = block.arrays["seg_inflow"]
            block.take_inputs(inputs)
            block.run(DT)
            # the Gatherv: the blocks' pieces side by side, in order
            start = stop
            stop = start + block.outflow_sent.shape[0]
            main.outflow_given[start:stop] = block.outflow_sent
        assert stop == main.outflow_given.shape[0]
        for block in blocks:
            own = block.rows[: block.n_own]
            for name in _WRITTEN:
                found[name][day, own] = block.arrays[name][: block.n_own]
    return found


@pytest.mark.parametrize("n_blocks", BLOCK_COUNTS)
def test_the_cut(network, n_blocks):
    """Every segment is in one block; a subtree is whole (what drains
    out of a block of subtrees drains into the main stem); nothing
    drains out of the main stem; the loads are near even."""
    start = network["start"]
    to_seg = start["tosegment0"]
    n_seg = to_seg.shape[0]
    block_of = subtree_blocks(to_seg, start["segment_order"], n_blocks)
    assert block_of.min() >= 0 and block_of.max() <= n_blocks
    has_down = to_seg >= 0
    down_block = block_of[to_seg[has_down]]
    here = block_of[has_down]
    assert ((down_block == here) | (down_block == n_blocks)).all()

    column = sent_columns(to_seg, block_of, n_blocks)
    sent = np.flatnonzero(column >= 0)
    # the outlets sent to the main stem: numbered 0 .., by block
    assert sorted(column[sent]) == list(range(sent.shape[0]))
    by_column = sent[np.argsort(column[sent])]
    assert (np.diff(block_of[by_column]) >= 0).all()
    assert (block_of[to_seg[sent]] == n_blocks).all()

    sizes = np.bincount(block_of, minlength=n_blocks + 1)
    assert sizes.sum() == n_seg
    if n_blocks == 1:
        assert sizes[1] == 0  # no main stem: one block holds all
    else:
        # a subtree is no larger than the cut, so the largest block is
        # within one cut of the mean
        assert sizes[:n_blocks].max() <= n_seg / n_blocks + n_seg // n_blocks


@pytest.mark.parametrize("n_blocks", BLOCK_COUNTS)
def test_blocks_equal_the_whole_network(network, n_blocks):
    blocks = _cut(network["start"], n_blocks)
    found = _run_blocks(blocks, network["volumes"])
    for name in _WRITTEN:
        assert _same_bits(found[name], network["reference"][name]), name
    # the reference is a routed network, not zeros
    assert np.unique(network["reference"]["seg_outflow"][-1]).shape[0] > 100


def test_the_order_of_the_given_rows_matters(network):
    """The must-differ case: on the repointed network a main stem
    with its given rows FIRST in segment_order, IN REVERSE, does not
    give the reference. Every given row still comes before the
    segment it drains into, so only the order of the adds changed.
    On the real networks nothing has to differ (ucb has at most 4
    main stem segments where an order could show)."""
    n_blocks = 8
    blocks = _cut(network["start"], n_blocks)
    main = blocks[-1]
    order = main.arrays["segment_order"]
    is_given = order >= main.n_own
    moved = np.concatenate([order[is_given][::-1], order[~is_given]])
    blocks[-1] = dataclasses.replace(
        main, arrays=main.arrays | {"segment_order": moved}
    )
    found = _run_blocks(blocks, network["volumes"])
    differing = [
        name
        for name in _WRITTEN
        if not _same_bits(found[name], network["reference"][name])
    ]
    if network["name"] == "ucb_repointed":
        to_seg = network["start"]["tosegment0"]
        n_into = np.bincount(to_seg[to_seg >= 0], minlength=to_seg.shape[0])
        assert (n_into >= 3).sum() > 100
        assert "seg_outflow" in differing
    else:
        assert set(differing) <= set(_WRITTEN)


def test_in_a_serial_model_it_is_prms_channel(network):
    """Without set_decomposition() the class runs PRMSChannel's
    advance and calculate."""
    found = _run_model(network["name"], PRMSChannelMPI)
    for name in _WRITTEN:
        assert _same_bits(found[name], network["reference"][name]), name


def test_kernel_args_are_the_package_kernels():
    """The block kernel takes PRMSChannel._calculate's arguments, in
    its order, then four more; the written ones are the class's
    variables."""
    package = PRMSChannel._calculate.py_func.__code__
    assert package.co_varnames[: len(_KERNEL_ARGS)] == _KERNEL_ARGS
    assert package.co_varnames[len(_KERNEL_ARGS)] == "s_per_time"
    assert sorted(_WRITTEN) == sorted(PRMSChannel.get_var_names())
    assert _INPUTS == PRMSChannel.get_inputs()


# -- ModelMPI._decompose_processes, on stand-ins (no MPI) ---------------


class _StandIn:
    """What _decompose_processes reads of a process."""

    def __init__(
        self,
        variables: tuple[str, ...] = (),
        inputs: tuple[str, ...] = (),
        mutable_inputs: tuple[str, ...] = (),
        budgets: bool = False,
    ) -> None:
        self._variables = variables
        self._inputs = inputs
        self._mutable_inputs = mutable_inputs
        self._budgets = {"mass": None} if budgets else {}

    def get_var_names(self, include_budget: bool = False) -> tuple[str, ...]:
        return self._variables

    def get_inputs(self) -> tuple[str, ...]:
        return self._inputs

    def get_mutable_inputs(self) -> tuple[str, ...]:
        return self._mutable_inputs

    def get_restart_state(self) -> dict[str, np.ndarray]:
        return {}


class _DividedStandIn(_StandIn):
    def set_decomposition(self, comm: Any, names: tuple[str, ...]) -> None:
        self.given = (comm, names)


def _stand_in_model(**changes: Any) -> Any:
    """The attributes _decompose_processes reads of a ModelMPI: an
    hru process, a divided channel fed by two MapMPI, a process after
    it on the segment grid, and a Map from the segment grid on."""
    channel = _DividedStandIn(
        variables=("seg_inflow", "seg_outflow", "outflow_ts", "seg_width"),
        inputs=("seg_sroff_vol", "seg_gwres_flow_vol"),
    )

    def a_map(source_grid, target_grid, source_var, target_var):
        return types.SimpleNamespace(
            source_grid=source_grid,
            target_grid=target_grid,
            source_var=source_var,
            target_var=target_var,
        )

    model = types.SimpleNamespace(
        _mpi_grid="nhru",
        _comm="the comm",
        _serial_grids_all_ranks=False,
        model_dict={
            # on the distributed grid: never divided, whatever it has
            "runoff": _DividedStandIn(variables=("sroff_vol",)),
            "channel": channel,
            "temperature": _StandIn(
                variables=("seg_tave",), inputs=("seg_outflow",)
            ),
        },
        _proc_grid={
            "runoff": "nhru",
            "channel": "nsegment",
            "temperature": "nsegment",
        },
        maps={
            "sroff": a_map("nhru", "nsegment", "sroff_vol", "seg_sroff_vol"),
            "gw": a_map(
                "nhru", "nsegment", "gwres_flow_vol", "seg_gwres_flow_vol"
            ),
            "width": a_map("nsegment", "nlake", "seg_width", "lake_width"),
        },
    )
    for name, value in changes.items():
        setattr(model, name, value)
    return model


def test_decompose_names_what_rank_0_reads():
    """Requested output, another process's input on the grid and a
    Map's source, in the order of the class's declarations; a
    variable nothing reads (outflow_ts) is not sent at each step."""
    model = _stand_in_model()
    ModelMPI._decompose_processes(model, ["seg_inflow", "seg_tave"])
    assert list(model._divided) == ["channel"]
    assert model.model_dict["channel"].given == (
        "the comm",
        ("seg_inflow", "seg_outflow", "seg_width"),
    )
    assert not hasattr(model.model_dict["runoff"], "given")


def test_decompose_raises_with_serial_grids_all_ranks():
    model = _stand_in_model(_serial_grids_all_ranks=True)
    with pytest.raises(NotImplementedError, match="serial_grids_all_ranks"):
        ModelMPI._decompose_processes(model, [])


def test_decompose_raises_on_an_input_only_rank_0_holds():
    model = _stand_in_model()
    del model.maps["gw"]
    with pytest.raises(ValueError, match="seg_gwres_flow_vol"):
        ModelMPI._decompose_processes(model, [])


def test_decompose_raises_on_a_variable_edited_in_place():
    model = _stand_in_model()
    model.model_dict["temperature"] = _StandIn(mutable_inputs=("seg_outflow",))
    with pytest.raises(NotImplementedError, match="seg_outflow"):
        ModelMPI._decompose_processes(model, [])


def test_decompose_raises_on_a_budget():
    model = _stand_in_model()
    model.model_dict["channel"]._budgets = {"mass": None}
    with pytest.raises(NotImplementedError, match="budget"):
        ModelMPI._decompose_processes(model, [])
