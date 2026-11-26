"""
hydrology/prms_channel_mpi.py
=============================
PRMSChannelMPI: PRMSChannel with its segments divided over the MPI
ranks. The same fields and the same answers, bit for bit, as
PRMSChannel on the whole network. The measurements and the reasons
are in ``dag_research.md`` (repo root).

PROPOSAL (Nov 2026, branch dag_research): written by Claude while JLM
was away; every design choice here is listed in ``dag_research.md``
section 13 for JLM to accept or change.

The segment grid stays what it is under ModelMPI: a serial grid, held
whole by every rank and fed by MapMPI. This class is to PRMSChannel
what MapMPI is to Map: the caller names it in ``process_dict``,
ModelMPI calls ``set_decomposition`` once at the build, and the class
makes its own messages (mpi4py, imported only then). Without that
call (a serial Model) it IS PRMSChannel.

THE CUT (Mizukami et al. 2021, doi 10.1029/2020MS002434), for N ranks:
the main stem is every segment with more than ``nsegment // N``
segments upstream of and including it. The subtrees that drain into
it, and the basins without a main stem, are given largest first to
the least loaded of N blocks. Rank r routes block r; rank 0 also
routes the main stem. Within a block the segments keep their order in
``segment_order``.

WHAT A RANK HOLDS: its block's rows of every array the kernel takes,
cut out once at ``set_decomposition`` and numbered again
(``_Block.arrays``; rank 0 holds two blocks, its own and the main
stem). These are the working arrays: the measured gain depends on
them being small (18.3 ns per segment-hour on the whole CONUS
network, about 12 in a block of 900). They are a second set of arrays
for the same names, of block size, allocated once.

ONE STEP (``calculate``):
1. Every rank takes its rows of the three Map-fed inputs from the
   grid dataset (every rank has them: a MapMPI ends in an Allreduce)
   and runs the kernel on its block. The kernel records the 24 hourly
   outflows of each subtree outlet that drains into the main stem.
2. One Gatherv: those hourly outflows land side by side in rank 0's
   ``outflow_given``.
3. Rank 0 runs the kernel on the main stem.
4. One Gatherv of the variables that something on rank 0 reads every
   step (``names_each_step``: requested output, another process's
   input, a Map's source); rank 0 puts them into the grid dataset in
   the grid's order.

So on rank 0 the grid dataset holds the variables of
``names_each_step`` at every step, and every variable after
``gather_all()`` (ModelMPI calls it at the end of ``run()`` and in
``write_restart()``). On the other ranks the dataset keeps the values
of the build, as for every serial-grid process.

WHY THE ANSWERS ARE THE SAME. A subtree needs nothing from outside
itself. The main stem block holds each subtree outlet that drains
into it as ONE MORE ROW, at the outlet's place in ``segment_order``.
That row is not a segment of the main stem: each hour it is given the
outlet's outflow as its lateral inflow, and its ``tsi = -1``,
``ts = 1`` make the kernel's own lines pass it on unchanged
(``outflow_ts = inflow_ts``). So the adds into a main stem segment
are the same values in the same order as on the whole network, which
matters to the last bit where three or more segments meet
(``tests/test_prms_channel_blocks.py`` has a network where a wrong
order shows).

THE KERNEL (``_calculate_block``) is ``PRMSChannel._calculate`` with
three short loops added OUTSIDE the loop over segments, marked ADDED.
Two kernels with one body: a change to one must be made to the other,
and ``tests/test_prms_channel_blocks.py`` fails if they differ in any
bit. (One kernel with four more arguments is the alternative; see
``dag_research.md``.)

NOT handled, each raises at the build (ModelMPI): an input that only
rank 0 computes; ``control["serial_grids_all_ranks"]``; another
process that edits one of this class's variables in place.
"""

import dataclasses
import heapq
from collections.abc import Callable
from typing import Any

import numba
import numpy as np

from pws_phoenix import NUMBA_CACHE
from pws_phoenix.clock import Time
from pws_phoenix.hydrology.prms_channel import (
    _ROUTING_STEP_HOURS,
    _ZERO_CFS,
    _ZERO_CUBIC_FEET,
    PRMSChannel,
)

# the array arguments of PRMSChannel._calculate, in its order
_KERNEL_ARGS = (
    "seg_lateral_inflow",
    "seg_upstream_inflow",
    "seg_inflow",
    "seg_outflow",
    "seg_stor_change",
    "channel_outflow_vol",
    "inflow_ts",
    "outflow_ts",
    "inflow_ts_prev",
    "seg_current_sum",
    "seg_sroff_vol",
    "seg_ssres_flow_vol",
    "seg_gwres_flow_vol",
    "segment_order",
    "tosegment0",
    "ts",
    "tsi",
    "c0",
    "c1",
    "c2",
)
# the kernel writes the first ten: every variable of the class
_WRITTEN = _KERNEL_ARGS[:10]
_INPUTS = ("seg_sroff_vol", "seg_ssres_flow_vol", "seg_gwres_flow_vol")


def subtree_blocks(
    tosegment0: np.ndarray, segment_order: np.ndarray, n_blocks: int
) -> np.ndarray:
    """The block of each segment: 0 .. n_blocks - 1, and n_blocks for
    the main stem (THE CUT in the module docstring). ``tosegment0`` is
    0-based, negative = outlet; ``segment_order`` is upstream to
    downstream. Build-time python loops over the segments."""
    n_seg = tosegment0.shape[0]
    # the count of segments upstream of and including each segment
    size = np.ones(n_seg, dtype=np.int64)
    for jseg in segment_order:
        if tosegment0[jseg] >= 0:
            size[tosegment0[jseg]] += size[jseg]
    main = size > n_seg // n_blocks
    drains_to_main = np.where(tosegment0 >= 0, main[tosegment0], True)
    # a subtree's most downstream segment
    heads = np.flatnonzero(~main & drains_to_main)

    block_of = np.full(n_seg, n_blocks, dtype=np.int64)
    loads = [(0, block) for block in range(n_blocks)]
    largest_first = heads[np.lexsort((heads, -size[heads]))]
    for head in largest_first:
        load, block = loads[0]
        heapq.heapreplace(loads, (load + int(size[head]), block))
        block_of[head] = block
    for jseg in segment_order[::-1]:  # downstream to upstream
        if not main[jseg] and block_of[jseg] == n_blocks:
            block_of[jseg] = block_of[tosegment0[jseg]]
    return block_of


def sent_columns(
    tosegment0: np.ndarray, block_of: np.ndarray, n_blocks: int
) -> np.ndarray:
    """Per segment: the row of the main stem's ``outflow_given`` that
    holds its hourly outflow; -1 for a segment that is not a subtree
    outlet draining into the main stem. The rows are ordered by block,
    then by segment, so one block's rows are contiguous and the
    blocks' pieces of one Gatherv land side by side."""
    drains_to_main = (tosegment0 >= 0) & (block_of[tosegment0] == n_blocks)
    sent = np.flatnonzero((block_of < n_blocks) & drains_to_main)
    sent = sent[np.argsort(block_of[sent], kind="stable")]
    column = np.full(tosegment0.shape[0], -1, dtype=np.int64)
    column[sent] = np.arange(sent.shape[0])
    return column


@numba.njit(cache=NUMBA_CACHE)
def _calculate_block(
    seg_lateral_inflow: np.ndarray,
    seg_upstream_inflow: np.ndarray,
    seg_inflow: np.ndarray,
    seg_outflow: np.ndarray,
    seg_stor_change: np.ndarray,
    channel_outflow_vol: np.ndarray,
    inflow_ts: np.ndarray,
    outflow_ts: np.ndarray,
    inflow_ts_prev: np.ndarray,
    seg_current_sum: np.ndarray,
    seg_sroff_vol: np.ndarray,
    seg_ssres_flow_vol: np.ndarray,
    seg_gwres_flow_vol: np.ndarray,
    segment_order: np.ndarray,
    tosegment0: np.ndarray,
    ts: np.ndarray,
    tsi: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    s_per_time: np.float64,
    given_rows: np.ndarray,
    outflow_given: np.ndarray,
    sent_rows: np.ndarray,
    outflow_sent: np.ndarray,
) -> None:
    """PRMSChannel._calculate on one block. The lines are that
    kernel's, but for the three loops marked ADDED (THE KERNEL in the
    module docstring). Keep the two in step."""
    n_seg = seg_outflow.shape[0]

    for jj in range(n_seg):
        seg_lateral_inflow[jj] = (
            seg_sroff_vol[jj] + seg_ssres_flow_vol[jj] + seg_gwres_flow_vol[jj]
        ) / s_per_time

    seg_inflow[:] = _ZERO_CFS
    seg_outflow[:] = _ZERO_CFS
    inflow_ts[:] = _ZERO_CFS
    seg_current_sum[:] = _ZERO_CFS

    for ihr in range(24):
        seg_upstream_inflow[:] = _ZERO_CFS

        # ADDED (the main stem only): a row that stands for a subtree
        # outlet routed by another block is given that outlet's
        # outflow of this hour as its lateral inflow. Its tsi = -1 and
        # ts = 1 make the lines below pass it on unchanged.
        for kk in range(given_rows.shape[0]):
            seg_lateral_inflow[given_rows[kk]] = outflow_given[kk, ihr]

        for jseg in segment_order:
            seg_current_inflow = (
                seg_lateral_inflow[jseg] + seg_upstream_inflow[jseg]
            )
            seg_inflow[jseg] += seg_current_inflow
            inflow_ts[jseg] += seg_current_inflow
            seg_current_sum[jseg] += seg_upstream_inflow[jseg]

            remainder = (ihr + 1) % tsi[jseg]
            if remainder == 0:
                inflow_ts[jseg] /= ts[jseg] / _ROUTING_STEP_HOURS
                if tsi[jseg] > 0:
                    outflow_ts[jseg] = (
                        inflow_ts[jseg] * c0[jseg]
                        + inflow_ts_prev[jseg] * c1[jseg]
                        + outflow_ts[jseg] * c2[jseg]
                    )
                else:
                    outflow_ts[jseg] = inflow_ts[jseg]
                inflow_ts_prev[jseg] = inflow_ts[jseg]
                inflow_ts[jseg] = _ZERO_CFS

            seg_outflow[jseg] += outflow_ts[jseg]
            to_seg = tosegment0[jseg]
            if to_seg >= 0:
                seg_upstream_inflow[to_seg] += outflow_ts[jseg]

        # ADDED (a block of subtrees only): this hour's outflow of
        # each subtree outlet that drains into the main stem
        for kk in range(sent_rows.shape[0]):
            outflow_sent[kk, ihr] = outflow_ts[sent_rows[kk]]

    for jj in range(n_seg):
        seg_outflow[jj] /= 24.0
        seg_inflow[jj] /= 24.0
        seg_upstream_inflow[jj] = seg_current_sum[jj] / 24.0
        seg_stor_change[jj] = (seg_inflow[jj] - seg_outflow[jj]) * s_per_time
        if tosegment0[jj] < 0:
            channel_outflow_vol[jj] = seg_outflow[jj] * s_per_time
        else:
            channel_outflow_vol[jj] = _ZERO_CUBIC_FEET

    # ADDED (a block of subtrees only): within its block a subtree
    # outlet that drains into the main stem has no downstream segment
    # (tosegment0 < 0), but it is not an outlet of the domain
    for kk in range(sent_rows.shape[0]):
        channel_outflow_vol[sent_rows[kk]] = _ZERO_CUBIC_FEET


@dataclasses.dataclass
class _Block:
    """One block's rows of the kernel's arrays, numbered again."""

    # each row's segment in the whole network: the block's own
    # segments first, then (the main stem only) the given outlets
    rows: np.ndarray
    n_own: int
    arrays: dict[str, np.ndarray]
    given_rows: np.ndarray  # rows of this block that are given outlets
    outflow_given: np.ndarray  # (given outlets, 24)
    sent_rows: np.ndarray  # rows of this block that are sent outlets
    outflow_sent: np.ndarray  # (sent outlets, 24)

    def __post_init__(self) -> None:
        # the kernel's array arguments, looked up once
        self._args = tuple(self.arrays[name] for name in _KERNEL_ARGS)

    def take_inputs(self, refs: dict[str, np.ndarray]) -> None:
        """This block's rows of the Map-fed inputs, from the grid
        dataset's arrays. mode="clip": numpy copies through a second
        buffer under the default mode; the rows are valid by
        construction."""
        for name in _INPUTS:
            np.take(refs[name], self.rows, out=self.arrays[name], mode="clip")

    def run(self, s_per_time: np.float64) -> None:
        # (typed loosely: mypy cannot count the arguments of a numba
        # dispatcher behind a starred tuple)
        kernel: Callable[..., None] = _calculate_block
        kernel(
            *self._args,
            s_per_time,
            self.given_rows,
            self.outflow_given,
            self.sent_rows,
            self.outflow_sent,
        )


def make_block(
    refs: dict[str, np.ndarray],
    block_of: np.ndarray,
    column: np.ndarray,
    n_blocks: int,
    block: int,
) -> _Block:
    """Block ``block`` of the cut, its arrays COPIED from ``refs`` (the
    whole network's arrays under the kernel's argument names). Block
    ``n_blocks`` is the main stem, with one more row for each outlet
    that ``column`` gives it."""
    is_main = block == n_blocks
    own = np.flatnonzero(block_of == block)
    n_own = own.shape[0]
    given = np.zeros(0, dtype=np.int64)
    if is_main:
        outlets = np.flatnonzero(column >= 0)
        given = outlets[np.argsort(column[outlets])]  # row kk = column kk
    rows = np.concatenate([own, given])
    new_index = np.full(block_of.shape[0], -1, dtype=np.int64)
    new_index[rows] = np.arange(rows.shape[0])

    # fancy indexing: each array is a copy of the block's rows
    arrays = {name: refs[name][rows] for name in _KERNEL_ARGS}
    to_seg = refs["tosegment0"][rows]
    # a segment that drains out of the block is an outlet within it
    arrays["tosegment0"] = np.where(to_seg >= 0, new_index[to_seg], -1)
    order = refs["segment_order"]
    arrays["segment_order"] = new_index[order[new_index[order] >= 0]]
    # a given row passes its lateral inflow on unchanged
    arrays["ts"][n_own:] = 1.0
    arrays["tsi"][n_own:] = -1

    n_given = given.shape[0]
    sent = np.zeros(0, dtype=np.int64)
    if not is_main:
        # ascending segment = ascending column within one block
        sent = np.flatnonzero(column[rows] >= 0)
    return _Block(
        rows=rows,
        n_own=n_own,
        arrays=arrays,
        given_rows=n_own + np.arange(n_given),
        outflow_given=np.zeros((n_given, 24)),
        sent_rows=sent,
        outflow_sent=np.zeros((sent.shape[0], 24)),
    )


@dataclasses.dataclass
class _Gather:
    """The buffers of one Gatherv of some variables to rank 0."""

    names: tuple[str, ...]
    send: np.ndarray  # (this rank's block rows, names)
    # rank 0 only: (nsegment, names), the blocks' rows in rank order
    # and then the main stem's; and the Gatherv's receive argument
    recv: np.ndarray | None
    recv_spec: list[Any] | None


class PRMSChannelMPI(PRMSChannel):
    """PRMSChannel with its segments divided over the MPI ranks (see
    the module docstring). In a serial Model it is PRMSChannel."""

    _comm: Any = None

    def set_decomposition(
        self, comm: Any, names_each_step: tuple[str, ...]
    ) -> None:
        """Cut the network for ``comm.size`` ranks and copy this
        rank's rows. Called ONCE by ModelMPI, on every rank, after the
        starting state is in place (init hooks, then supplied state or
        a restart): the blocks' state is copied from the grid dataset
        here. No collective: every rank computes the same cut from the
        same arrays.

        ``names_each_step``: this class's variables that rank 0 must
        hold in the grid dataset at every step."""
        from mpi4py import MPI

        unknown = [nn for nn in names_each_step if nn not in _WRITTEN]
        if unknown:
            # the same names on every rank: all ranks raise
            raise ValueError(
                f"PRMSChannelMPI: {unknown} are not its variables "
                f"({list(_WRITTEN)})."
            )
        refs = self._refs
        n_blocks = comm.size
        on_root = comm.rank == 0
        block_of = subtree_blocks(
            refs["tosegment0"], refs["segment_order"], n_blocks
        )
        column = sent_columns(refs["tosegment0"], block_of, n_blocks)

        self._comm = comm
        self._block = make_block(refs, block_of, column, n_blocks, comm.rank)
        self._main: _Block | None = None
        if on_root:
            self._main = make_block(refs, block_of, column, n_blocks, n_blocks)
        self._blocks = [
            bb for bb in (self._block, self._main) if bb is not None
        ]

        # the outlets' hourly outflow: the blocks' pieces fill rank 0's
        # outflow_given in rank order
        self._given_spec: list[Any] | None = None
        # rank 0: the segment of each row of a _Gather's recv
        self._recv_rows: np.ndarray | None = None
        row_counts = np.bincount(block_of, minlength=n_blocks + 1)[:n_blocks]
        if self._main is not None:
            sent_counts = 24 * np.bincount(
                block_of[column >= 0], minlength=n_blocks
            )
            self._given_spec = [
                self._main.outflow_given,
                sent_counts,
                np.concatenate([[0], np.cumsum(sent_counts)[:-1]]),
                MPI.DOUBLE,
            ]
            # stable: by block, then by segment = each block's `rows`;
            # the main stem (block n_blocks) comes last
            self._recv_rows = np.argsort(block_of, kind="stable")

        def gather_buffers(names: tuple[str, ...]) -> _Gather:
            n_names = len(names)
            send = np.zeros((self._block.n_own, n_names))
            if not on_root:
                return _Gather(names, send, None, None)
            recv = np.zeros((block_of.shape[0], n_names))
            counts = n_names * row_counts
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            return _Gather(
                names, send, recv, [recv, counts, starts, MPI.DOUBLE]
            )

        self._each_step = gather_buffers(tuple(names_each_step))
        self._everything = gather_buffers(_WRITTEN)

    def advance(self) -> None:
        if self._comm is None:
            super().advance()
            return
        # PRMSChannel.advance on each block this rank holds
        for block in self._blocks:
            block.arrays["inflow_ts_prev"][:] = block.arrays["seg_inflow"]

    def calculate(self, dt: np.float64, time: Time) -> None:
        if self._comm is None:
            super().calculate(dt, time)
            return
        block = self._block
        block.take_inputs(self._refs)
        block.run(dt)
        self._comm.Gatherv(block.outflow_sent, self._given_spec, root=0)
        if self._main is not None:
            self._main.take_inputs(self._refs)
            self._main.run(dt)
        self._gather(self._each_step)

    def gather_all(self) -> None:
        """COLLECTIVE: every variable of every segment into rank 0's
        grid dataset. ModelMPI calls it on every rank at the end of
        run() and in write_restart()."""
        if self._comm is not None:
            self._gather(self._everything)

    def _gather(self, gather: _Gather) -> None:
        """One Gatherv of ``gather.names``, then rank 0 puts each into
        the grid dataset's array in the grid's order. The message is
        (rows, names): the ranks' pieces are then one (rows, names)
        array on rank 0, with the main stem's rows written after
        them."""
        if not gather.names:
            return
        for ii, name in enumerate(gather.names):
            gather.send[:, ii] = self._block.arrays[name]
        self._comm.Gatherv(gather.send, gather.recv_spec, root=0)
        main = self._main
        if main is None:
            return
        assert gather.recv is not None and self._recv_rows is not None
        n_from_blocks = gather.recv.shape[0] - main.n_own
        for ii, name in enumerate(gather.names):
            gather.recv[n_from_blocks:, ii] = main.arrays[name][: main.n_own]
            self._refs[name][self._recv_rows] = gather.recv[:, ii]
