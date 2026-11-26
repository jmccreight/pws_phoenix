<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [The channel over MPI processes: research on branch `dag_research`](#the-channel-over-mpi-processes-research-on-branch-dag_research)
  - [1. Result](#1-result)
  - [2. Where the channel's time is today](#2-where-the-channels-time-is-today)
  - [3. What the kernel's order requires](#3-what-the-kernels-order-requires)
  - [4. Bounds from the network alone](#4-bounds-from-the-network-alone)
  - [5. Processes, not threads](#5-processes-not-threads)
  - [6. The cut](#6-the-cut)
  - [7. The kernel copy, and why the answers are the same](#7-the-kernel-copy-and-why-the-answers-are-the-same)
  - [8. Measurements](#8-measurements)
    - [8.1 Blocks as independent processes, no messages (the ceiling)](#81-blocks-as-independent-processes-no-messages-the-ceiling)
    - [8.2 The same, with the kernel copy](#82-the-same-with-the-kernel-copy)
    - [8.3 Under MPI, with the messages](#83-under-mpi-with-the-messages)
    - [8.4 Small domains, LaNueva](#84-small-domains-lanueva)
  - [9. Literature](#9-literature)
  - [10. The earlier note in `RESEARCH_NOTES/`](#10-the-earlier-note-in-research_notes)
  - [11. What is not measured](#11-what-is-not-measured)
  - [12. How to run the scripts](#12-how-to-run-the-scripts)
  - [13. Design in `ModelMPI`](#13-design-in-modelmpi)
    - [13.1 What is built](#131-what-is-built)
    - [13.2 The five questions](#132-the-five-questions)
    - [13.3 What a rank holds, and the kernel](#133-what-a-rank-holds-and-the-kernel)
    - [13.4 What is checked](#134-what-is-checked)
    - [13.5 Not done, and to decide](#135-not-done-and-to-decide)
  - [14. FlowGraph](#14-flowgraph)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# The channel over MPI processes: research on branch `dag_research`

Nov 24-25, 2026. JLM and Claude. The question: `PRMSChannel` routes
the segments in upstream-to-downstream order, one after another, on
one core. Under `ModelMPI` it runs on rank 0 alone, and at 16 ranks
it is most of the CONUS step. How much faster can it be made WITHOUT
changing its answers?

- [1. Result](#1-result)
- [2. Where the channel's time is today](#2-where-the-channels-time-is-today)
- [3. What the kernel's order requires](#3-what-the-kernels-order-requires)
- [4. Bounds from the network alone](#4-bounds-from-the-network-alone)
- [5. Processes, not threads](#5-processes-not-threads)
- [6. The cut](#6-the-cut)
- [7. The kernel copy, and why the answers are the same](#7-the-kernel-copy-and-why-the-answers-are-the-same)
- [8. Measurements](#8-measurements)
- [9. Literature](#9-literature)
- [10. The earlier note in `RESEARCH_NOTES/`](#10-the-earlier-note-in-research_notes)
- [11. What is not measured](#11-what-is-not-measured)
- [12. How to run the scripts](#12-how-to-run-the-scripts)
- [13. Design in `ModelMPI`](#13-design-in-modelmpi)
- [14. FlowGraph](#14-flowgraph)

Every number below was measured unless the sentence says it is an
inference. Hovenweep numbers are JLM's runs (one node, 2 x 64 cores,
`srun -n N -c 1 --mem=0`); LaNueva numbers are Claude's.

## 1. Result

The network is cut into a main stem and the subtrees that drain into
it. Each MPI rank routes a block of whole subtrees; rank 0 then
routes the main stem, given the hourly outflow of each subtree's
outlet. On CONUS (58,489 segments) with the messages between ranks
and one output array gathered to rank 0:

| ranks | ms per step | against 25.7 ms on one core |
|---|---|---|
| 1 | 28.71 | 0.9 |
| 8 | 3.77 | 6.8 |
| 16 | 1.76 | 14.6 |
| 32 | 1.24 | 20.7 |
| 64 | 1.17 | 21.9 |

Every array the kernel writes, for every segment, equals the present
kernel on the whole network bit for bit (CONUS at 16 and 64 blocks;
drb and ucb at 1 to 32).

The gain stops near 32 ranks: by then the main stem on rank 0 and the
gather of the output array are most of the step.

These numbers are from three scripts in `scratch/` (section 12), not
from the package. A first version of the same cut IN the package,
`PRMSChannelMPI` under `ModelMPI`, is section 13: it gives the
reference's answers bit for bit on drb and ucb, and it is not yet
timed on CONUS.

## 2. Where the channel's time is today

CONUS, Hovenweep, one core, per step:

| part | ms |
|---|---|
| six hru kernels | 50.6 (366-day mean) to 72.4 (January) |
| channel `calculate` | 25.7 to 35.6 |
| three Maps | 1.3 to 1.8 |

The hru part divides over processes (measured Nov 19, 2026 as
independent serial Models, one per block of hrus, no channel: 6.65 ms
at 8 processes, 1.51 at 32). The channel does not: under `ModelMPI`
the segment grid is held whole by every rank and its processes run on
rank 0. The 16-rank CONUS step without output is 38.5 ms.
Claude's inference from 50.6 / 16 = 3.2 ms for the hru part: about 33
of the 38.5 ms are the channel on rank 0, and more ranks will not
shorten the step.

The kernel costs 18.3 ns per segment per routing hour on Hovenweep
(25.7 ms / 58,489 / 24), and 3.7 to 5 on LaNueva.

## 3. What the kernel's order requires

`PRMSChannel._calculate` (`pws_phoenix/hydrology/prms_channel.py`):

```python
for ihr in range(24):
    seg_upstream_inflow[:] = _ZERO_CFS
    for jseg in segment_order:
        seg_current_inflow = (
            seg_lateral_inflow[jseg] + seg_upstream_inflow[jseg]
        )
        ...
        to_seg = tosegment0[jseg]
        if to_seg >= 0:
            seg_upstream_inflow[to_seg] += outflow_ts[jseg]
```

Within one hour a segment reads `seg_upstream_inflow[jseg]`, which
every segment directly upstream of it has added to in THIS hour. So:

1. A segment must run after every segment upstream of it, within the
   hour. Two segments with neither upstream of the other may run in
   either order or at the same time.
2. The order of the ADDS into one segment matters to the last bit
   when three or more segments drain into it: in floating point
   `(a + b) + c` is not always `(a + c) + b`. With two tributaries
   the sum is the same in either order (`a + b` equals `b + a`, and
   the first add is to zero).

Point 2 is why `Discretization.topological_order` replicates
pywatershed's construction exactly, and it constrains the cut
(section 7).

## 4. Bounds from the network alone

`scratch/network_speedup_bound.py` counts every segment as one unit of
work and counts starting a thread or sending a message as free. Three
ways to divide the work over P workers:

- **levels**: a segment's level is the longest path to it from a
  headwater. The segments of one level run together.
  Time = sum over the levels of `ceil(width / P)`.
- **basins**: whole basins, one per outlet, packed onto the workers.
  Time = the most loaded worker.
- **subtrees**: the segments with more than `cut` segments upstream
  of and including them are the main stem, run by one worker; the
  subtrees that drain into it are packed onto the workers.
  Time = the most loaded worker + the main stem, at the best `cut`.

CONUS: 58,489 segments, 1,155 outlets, 391 levels; the largest basin
holds 39% of the segments.

| workers | levels | basins | subtrees |
|---|---|---|---|
| 8 | 7.7 | 2.5 | 7.9 |
| 16 | 14.8 | 2.5 | 14.9 (cut 3,632, main stem 300) |
| 32 | 27.3 | 2.5 | 24.8 |
| 64 | 46.9 | 2.5 | 31.2 |
| 128 | 72.7 | 2.5 | 31.9 |

- Whole basins are useless on CONUS: the Mississippi is 39%.
- Levels and subtrees agree to 16 workers. Past 32 the levels bound
  keeps rising and the subtrees bound stops, because the main stem is
  one worker's.
- The 50 widest levels hold 96% of the segments. 270 of the 391
  levels are narrower than 8 and hold 593 segments together.
- drb has 456 segments in 72 levels; ucb 1,942 in 92.

What the two usable methods cost in starts or messages per step:

- subtrees, with the kernel's loops as they are (`for ihr` outside,
  `for jseg` inside): each worker runs its 24 hours without stopping.
  One hand-over per step.
- levels: each level of each hour is a separate parallel section.
  Counting only the 121 levels that are 8 or more wide (391 - 270):
  121 x 24 = 2,904 sections per step with the loops as they are, and
  121 with the hour loop moved inside the segment (which needs a
  `(24, nsegment)` buffer of hourly outflow).

## 5. Processes, not threads

Decision with JLM, Nov 24, 2026: subtrees spread over MPI processes,
not numba threads on rank 0.

- Our own hru kernels on CONUS: 5.2 times faster at 128 threads,
  33.5 times at 32 processes.
- mizuRoute found the same (section 9): 16 OpenMP threads gave 2.8
  to 4.6 times on the Mississippi.
- Every rank already holds the whole segment grid and its lateral
  inflows: each `MapMPI` ends in an `Allreduce` that fills the target
  buffer on every rank. Nothing more has to be sent TO a rank before
  it can route its subtrees.

## 6. The cut

The rule of Mizukami et al. 2021, as `scratch/channel_subtree_timing.py`
`partition` applies it for N blocks:

1. `size[j]` = the count of segments upstream of and including `j`.
2. The main stem is every segment with `size > n_seg // N`.
3. A subtree's outlet ("head" in the script) is a segment that is not
   on the main stem and drains into it, or a domain outlet that is
   not on the main stem.
4. The subtrees are taken largest first, each given to the least
   loaded of the N blocks.
5. Block N is the main stem.

Within a block the segments keep their present order in
`segment_order`.

On CONUS:

| blocks | main stem segments | subtree outlets that drain into it |
|---|---|---|
| 8 | 132 | not recorded |
| 16 | 291 | 249 |
| 32 | 548 | not recorded |
| 64 | 962 | 884 |

## 7. The kernel copy, and why the answers are the same

A subtree needs nothing from outside itself: its segments' lateral
inflows are already on the rank, and nothing drains into it. So a
block of subtrees run by the PRESENT kernel gives the whole network's
answers for its segments. Checked: 58,198 subtree segments of CONUS
at 16 blocks, bit for bit.

The main stem needs, each hour, the outflow of every subtree outlet
that drains into it. `calculate_block` in
`scratch/channel_subtree_timing.py` is `PRMSChannel._calculate` with
three added places:

```python
for jseg in segment_order:
    # ADDED: a row routed by another block only passes on the
    # outflow it was given
    kk = given_index[jseg]
    if kk >= 0:
        seg_upstream_inflow[tosegment0[jseg]] += outflow_given[kk, ihr]
        continue
    ...
    seg_outflow[jseg] += outflow_ts[jseg]
    # ADDED: the hourly outflow of a subtree outlet that
    # drains into the main stem
    kk = sent_index[jseg]
    if kk >= 0:
        outflow_sent[kk, ihr] = outflow_ts[jseg]
```

```python
# ADDED `and sent_index[jj] < 0`: a subtree outlet that
# drains into the main stem is not a domain outlet
if tosegment0[jj] < 0 and sent_index[jj] < 0:
    channel_outflow_vol[jj] = seg_outflow[jj] * s_per_time
```

- A block writes `outflow_sent[column, hour]` for each of its outlets
  that drain into the main stem: a `(outlets, 24)` array per block.
- The main stem block holds each such outlet as ONE MORE ROW, at the
  outlet's present place in `segment_order`. The row is not routed.
  Each hour it adds the outflow it was given to the segment it drains
  into.

The second point is what keeps the last bit (section 3, point 2): the
adds into a main stem segment are the same values in the same order
as in the whole network. Evidence that the order is needed and that
the kept order is enough:

- ucb has at most 4 places where 3 or more segments meet and one of
  them is a given row. A wrong order did NOT show there.
- An altered ucb (600 segments re-pointed one segment further down,
  which gives 481 confluences of 3 or more): with the given rows at
  their present places every array is equal; with the given rows
  moved to the front of the order, 7 arrays differ. (The test
  network of section 13.4 is a second such network. There the given
  rows moved to the front in their own order change nothing, and in
  reverse order they change 7 arrays.)

`verify` requires all segments, main stem included, to equal the
package kernel bit for bit in all 10 written arrays (compared as
64-bit patterns, so `-0.0` is not `0.0`): it passes on drb and ucb at
1 to 32 blocks and on CONUS at 16 and 64.

## 8. Measurements

### 8.1 Blocks as independent processes, no messages (the ceiling)

`channel_subtree_timing.py campaign`: N + 1 processes, each calling
the PACKAGE kernel on its block for 20 s at the same time; the main
stem without its tributaries' rows. CONUS, Hovenweep, Nov 24, 2026,
ms per step:

| processes | main stem segments | slowest block | main stem | sum | speedup | bound | core ms |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 25.719 | 0 | 25.719 | 1.0 | 1.0 | 25.7 |
| 8 | 132 | 3.163 | 0.043 | 3.206 | 8.0 | 7.9 | 24.9 |
| 16 | 291 | 1.393 | 0.086 | 1.480 | 17.4 | 14.8 | 21.0 |
| 32 | 548 | 0.578 | 0.150 | 0.729 | 35.3 | 24.7 | 17.5 |
| 64 | 962 | 0.268 | 0.276 | 0.544 | 47.3 | 31.2 | 16.4 |

`sum` = the slowest block + the main stem. `bound` =
`n_seg / (largest block + main stem)`. `core ms` = the blocks and the
main stem added up: what one core would take running them one after
another.

- No block is much slower than the others (the slowest is within 7%
  of the typical one), and no process started late.
- The speedup is ABOVE the bound from 16 processes on, because the
  work is cheaper in a small block: 18.3 ns per segment-hour for the
  whole network, about 12 in a block of 900 segments. The `core ms`
  column says the same: blocks run one after another on ONE core
  would take 16.4 ms, 1.6 times less than the whole network.
- At 64 the main stem is half the time; at 16 it is 6%.

### 8.2 The same, with the kernel copy

`campaign --copy`, Nov 25, 2026. `sum` ms:

| processes | package kernel | kernel copy | copy / package |
|---|---|---|---|
| 1 | 25.72 | 28.61 | 1.11 |
| 8 | 3.21 | 3.57 | 1.11 |
| 16 | 1.48 | 1.65 | 1.12 |
| 32 | 0.73 | 0.93 | 1.27 |
| 64 | 0.54 | 0.70 | 1.28 |

One campaign each, a day apart. The 11% is steady from 1 to 16
processes; Claude takes it as the cost of the two added index reads
per segment per hour (`given_index[jseg]`, `sent_index[jseg]`). The
main stem's rows nearly double with the given outlets: 0.086 to 0.112
ms at 16, 0.276 to 0.390 at 64.

### 8.3 Under MPI, with the messages

`scratch/channel_subtree_mpi.py`: rank r runs block r; one `Gatherv`
puts every rank's `outflow_sent` into rank 0's `outflow_given` (no
copy on rank 0); rank 0 runs the main stem; one `Gatherv` per output
array, which rank 0 then puts into the whole network's order. CONUS,
Hovenweep, Nov 25, 2026, one run per count, `seg_outflow` gathered,
rank 0's medians in ms per step:

| ranks | step | block | outlet message | main stem | output gather | against 25.7 ms |
|---|---|---|---|---|---|---|
| 1 | 28.71 | 28.57 | 0.009 | 0.008 | 0.118 | 0.9 |
| 8 | 3.77 | 3.40 | 0.104 | 0.066 | 0.195 | 6.8 |
| 16 | 1.76 | 1.36 | 0.030 | 0.124 | 0.240 | 14.6 |
| 32 | 1.24 | 0.66 | 0.058 | 0.229 | 0.290 | 20.7 |
| 64 | 1.17 | 0.32 | 0.114 | 0.425 | 0.309 | 21.9 |

- The bit-for-bit check passed at every count.
- The outlet message is cheap. Its time includes the wait for the
  slowest rank's block.
- The cost to design around is the gather of an OUTPUT array to
  rank 0: 0.2 to 0.3 ms per array per step. About 0.1 of it is
  putting the values into the whole network's order (it is 0.118 at
  1 rank, where nothing is sent). The CONUS control file asks for
  three segment variables.
- Past 32 ranks the step is flat: at 64 the main stem and the gather
  are 0.73 of the 1.17 ms.
- The mean step equals the median at every count; the largest step
  is about 8 ms. (On LaNueva at 8 ranks some steps stall: ucb's
  median is 0.24 ms, its mean 4.5 ms, its largest 403 ms. Hovenweep
  does not do this.)

### 8.4 Small domains, LaNueva

- ucb at 8 processes: 6.4 times faster (bound 5.8). The blocks one
  after another on one core: 0.17 ms against 0.239 for the whole
  network. drb shows no such gain.
- The kernel copy is about 20% slower than the package kernel there
  (ucb, whole network: 0.292 against 0.241 ms).

## 9. Literature

The PDFs are in `scratch/papers/` (untracked; the publishers refuse
automated fetches, JLM downloaded them).

- **Mizukami et al. 2021**, JAMES, doi 10.1029/2020MS002434
  (mizuRoute). The cut of section 6 is theirs: cut size = reaches /
  tasks; subtrees packed largest first onto the least loaded MPI
  task, the small ones onto the main stem's task. They cut the main
  stem again, by Strahler order, into branches for OpenMP threads.
  Read by Claude off the log plots of their Figure 7, so
  approximate: on the Mississippi their IRF routing goes from 45 ms
  serial to 1.6 ms at 32 MPI tasks, with no gain past 20 to 30 tasks
  (their text); 16 OpenMP threads alone gave 2.8 times (IRF) and 4.6
  times (KWT).
- **Yang et al. 2024**, Water 16(14) 1965, doi 10.3390/w16141965.
  Not measured on a machine: a "conceptual machine", the same formula
  as the levels bound of section 4. Yangtze: 53,600 reaches, longest
  path 1,042 (segments per level 51; CONUS is 150). Giving each reach
  its LATEST allowed level (counted from the outlet) instead of its
  earliest holds 80% efficiency to 25 processors instead of 14. Their
  breaks of the main stem use approximate inflow.
- **David et al. 2015** (RAPID), doi 10.1002/2014WR016650:
  approximate inflow at the boundaries between sub-basins, 8 times at
  16 cores on the Mississippi. Not exact, so not used here.

Our 14.6 times at 16 ranks and flat past 32 is the same shape as
mizuRoute's result, from the same cut.

## 10. The earlier note in `RESEARCH_NOTES/`

`RESEARCH_NOTES/river-routing-dag-vs-parallel.md` asks a different
question: drop the order by giving each segment the PREVIOUS
substep's upstream outflow, at the price of changed answers. Its own
advice was to measure first what the order allows without changing
any answer ("Do this before breaking the DAG"; its experiments 1 and
5). That is what this branch did:

- Its experiment 1, the level profile: section 4 (CONUS 391 levels,
  150 segments per level; its estimate of the width from Hack's law
  is replaced by the measured widths). The distributions of K, `ts`
  and `c0` in that experiment are NOT done.
- Its experiment 5, the exact baseline that any approximate scheme
  must beat: 14.6 times at 16 ranks, 21.9 at 64, with every answer
  unchanged. The baseline here is by subtrees over processes, not by
  levels with vectorized loops as the note proposed.
- Its experiments 2 to 4 (lagged inflow, lag compensation, halo
  reaches) are not done. With the exact cut at 1.2 to 1.8 ms per
  CONUS step, the channel is no longer the largest part of a 16-rank
  step, which removes the present reason to do them.

## 11. What is not measured

- More than one output array gathered per step, or a gather per
  window.
- The cut inside a real `ModelMPI` step, beside the hru kernels and
  the three `MapMPI` (about 1.7 ms per CONUS step at 1 core).
- Other processes on the segment grid (stream temperature, a
  FlowGraph).
- A second cut of the main stem, which is what a gain past 32 ranks
  needs (Mizukami's Strahler branches, or the levels of section 4
  inside the main stem).
- The latest-level order of Yang 2024 in `network_speedup_bound.py`.
- The package's block kernel (section 13) on CONUS. It has no added
  line inside the loop over segments, so it should not carry the 11%
  of section 8.2; on LaNueva it is 2% (drb) and 7% (ucb) slower than
  the package kernel on the whole network, where the kernel copy of
  the scripts is 20% slower.

## 12. How to run the scripts

All three are in `scratch/`, which is not tracked. Each script's
docstring is its manual. From the repo root, with `/tmp/dag_channel`
made first:

```
python scratch/network_speedup_bound.py test_data/drb_2yr/parameters_dis_seg.nc --widths
python scratch/channel_subtree_timing.py save <translation yaml or PRMS control file> /tmp/dag_channel/drb_2yr.npz
python scratch/channel_subtree_timing.py verify /tmp/dag_channel/drb_2yr.npz --blocks 8
python scratch/channel_subtree_timing.py campaign /tmp/dag_channel/drb_2yr.npz --local --copy
mpirun -n 8 python scratch/channel_subtree_mpi.py /tmp/dag_channel/drb_2yr.npz
```

On Hovenweep, inside an `salloc` allocation (`campaign` calls `srun`
itself):

```
python channel_subtree_timing.py campaign conus_channel_args.npz --copy
srun -n 16 -c 1 --mem=0 python channel_subtree_mpi.py conus_channel_args.npz
```

## 13. Design in `ModelMPI`

Written and built by Claude on Nov 24-25, 2026 while JLM was away
("You may implement and run checks"). EVERY choice below is a
proposal for JLM to accept or change. Nothing is committed. The
serial path and the default MPI path are unchanged: the divided
channel is used only when the caller names its class.

### 13.1 What is built

| file | what |
|---|---|
| `pws_phoenix/hydrology/prms_channel_mpi.py` (new) | `subtree_blocks`, `sent_columns` (the cut); `_calculate_block` (the kernel); `PRMSChannelMPI(PRMSChannel)` |
| `pws_phoenix/model.py` | `ModelMPI._decompose_processes` (new, called last in `_build`); three places in `run()`; one in `write_restart()` |
| `pws_phoenix/prms_translate/assemble_mpi.py` | `replace_classes={slot: class}` on `mpi_model_from_control` and `mpi_model_from_translation` |
| `tests/test_prms_channel_blocks.py` (new, serial) | the cut, the kernel, the order of the adds, the build's raises |
| `tests/test_prms_channel_divided_mpi.py` (new, MPI) | the class under `ModelMPI` against `PRMSChannel` on rank 0 |
| `tests/test_benchmark_coverage.py` | `PRMSChannelMPI` listed as not benchmarked |
| `scratch/conus_mpi_output_run.py` | `--channel-divided`, `--same-store A B` |

How it is asked for:

```python
from pws_phoenix.hydrology.prms_channel_mpi import PRMSChannelMPI

model = mpi_model_from_translation(
    yaml, control, replace_classes={"prms_channel": PRMSChannelMPI}
)
```

or `"class": PRMSChannelMPI` in a hand-written `process_dict`.

### 13.2 The five questions

**1. Hand-written messages, or a segment grid decomposed by
mpixarray? Chosen: hand-written (mpi4py), on the segment grid as it
is today.** The segment grid stays a serial grid, held whole by every
rank and fed by `MapMPI`. `PRMSChannelMPI` is to `PRMSChannel` what
`MapMPI` is to `Map`: the caller names the class, `ModelMPI` calls
its `set_decomposition(comm, names_each_step)` once at the build, and
the class makes its own messages. Reasons:

- `ModelMPI` relies on mpixarray dividing a dim into contiguous
  blocks in rank order (`model.py`: "scheme "single" = contiguous
  rank-ordered blocks"). A block of subtrees is not contiguous in the
  segment dim. The segments would have to be stored in block order
  and every output put back.
- mpixarray streams ONE decomposed dataset. A second decomposed grid
  needs two streams stepped together, which is the open question to
  the mpixarray developer (`CLAUDE.md` "Input structuring").
- The main stem is not a block like the others: one rank routes it
  after all the others, with their outlets' 24 hourly values. Claude
  knows of no mpixarray call for that; its source is outside the
  project directory and was not read.
- The messages are cheap (section 8.3: the outlet message is 0.03 to
  0.1 ms per CONUS step).

What it costs: mpi4py code in a process module, where until now only
`maps.py` and `model.py` had any. `CLAUDE.md` says of FlowGraph "No
FlowGraph MPI code"; this is the first process with MPI code.

**2. Output gathered per step, per window, or put in order only at
the write? Chosen: per step, and only the variables rank 0 reads.**
At the build `ModelMPI._decompose_processes` works out
`names_each_step`: the channel's variables that are requested output,
or an input of another process on the segment grid, or the source of
a Map. After the main stem, ONE `Gatherv` carries all of those names
(a `(rows, names)` array per rank), and rank 0 writes each into the
grid dataset's array in the grid's order. The rank-0 zarr `Output`
and every other process read the dataset exactly as today.

- Measured in the scripts: 0.2 to 0.3 ms per variable per CONUS
  step with one `Gatherv` per variable, of which about 0.1 is the
  reordering on rank 0. The package sends all names in one message;
  that is not timed on CONUS.
- Per window: each rank would keep `(depth, rows)` of each name and
  send once per window. It saves the message, not the reordering,
  and the zarr `Output` would need a way to take a window at once.
  It also cannot serve a process on rank 0 that reads the variable
  every step.
- In order only at the write: `Output` would keep the gathered rows
  in block order and permute a whole chunk when it writes. It saves
  the 0.1 ms per variable per step, for output-only names.
- Neither is built. With three output names at 16 ranks the gather
  is Claude's estimate of 0.5 to 0.8 ms of a step whose hru part is
  about 3 ms: worth doing after a CONUS timing says so.

**3. Restart. Chosen: the file format and `write_restart` /
`restart_read` are unchanged.** `ModelMPI.write_restart` first calls
`gather_all()` (every variable of every segment to rank 0), then
writes the segment grid from rank 0's copy as before. A restart is
read as before: every rank restores the whole segment grid, and
`set_decomposition`, which runs after it, copies each rank's rows.
Supplied initial state takes the same path. Tested: the restart file
is byte for byte the reference's, and a run warm-started from it
equals the reference.

**4. What becomes of "serial grids run on rank 0 only"? Chosen: it
stays the rule, with one named exception.** A serial-grid process
that has `set_decomposition()` runs on every rank and is never called
through `_call_alone`. Everything else on a serial grid still runs on
rank 0 only. On rank 0 the grid dataset holds the channel's
`names_each_step` at every step and all of its variables after
`run()` and in `write_restart()`. On the other ranks it keeps the
values of the build, which is today's rule. `control
["serial_grids_all_ranks"]` with a divided process raises at the
build (only rank 0 is given the channel's values).

**5. The other processes on the segment grid. Chosen: they stay on
rank 0 and read the channel from the grid dataset.** Tested with
drb's stream temperature control file: hydraulic geometry reads
`seg_outflow`, which is not a requested output there, so it is in
`names_each_step` only because a process reads it; every variable of
hydraulic geometry and stream temperature equals the reference.
What raises at the build:

- an input of the divided process that is not fed by a `MapMPI` from
  the distributed grid (only rank 0 would hold it: a time-varying
  segment input, or the output of a rank-0 process);
- another process that declares one of the channel's variables as a
  `mutable_input` (its edit would not reach the ranks);
- a budget or python-attribute restart state on the divided process.

### 13.3 What a rank holds, and the kernel

- Each rank holds ITS OWN COPY of its block's rows of the 20 arrays
  the kernel takes, cut out and numbered again at
  `set_decomposition` (rank 0 holds two such blocks: its subtrees and
  the main stem). This is a second set of arrays for the channel's
  names, which the "one ndarray per name" rule otherwise forbids. The
  reason is the measurement of section 8.1: 18.3 ns per segment-hour
  on the whole network, about 12 in a block of 900. The alternative,
  a rank running the kernel over a part of `segment_order` on the
  whole-network arrays, needs no copies and no gather on rank 0 for
  its own rows, but every whole-array line of the kernel
  (`seg_upstream_inflow[:] = _ZERO_CFS`, 24 times a step) would have
  to become a loop over the rank's segments. Not measured.
- The kernel is `_calculate_block`: `PRMSChannel._calculate` with
  three short loops added, all OUTSIDE the loop over segments.

```python
for ihr in range(24):
    seg_upstream_inflow[:] = _ZERO_CFS
    # ADDED (the main stem only)
    for kk in range(given_rows.shape[0]):
        seg_lateral_inflow[given_rows[kk]] = outflow_given[kk, ihr]
    for jseg in segment_order:
        ...  # PRMSChannel._calculate's lines, unchanged
    # ADDED (a block of subtrees only)
    for kk in range(sent_rows.shape[0]):
        outflow_sent[kk, ihr] = outflow_ts[sent_rows[kk]]
```

  A given row (section 7) is no longer a branch in the loop. It is
  data: its `tsi = -1` and `ts = 1.0` make the kernel's own lines
  compute `outflow_ts = inflow_ts`, and each hour its lateral inflow
  is set to the outflow it was given. The row then adds exactly that
  value to the segment it drains into, at its place in the order.
- Two kernels with one body is a hazard: a change to one must be made
  to the other. `tests/test_prms_channel_blocks.py` fails on any bit
  of difference. The alternative is ONE kernel, `PRMSChannel
  ._calculate` with four more arguments that are empty in serial. It
  was not done here because it touches the serial path, the kernel
  units test and the benchmarks; the LaNueva cost quoted in section
  11 (2% and 7%) is what the serial channel might pay, and it should
  be measured on Hovenweep before choosing.

### 13.4 What is checked

All on LaNueva, environment `pwpx_mpi`.

- `tests/test_prms_channel_blocks.py` (54 tests, no MPI): on drb,
  ucb and a ucb with
  600 segments re-pointed one segment further down (189 segments
  with 3 or more draining in), 1 to 33 blocks driven as the ranks
  drive them equal `PRMSChannel` in a serial Model on 4 days, every
  variable of every segment, as 64-bit patterns. On the re-pointed
  network the given rows put first in REVERSE order change 7 of the
  10 arrays, so the test can see a wrong order. (Put first in their
  own order they change nothing there: the first two adds into a
  segment commute.)
- `tests/test_prms_channel_divided_mpi.py` (24 tests), at 2, 3, 4
  and 7 ranks, for drb's `nhm.control` and its
  `nhm_stream_temp.control`: byte for byte against `PRMSChannel` on
  rank 0 for the zarr output of every step, every segment-grid
  variable after `run()`, the restart file, and a warm-started
  second period.
- ucb_2yr (1,942 segments) through `scratch/conus_mpi_output_run.py`,
  the control file's four segment output variables: 366 days at 4
  ranks, 60 days at 1 rank (one block, no main stem) and at 12 ranks
  (main stem 119 segments, given 100 outlets). `--same-store` prints
  `SAME` for each.
- `ruff`, `mypy` and the whole sweep `tests/run_tests.sh` (the serial
  suite, 733 tests, and the 26 MPI files at 4 ranks) pass with these
  changes.
- NOT checked: CONUS, any machine but LaNueva, a rank count above 12.

### 13.5 Not done, and to decide

- **Time it on CONUS.** On Hovenweep, with the scripts copied to
  `domains_pws_phoenix/scratch/` and the package pulled:

```
DOMAINS=<the domains_pws_phoenix directory on Hovenweep>
OUT=${DOMAINS}/conus_gf_v1.1_gridmet/output_mpi_divided
DAYS=366 OUT=${OUT} EXTRA="--no-output" bash ${DOMAINS}/scratch/conus_mpi_output_run.sbatch
DAYS=366 OUT=${OUT} EXTRA="--no-output --channel-divided" bash ${DOMAINS}/scratch/conus_mpi_output_run.sbatch
DAYS=366 OUT=${OUT} bash ${DOMAINS}/scratch/conus_mpi_output_run.sbatch
DAYS=366 OUT=${OUT} EXTRA="--channel-divided" bash ${DOMAINS}/scratch/conus_mpi_output_run.sbatch
srun -n 1 -c 1 --mem=0 python ${DOMAINS}/scratch/conus_mpi_output_run.py --same-store ${OUT}/segment_n16_d366_w183.zarr ${OUT}/segment_n16_d366_w183_divided.zarr
```

  (inside an `salloc` allocation with `pwpx_mpi` active, as on
  Nov 24; the last line is under `srun` because the script imports
  mpi4py.)

  The first pair is the step without output (38.5 ms at 16 ranks on
  Nov 24). Claude's inference from sections 2 and 8.3, not a
  measurement: about 33 ms of it is the channel on rank 0, and the
  divided channel should bring the step to 6 to 8 ms. The second
  pair writes about 11 GB each and the last line must print `SAME`.
- **`mpi_model_from_translation` cannot warm start.** Found while
  writing the test: it always supplies the translation's initial
  state, and `ModelMPI` raises on supplied state with
  `control["restart_read"]`. The test builds its warm start from the
  kit. Not changed.
- **Documentation, if the design is accepted:** `CLAUDE.md` "MPI
  path" (the exception to rank 0 only, the three raises, "No
  FlowGraph MPI code"), an entry in `docs/design_description.md`,
  the two-kernel hazard in `docs/MAINTENANCE.md`, a kernel units test
  for `_calculate_block`, and the README of the benchmarks. None is
  edited yet.
- **Every rank still receives the whole segment grid's lateral
  inflows** (each `MapMPI` ends in an `Allreduce` of `nsegment`
  values) though it uses only its block's. The earlier idea of a
  `Reduce` to rank 0 no longer fits; a rank needs its own rows.
- **The balance of the blocks** counts each segment as one unit, as
  the scripts do. Section 8.1 found the slowest block within 7% of
  the typical one on CONUS.
- **Past 32 ranks** the main stem needs its own cut (section 11).
- **The cut is recomputed at every build** by python loops over the
  segments (`subtree_blocks`): not timed on CONUS; the scripts'
  `partition` is the same code and was not noticed as slow.

## 14. FlowGraph

Whether the same cut carries over: yes for the structure, with four
things to build. Read from `flow_graph.py` (`_build_graph_kernel`);
nothing is implemented or measured.

The graph kernel has the channel's shape:

```python
for istep in range(n_substeps):
    state.node_upstream_inflow_sub[:] = 0.0
    for pos in range(n_nodes):
        inode = node_order[pos]
        ...  # the node type's substep
        to_node = to_graph_index[inode]
        if to_node >= 0:
            state.node_upstream_inflow_sub[to_node] += (
                state.node_outflow_substep[inode]
            )
```

A node reads only its own rows, its lateral inflow and what the nodes
upstream of it added in this substep. So a subtree of nodes needs
nothing from outside itself, and the main stem needs each subtree
outlet's `node_outflow_substep` for every substep: `(outlets,
n_substeps)` instead of `(outlets, 24)`. `subtree_blocks` and
`sent_columns` take `to_graph_index` and `node_order` as they are.

What is different, and would have to be built:

1. **The given row.** `PassThroughFlowNode`'s substep is already what
   a given row needs (`node_outflow_substep = node_upstream_inflow_sub
   + node_lateral_inflow`). The main stem's graph would hold each
   given outlet as a pass-through node whose lateral inflow is set
   each substep, by the same two added loops as in section 13.3, in
   `_build_graph_kernel`. A graph composed without the pass-through
   type would need it added for the main stem.
2. **The blocks' arrays.** A composed class's fields are `(nnodes,)`
   arrays in a namedtuple (`graph_state`); a block is a second
   namedtuple of its rows, with `to_graph_index` and `node_order`
   numbered again. The per-type init hooks work on the dataset and
   would run before the cut, as the channel's do.
3. **Inputs on every rank.** `node_obs_flow` (obsin) and
   `node_source_sink` are time-varying inputs of the graph's serial
   grid, which only rank 0 advances today. `_decompose_processes`
   raises on exactly this. Either the ranks advance those inputs
   too, or rank 0 sends each rank its rows.
4. **Budgets.** Every FlowGraph declares a mass budget, checked per
   node by the Model from the grid dataset's arrays.
   `_decompose_processes` raises on a budget. The check would have to
   run on each rank's block arrays, with the agreement across ranks
   that hru budgets already have.

Two smaller points:

- Counting each node as one unit of work is less true: a STARFIT node
  costs more than a channel node, and reservoirs sit on large rivers,
  that is on the main stem, which rank 0 routes alone.
- The cut, the gather buffers and the outlet message in
  `prms_channel_mpi.py` do not depend on the channel. They would move
  to a module of their own when a second process uses them.

Recommendation (Claude): settle the channel's design on CONUS first
(section 13.5), then do FlowGraph as its own piece of work, items 3
and 4 first because they change `ModelMPI` and not only the graph.
