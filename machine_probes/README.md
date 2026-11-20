<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [machine_probes](#machine_probes)
  - [The read probe](#the-read-probe)
    - [The order of the dims decides the read](#the-order-of-the-dims-decides-the-read)
    - [Run it](#run-it)
    - [What one run leaves](#what-one-run-leaves)
    - [Layouts](#layouts)
    - [Readers](#readers)
    - [How a read is made cold](#how-a-read-is-made-cold)
    - [A new machine](#a-new-machine)
    - [Not covered](#not-covered)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# machine_probes

Measurements of a MACHINE for a fixed version of the code: an HPC
system (its file system settings, its MPI-IO, the HDF5 of the
environment) or a laptop. `benchmarks/` (asv) follows the code over
commits on one machine; this follows the machines.

So far there is one probe, the read probe. A step-timing probe and an
environment check are planned.

## The read probe

Question: how long does a model wait for its forcing, read cold from a
`(time, hru)` netCDF, per layout of the file, per rank count and per
reader?

It was written after the Nov 20, 2026 measurements on Hovenweep, where
128 ranks reading their blocks of the CONUS `cbh.nc` (one Lustre
stripe) took about 16.5 minutes through the read `ModelMPI` does, and
82 s from a copy on 8 stripes.

### The order of the dims decides the read

The advice, by how many days one read takes and not by model:

- `(time, hru)`: serial runs, and `ModelMPI` at a shallow window
  (`control["mpi_window_steps"]`, default 1 day).
- `(hru, time)`: `ModelMPI` reading the whole record, or a window of
  years.
- In between: measure with this probe.

A variable is stored one element of its first dim after another. A
read that takes ALL of the last dim is one unbroken run of the file. A
read that takes part of the last dim is one piece per element of the
first dim it covers.

A model makes one of two reads:

- One day of all hrus (or of a rank's block): the serial `Input`
  reading from disk, and a `ModelMPI` rank at a window of 1 day. It
  is unbroken in a `(time, hru)` file.
- Many days of a rank's block of hrus: a `ModelMPI` rank at a deep
  window (`control["mpi_window_steps"]`). It is unbroken in a
  `(hru, time)` file.

So neither order is the better file: it depends on how many days one
read takes.

Pieces per read and variable for CONUS (114,958 hrus, 17,167 days,
899 hrus per rank at 128 ranks):

| one read                      | `(time, hru)` file   | `(hru, time)` file        |
| ----------------------------- | -------------------- | ------------------------- |
| serial, one day of all hrus   | 1                    | 114,958 pieces of 1 value |
| serial, `control["load_all"]` | 1                    | 1                         |
| MPI rank, window of 1 day     | 1                    | 899 pieces of 1 value     |
| MPI rank, window of 365 days  | 365 pieces of 899    | 899 pieces of 365         |
| MPI rank, the whole record    | 17,167 pieces of 899 | 1                         |

The counts are arithmetic, not measurements. Measured so far is the
last row (Hovenweep, Nov 20, 2026, 8 stripes, cold, 64 and 128
ranks): 81 to 132 s from the `(time, hru)` file, 8 s from the
`(hru, time)` file. The `step` reader measures the first row.

Chunks (`blocks<N>`, below) keep the `(time, hru)` order and still put
a rank's block together.

### Run it

Activate the environment (`pwpx_mpi`; `pwpx` on Hovenweep), get an
allocation under Slurm, then from the repo root:

```bash
bash machine_probes/run_read_probes.sh screen
bash machine_probes/run_read_probes.sh matrix
```

- `screen`: every reader once, at one rank count (the last of
  `RANKS_LIST`). It shows which readers differ on this machine.
- `matrix`: the reader `ModelMPI` uses (`mpio`) at every rank count,
  3 repeats. Add a reader the screen showed to matter:
  `READERS="mpio netCDF4" bash machine_probes/run_read_probes.sh matrix`.

The script makes the layout files that are missing, runs the reads and
prints the table. Every setting is a variable at the top of
`run_read_probes.sh` and can be given in the environment instead.

### What one run leaves

One run is one experiment:

```
machine_probes/logs/<cluster>/<domain>/<date_time>_<stage>/
    experiment.txt                        every variable as it ran, the commit
    make_layout.<layout>.log
    <layout>.r<ranks>.<reader>.<repeat>.log   one launch
    summary.txt                           the table
```

`logs/` is not tracked by git. Each launch log starts with a `#`
header: host, cluster, domain, layout, launcher, the file system and
its stripes, the chunks, package and library versions.

The table of any set of logs, also from different machines or domains:

```bash
python machine_probes/summarize.py machine_probes/logs/<cluster>/<domain>/<run> [more ...]
```

It reads the Nov 20, 2026 Hovenweep logs too (the line format is kept).

### Layouts

A layout is where the bytes lie. `LAYOUTS` holds one word per layout,
`name:stripes:stripe_size:arrangement`; the file is
`<LAYOUT_DIR>/cbh_<name>.nc`, and an existing file is used as it is.

| arrangement  | on disk                                                    |
| ------------ | ---------------------------------------------------------- |
| `contiguous` | `(time, hru)`, no chunks: what `translate_domain` writes   |
| `blocks<N>`  | `(time, hru)` in chunks of all days x `ceil(nhru / N)` hrus |
| `transposed` | `(hru, time)`, no chunks                                   |

In a `contiguous` file a rank's block of hrus is one small piece per
day, spread over the whole variable. The other two put a block's bytes
together.

`blocks<N>` names a chunk WIDTH for any domain: the block mpixarray
gives each of N ranks. mpixarray was seen to give every rank
`ceil(nhru / ranks)` hrus and the last rank the rest (Hovenweep, Nov
20, 2026: 114,958 hrus over 128 ranks = 127 blocks of 899 and one of
785), so at N ranks every rank reads exactly one chunk. The file is
not tied to N ranks: at another rank count a rank touches at most one
chunk more than its block needs. The reader `mpio-aligned` measures
what that costs (about 4% in the first Hovenweep run, on chunks of 449
hrus read by 128 ranks). `transposed` has no chunks, so it has no such
cost at any rank count.

Stripes are a Lustre setting (`lfs setstripe`); on another file system
they are ignored and the log says so. Nothing is compressed.

`make_layout.py` holds one whole variable in memory (7.9 GB for
CONUS).

### Readers

A rank's block, for the readers with MPI, is the one mpixarray gives
it: before the timed reads the probe makes the calls `ModelMPI` makes
(`open_dataset`, `parallelize` over the hru dimension, an allgather of
the local sizes). The readers without MPI cannot ask mpixarray, so
they take the rule above (`block_bounds` in `read_probe.py`). The
`# blocks:` line of every MPI log says whether the two are the same,
or that `parallelize` raised on this file.

Every reader reads each variable twice in one launch: `first` (cold,
when the cache was emptied) and `second` (warm).

| reader            | what it reads                                                          |
| ----------------- | ---------------------------------------------------------------------- |
| `mpio`            | the block, as `ModelMPI` does: mpixarray's open, independent MPI-IO     |
| `mpio-collective` | the block through h5py inside `with dset.collective:`                  |
| `mpio-aligned`    | as `mpio`, with the blocks moved onto chunk boundaries (chunked files) |
| `netCDF4`         | the block through netCDF-C, serial HDF5, no MPI                        |
| `xarray`          | the block through `xr.open_dataset`                                    |
| `rows`            | the block's number of bytes in the order the file stores them          |
| `step`            | one process: one day of all hrus per read, as the serial `Input` reads |

`rows` is what the file system gives when nothing has to be gathered.
`xarray` and `step` are in neither stage; name them in `READERS`.
`step` stops after `STEP_MAX_SECONDS` per variable and reports the
days it read.

### How a read is made cold

Before every launch the script drops the file from the file cache of
every node of the allocation (`os.posix_fadvise(...,
POSIX_FADV_DONTNEED)`, run on the node through the launcher), then
checks each node with `fincore` and stops when more than
`MAX_CACHED_PERCENT` of the file is still cached. macOS has neither
call, so on a laptop `EMPTY_CACHE=0` and the `first` read is of unknown
state.

### A new machine

Add a branch to the `case "$CLUSTER"` block of `run_read_probes.sh`:
the source file, where layouts go, the rank counts and the three
launchers. The cluster name is `$SLURM_CLUSTER_NAME`, or the host name
off Slurm; a name with no branch gets the laptop settings (a
synthesized file under `machine_probes/layouts/`, `mpirun`).

To learn on a new system before the first run: whether the file system
is Lustre (`lfs` exists), whether `fincore` exists on the compute
nodes, and whether dropping the cache empties it (the script checks
the last two and stops).

- Hovenweep: set.
- Pathfinder: TODO, the branch has no paths yet.

### Not covered

- Chunks in time: every chunk holds all days.
- More than one node has not been run. The script empties and checks
  every node of the allocation, so a two-node run is an allocation of
  two nodes and `RANKS_LIST=256`.
- The `mpio` read does not go through `as_iterable` and `iterate`.
  It is the expression they come to: `ModelMPI` reads a window with
  `step.mpi.src[name].values`, and mpixarray's `src` is the opened
  dataset `.isel(...)` of the rank's block and the time window. The
  probe evaluates that for a window of all days
  (`control["mpi_window_steps"] = None`). What it leaves out is the
  rest of a build: buffers, the writer, the static parameters.
- `ModelMPI` does not accept a `transposed` file yet: its build raises
  on a streamed input whose dims are not `(time, hru)`. mpixarray
  itself streams one through every call `ModelMPI` makes, and hands
  each window back in the file's order, `(block, days)` (checked Nov
  20, 2026 on LaNueva and on Hovenweep, 4 ranks, a toy file). The
  probe reads the layout with its own indexing.
