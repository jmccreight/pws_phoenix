"""Read probe: time reads of a forcing netCDF on this machine.

    python read_probe.py read FILE --reader READER [--days N]
    python read_probe.py drop-cache FILE

`read` is started once per process (rank) by the launcher; see
run_read_probes.sh, which is the way to run it. Each process reads its
block of hrus over all days (or the first --days), per variable, twice
in the one launch: the `first` read is cold when the file cache was
emptied beforehand, the `second` is warm. Every read opens the file
again; the imports, the open and the read are timed separately.

The file holds 2-D variables on `time` and one space dimension, in
one of three arrangements (make_layout.py writes them):
  (time, hru), contiguous
  (time, hru), in chunks of (all days, some hrus)
  (hru, time), contiguous ("transposed")

A rank's block:
  readers with MPI     the block mpixarray gives the rank, found by the
                       calls ModelMPI._build makes (open_dataset,
                       parallelize over the hru dimension, the cartesian
                       communicator, an allgather of the local sizes).
                       This is done before the timed reads; it reads no
                       variable.
  readers without MPI  the probe's rule (block_bounds): every rank holds
                       ceil(n_hru / n_ranks) hrus and the last rank the
                       rest. It is what mpixarray was SEEN to do
                       (Hovenweep, Nov 20, 2026: 114,958 hrus over 128
                       ranks = 127 blocks of 899 and one of 785); it is
                       not np.array_split's rule.
The `# blocks:` header line of every MPI log says whether the two are
the same, so an mpixarray that cuts otherwise shows up there. When
parallelize raises, the line says so and the probe's rule is read.

Readers without MPI (the rank is read from the launcher's environment):
  netCDF4  the block through netCDF-C (serial HDF5)
  xarray   the block through xr.open_dataset (netCDF4 engine)
  rows     the same NUMBER OF BYTES as the block, in the order the
           file stores them: whole days of a (time, hru) contiguous
           file, whole hrus of a transposed one, whole chunks of a
           chunked one (all days, whatever --days says). What the file
           system gives when nothing has to be gathered.
  step     ONE process: one day of all hrus per read, day after day,
           as the serial file-backed `Input.advance` reads. Stops after
           --max-seconds per variable and reports the days it read; the
           second attempt reads those same days.
Readers with MPI (the rank is COMM_WORLD's):
  mpio             mpixarray.open_dataset, as ModelMPI opens its input
                   (h5netcdf, driver "mpio"), then
                   `ds.isel(time=..., <hru>=...)[name].values`: the
                   expression ModelMPI's window read
                   (`step.mpi.src[name].values`) comes to in mpixarray,
                   for a window of all days. An INDEPENDENT read.
  mpio-collective  h5py, driver "mpio", the block read inside
                   `with dset.collective:`
  mpio-aligned     as mpio, but the blocks are whole chunks (the chunks
                   are dealt to the ranks evenly), so no rank reads
                   part of a chunk. Against mpio on the
                   same chunked file this is the cost of blocks that
                   do not line up with chunks. Prints "not applicable"
                   for a file without chunks.

Output: `#` header lines from rank 0, then one line per rank, variable
and attempt, in the format of the Nov 20, 2026 Hovenweep logs (the file
name is the last field), so summarize.py reads both.
"""

import argparse
import dataclasses
import importlib.metadata
import os
import pathlib as pl
import platform
import shutil
import socket
import subprocess
import time
from typing import Any

MPI_READERS = ("mpio", "mpio-collective", "mpio-aligned")
READERS = MPI_READERS + ("netCDF4", "xarray", "rows", "step")
PACKAGES = (
    "numpy",
    "xarray",
    "netCDF4",
    "h5py",
    "h5netcdf",
    "mpi4py",
    "mpixarray",
    "mpix",
)
# (rank variable, process count variable), the first pair present wins:
# Slurm's srun, MPICH's mpiexec, Open MPI's mpirun (two spellings)
RANK_ENVIRONMENT = (
    ("SLURM_PROCID", "SLURM_STEP_NUM_TASKS"),
    ("SLURM_PROCID", "SLURM_NTASKS"),
    ("PMI_RANK", "PMI_SIZE"),
    ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
    ("PMIX_RANK", "OMPI_COMM_WORLD_SIZE"),
)


@dataclasses.dataclass
class Layout:
    """What the file holds, as far as the readers need it."""

    names: list[str]
    n_time: int
    n_hru: int
    hru_dim: str
    time_first: bool
    # the chunk's extent along hru and along time; None = contiguous
    chunk_hrus: int | None
    chunk_days: int | None
    header: list[str]


def drop_cache(path: str) -> None:
    """Ask the system to drop this file's pages from the file cache of
    the node this runs on (the Nov 20, 2026 `$DROP` one-liner)."""
    # by name: macOS has neither, and mypy there knows it
    advise = getattr(os, "posix_fadvise", None)
    dont_need = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dont_need is None:
        raise SystemExit(
            f"drop-cache: os.posix_fadvise does not exist on "
            f"{platform.system()}"
        )
    fd = os.open(path, os.O_RDONLY)
    try:
        advise(fd, 0, 0, dont_need)
    finally:
        os.close(fd)


def environment_rank() -> tuple[int, int]:
    """(rank, number of ranks) from the launcher's environment
    variables; (0, 1) when no launcher set them."""
    for rank_name, size_name in RANK_ENVIRONMENT:
        if rank_name in os.environ and size_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ[size_name])
    return 0, 1


BLOCK_RULE = (
    "the probe's rule (ceil(hrus / ranks) per rank, the last rank the rest)"
)


def block_bounds(n_hru: int, n_ranks: int, rank: int) -> tuple[int, int]:
    """[start, stop) of this rank's block of hrus by the rule mpixarray
    was seen to use: every rank holds ceil(n_hru / n_ranks) hrus and
    the last rank the rest. The readers without MPI read these;
    mpixarray_block() checks them against mpixarray at every MPI
    launch."""
    width = -(-n_hru // n_ranks)
    start = min(rank * width, n_hru)
    return start, min(start + width, n_hru)


def even_share(n_items: int, n_ranks: int, rank: int) -> tuple[int, int]:
    """[start, stop) of this rank's share of n_items dealt evenly: the
    first `n_items % n_ranks` ranks hold one more (np.array_split's
    rule). No rank is left empty while n_items >= n_ranks."""
    base, extra = divmod(n_items, n_ranks)
    start = rank * base + min(rank, extra)
    return start, start + base + (1 if rank < extra else 0)


def aligned_bounds(
    n_hru: int, chunk_hrus: int, n_ranks: int, rank: int
) -> tuple[int, int]:
    """[start, stop) in hrus of this rank's share of WHOLE chunks, the
    chunks dealt evenly. When the chunk width is the ranks' block
    width (blocks<N> read by N ranks) these are block_bounds's."""
    n_chunks = -(-n_hru // chunk_hrus)
    first, last = even_share(n_chunks, n_ranks, rank)
    return min(first * chunk_hrus, n_hru), min(last * chunk_hrus, n_hru)


def mpixarray_block(
    path: str, layout: Layout, modules: dict[str, Any], comm: Any
) -> tuple[int, int, str]:
    """This rank's [start, stop) of the hru dimension as mpixarray cuts
    it, and the `# blocks:` header text. The calls are those of
    ModelMPI._build and Discretization.decompose; the bounds are found
    as ModelMPI finds them (an allgather of the local sizes, summed
    over the lower ranks of the cartesian communicator).

    Every rank must reach the same outcome, so nothing is raised here:
    a failure is gathered and reported, and block_bounds's block is
    returned instead."""
    split = block_bounds(layout.n_hru, comm.size, comm.rank)
    start = stop = total = 0
    same = False
    error = ""
    try:
        world = modules["mpix"].world_comm()
        ds_input = modules["mpi_open_dataset"](path, comm=world, quiet=True)
        ds_mpi = ds_input.mpi.parallelize(
            "single", dims=[layout.hru_dim], comm=world
        )
        cart = ds_mpi.mpi.get_cartesian_communicator(comm=world)
        sizes = cart.allgather(int(ds_mpi.sizes[layout.hru_dim]))
        start = int(sum(sizes[: cart.rank]))
        stop = start + int(sizes[cart.rank])
        total = int(sum(sizes))
        same = (start, stop) == block_bounds(
            layout.n_hru, comm.size, cart.rank
        )
        ds_input.close()
    except Exception as err:
        error = f"{type(err).__name__}: {err}"
    errors = [ee for ee in comm.allgather(error) if ee]
    if errors:
        return (
            *split,
            f"{BLOCK_RULE}. mpixarray's parallelize raised on "
            f"{len(errors)} of {comm.size} ranks: {errors[0]}",
        )
    if total != layout.n_hru:
        return (
            *split,
            f"{BLOCK_RULE}. The local sizes after mpixarray's "
            f"parallelize add up to {total}, not {layout.n_hru} hrus",
        )
    all_same = comm.allreduce(same, op=modules["MPI"].LAND)
    verdict = "the same as" if all_same else "NOT the same as"
    return (
        start,
        stop,
        f"mpixarray's parallelize; {verdict} {BLOCK_RULE}, which the "
        "readers without MPI read",
    )


def file_system_type(path: str) -> str:
    if platform.system() != "Linux":
        return f"not determined on {platform.system()}"
    done = subprocess.run(
        ["stat", "-f", "-c", "%T", path], capture_output=True, text=True
    )
    return done.stdout.strip() or "unknown"


def stripe_settings(path: str) -> str:
    if shutil.which("lfs") is None:
        return "no lfs command"
    done = subprocess.run(
        ["lfs", "getstripe", "-c", "-S", path],
        capture_output=True,
        text=True,
    )
    text = done.stdout if done.returncode == 0 else done.stderr
    return ", ".join(text.split("\n")).strip(", ") or "unknown"


def machine_header(args: argparse.Namespace) -> list[str]:
    """The header lines that need no heavy import. Called before
    mpi4py is imported: it starts `stat` and `lfs`, and a process that
    has initialized MPI should not start others."""
    path = args.file
    lines = [
        f"host: {socket.gethostname()}",
        f"cluster: {args.cluster}",
        f"domain: {args.domain}",
        f"layout: {args.layout}",
        f"reader: {args.reader}",
        f"launcher: {args.launcher}",
        f"cache emptied before this launch: {args.cache_emptied}",
        f"file: {path}",
        f"file size: {os.path.getsize(path)} bytes",
        f"file system: {file_system_type(path)}",
        f"stripes: {stripe_settings(path)}",
        f"python: {platform.python_version()}",
    ]
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"{package}: {version}")
    return lines


def load_modules(reader: str) -> dict[str, Any]:
    """Import what the reader needs, in the order of the Nov 20, 2026
    script (the MPI stack first), and return the modules by name."""
    modules: dict[str, Any] = {}
    if reader in MPI_READERS:
        import h5py
        import mpix
        from mpi4py import MPI
        from mpixarray import open_dataset

        modules.update(
            h5py=h5py, mpix=mpix, MPI=MPI, mpi_open_dataset=open_dataset
        )
    import netCDF4
    import xarray

    modules.update(netCDF4=netCDF4, xarray=xarray)
    return modules


def inspect_file(path: str, modules: dict[str, Any]) -> Layout:
    """Read the dimensions, the arrangement and the chunks. Every
    2-D variable with a `time` dimension is probed; they must agree."""
    netCDF4 = modules["netCDF4"]
    with netCDF4.Dataset(path) as nc:
        names = [
            name
            for name, var in nc.variables.items()
            if len(var.dimensions) == 2 and "time" in var.dimensions
        ]
        if not names:
            raise SystemExit(
                f"{path}: no 2-D variable with a `time` dimension"
            )
        dims = nc[names[0]].dimensions
        chunking = nc[names[0]].chunking()
        header = [f"format: {nc.data_model}"]
        for name in names:
            var = nc[name]
            if var.dimensions != dims or var.chunking() != chunking:
                raise SystemExit(
                    f"{path}: {name!r} has dimensions "
                    f"{var.dimensions} and chunking {var.chunking()}; "
                    f"{names[0]!r} has {dims} and {chunking}. The "
                    "probed variables must agree."
                )
            header.append(
                f"{name}: {var.dtype}, dimensions {var.dimensions}, "
                f"chunking {var.chunking()}"
            )
        time_first = dims[0] == "time"
        hru_dim = dims[1] if time_first else dims[0]
        n_time = nc.dimensions["time"].size
        n_hru = nc.dimensions[hru_dim].size
        header.append(
            f"netCDF-C {netCDF4.__netcdf4libversion__}, its HDF5 "
            f"{netCDF4.__hdf5libversion__}"
        )
    if chunking == "contiguous":
        chunk_hrus = chunk_days = None
    else:
        chunk_days, chunk_hrus = chunking if time_first else reversed(chunking)
    return Layout(
        names=names,
        n_time=n_time,
        n_hru=n_hru,
        hru_dim=hru_dim,
        time_first=time_first,
        chunk_hrus=chunk_hrus,
        chunk_days=chunk_days,
        header=header,
    )


def block_index(
    layout: Layout, n_days: int, start: int, stop: int
) -> tuple[slice, slice]:
    """Index of hrus [start, stop) over the first n_days days."""
    if layout.time_first:
        return (slice(0, n_days), slice(start, stop))
    return (slice(start, stop), slice(0, n_days))


def require_whole_time_chunks(layout: Layout, reader: str) -> None:
    if layout.chunk_days != layout.n_time:
        raise SystemExit(
            f"{reader}: the chunks hold {layout.chunk_days} of "
            f"{layout.n_time} days. Only chunks of all days are handled."
        )


def rows_index(
    layout: Layout, n_days: int, n_ranks: int, rank: int
) -> tuple[slice, slice]:
    """The `rows` reader's index: as many bytes as the rank's block,
    in the order the file stores them."""
    if layout.chunk_hrus is not None:
        require_whole_time_chunks(layout, "rows")
        start, stop = aligned_bounds(
            layout.n_hru, layout.chunk_hrus, n_ranks, rank
        )
        return block_index(layout, layout.n_time, start, stop)
    start, stop = block_bounds(layout.n_hru, n_ranks, rank)
    if layout.time_first:
        n_lead, n_trail = layout.n_time, layout.n_hru
    else:
        n_lead, n_trail = layout.n_hru, layout.n_time
    # whole leading-dimension rows holding the block's number of values
    n_rows = -(-n_days * (stop - start) // n_trail)
    row0 = (rank * n_rows) % max(n_lead - n_rows, 1)
    return (slice(row0, row0 + n_rows), slice(None))


def open_file(reader: str, path: str, modules: dict[str, Any], comm: Any):
    if reader in ("mpio", "mpio-aligned"):
        return modules["mpi_open_dataset"](
            path, comm=modules["mpix"].world_comm(), quiet=True
        )
    if reader == "mpio-collective":
        return modules["h5py"].File(path, "r", driver="mpio", comm=comm)
    if reader in ("xarray", "step"):
        return modules["xarray"].open_dataset(path)
    return modules["netCDF4"].Dataset(path)


def read_block(
    reader: str,
    handle: Any,
    name: str,
    index: tuple[slice, slice],
    layout: Layout,
) -> int:
    """Read the indexed values; return the number of bytes read."""
    if reader in ("mpio", "mpio-aligned"):
        # ModelMPI's window read is `step.mpi.src[name].values`, and
        # mpixarray's `src` is the opened dataset `.isel(...)` by
        # dimension name of the rank's block and the time window
        # (mpixarray MPI_Object.src). This is that expression for a
        # window of n_days.
        if layout.time_first:
            dims = ("time", layout.hru_dim)
        else:
            dims = (layout.hru_dim, "time")
        return handle.isel(dict(zip(dims, index)))[name].values.nbytes
    if reader == "xarray":
        return handle[name][index].values.nbytes
    if reader == "mpio-collective":
        dset = handle[name]
        with dset.collective:
            return dset[index].nbytes
    var = handle[name]
    var.set_auto_maskandscale(False)
    return var[index].nbytes


def read_steps(
    handle: Any,
    name: str,
    layout: Layout,
    n_days: int,
    max_seconds: float,
) -> tuple[int, int]:
    """One day of all hrus per read, as `Input.advance` does
    (`self.data[index, :].values`), until n_days are read or
    max_seconds have passed. Returns (days read, bytes read)."""
    data = handle[name]
    n_bytes = 0
    days_read = 0
    t_start = time.perf_counter()
    for tt in range(n_days):
        if layout.time_first:
            n_bytes += data[tt, :].values.nbytes
        else:
            n_bytes += data[:, tt].values.nbytes
        days_read = tt + 1
        if time.perf_counter() - t_start > max_seconds:
            break
    return days_read, n_bytes


def read(args: argparse.Namespace) -> None:
    reader = args.reader
    env_rank, env_n_ranks = environment_rank()
    if env_rank == 0:
        for line in machine_header(args):
            print(f"# {line}")

    t_import = time.perf_counter()
    modules = load_modules(reader)
    t_import = time.perf_counter() - t_import
    if reader in MPI_READERS:
        comm = modules["MPI"].COMM_WORLD
        rank, n_ranks = comm.rank, comm.size
    else:
        comm = None
        rank, n_ranks = env_rank, env_n_ranks
    if reader == "step" and n_ranks != 1:
        raise SystemExit(
            f"step: start ONE process, not {n_ranks} (it reads all hrus)"
        )

    layout = inspect_file(args.file, modules)
    if rank == 0:
        for line in layout.header:
            print(f"# {line}")
        if reader in MPI_READERS:
            mpi_library = modules["MPI"].Get_library_version().split("\n")
            print(f"# MPI: {mpi_library[0]}")
            print(f"# h5py's HDF5: {modules['h5py'].version.hdf5_version}")
        print(f"# rank 0 imports {t_import:.2f} s", flush=True)

    n_days = layout.n_time if args.days is None else args.days
    start, stop = block_bounds(layout.n_hru, n_ranks, rank)
    blocks = (
        f"{BLOCK_RULE}; a reader without MPI, an mpio log says "
        "whether mpixarray cuts the same"
    )
    if reader in MPI_READERS:
        start, stop, blocks = mpixarray_block(args.file, layout, modules, comm)
    if rank == 0 and reader != "step":
        print(f"# blocks: {blocks}", flush=True)
    if reader == "mpio-aligned":
        if layout.chunk_hrus is None:
            if rank == 0:
                print(
                    "# mpio-aligned: not applicable, the variables "
                    "have no chunks",
                    flush=True,
                )
            return
        require_whole_time_chunks(layout, reader)
        start, stop = aligned_bounds(
            layout.n_hru, layout.chunk_hrus, n_ranks, rank
        )
    if reader == "rows":
        index = rows_index(layout, n_days, n_ranks, rank)
    else:
        index = block_index(layout, n_days, start, stop)
    if reader == "step":
        start, stop = 0, layout.n_hru

    host = socket.gethostname()
    file_name = pl.Path(args.file).name
    for name in layout.names:
        days_to_read = n_days
        for attempt in ("first", "second"):
            if comm is not None:
                comm.Barrier()
            t0 = time.perf_counter()
            handle = open_file(reader, args.file, modules, comm)
            t1 = time.perf_counter()
            if comm is not None:
                comm.Barrier()
            t2 = time.perf_counter()
            if reader == "step":
                days_to_read, n_bytes = read_steps(
                    handle, name, layout, days_to_read, args.max_seconds
                )
            else:
                n_bytes = read_block(reader, handle, name, index, layout)
            t3 = time.perf_counter()
            handle.close()
            megabytes = n_bytes / 1e6
            rate = megabytes / max(t3 - t2, 1e-9)
            # the field positions are those of the Nov 20, 2026 logs
            print(
                f"rank {rank:4d} of {n_ranks} {host} {reader} {name} "
                f"{attempt} hrus {start}:{stop} days {days_to_read} "
                f"{megabytes:.1f} MB open {t1 - t0:.3f} "
                f"read {t3 - t2:.3f} {rate:.1f} MB/s {file_name}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time reads of a (time, hru) netCDF; see the "
        "module docstring and run_read_probes.sh."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    drop = commands.add_parser(
        "drop-cache", help="drop FILE from this node's file cache"
    )
    drop.add_argument("file")

    reads = commands.add_parser("read", help="time this rank's reads")
    reads.add_argument("file")
    reads.add_argument("--reader", required=True, choices=READERS)
    reads.add_argument(
        "--days", type=int, default=None, help="default: all days"
    )
    reads.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="the step reader stops after this long per variable",
    )
    # recorded in the header only
    reads.add_argument("--cluster", default="not given")
    reads.add_argument("--domain", default="not given")
    reads.add_argument("--layout", default="not given")
    reads.add_argument("--launcher", default="not given")
    reads.add_argument("--cache-emptied", default="not given")

    args = parser.parse_args()
    if args.command == "drop-cache":
        drop_cache(args.file)
    else:
        read(args)


if __name__ == "__main__":
    main()
