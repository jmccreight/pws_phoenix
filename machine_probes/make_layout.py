"""Make one layout of a forcing netCDF for the read probe.

    python make_layout.py --source FILE --out FILE \
        --stripes 8 --stripe-size 4M --arrangement blocks256
    python make_layout.py --synthesize NTIME NHRU --out FILE \
        --stripes 8 --stripe-size 4M --arrangement transposed

A layout = where the bytes of the 2-D `time` variables lie on disk:

  --arrangement contiguous  (time, hru), no chunks
                blocks<N>   (time, hru), in chunks of all days x
                            ceil(number of hrus / N) hrus: the block
                            mpixarray gives each of N ranks, whatever
                            the domain, so at N ranks every rank reads
                            exactly one chunk. Other rank counts read
                            it too, each rank touching at most one
                            chunk more than its block needs.
                transposed  (hru, time), no chunks
  --stripes, --stripe-size  Lustre only: `lfs setstripe -c / -S`. On
                            any other file system they are ignored
                            (and that is printed).

Nothing is compressed. Dimension names and sizes, the other variables
and all attributes are those of the source; only `transposed` changes
the order of the two dimensions.

--source of a (time, hru) contiguous file with --arrangement contiguous
is a copy of the bytes (the `cp` of Nov 20, 2026). Every other case is
written through netCDF4, ONE WHOLE VARIABLE IN MEMORY at a time (7.9 GB
for CONUS; `transposed` holds a second copy while it writes).

--synthesize writes float32 `prcp`, `tmax`, `tmin` of random numbers on
(time, nhru) with a daily `time` coordinate.

The file is written in a new directory `<out>.partial/` that carries
the stripe settings (a file made in a directory inherits them), then
moved to --out. --out is never overwritten.
"""

import argparse
import os
import pathlib as pl
import platform
import shutil
import subprocess
import time
from typing import Any, NoReturn

import netCDF4
import numpy as np

SYNTHETIC_NAMES = ("prcp", "tmax", "tmin")
# HDF5: one chunk must be smaller than 4 GiB
MAX_CHUNK_BYTES = 4 * 1024**3 - 1


def stop(message: str) -> NoReturn:
    raise SystemExit(f"make_layout: {message}")


def parse_arrangement(text: str) -> tuple[str, int | None]:
    """("contiguous" | "chunked" | "transposed", N of blocks<N>)."""
    if text in ("contiguous", "transposed"):
        return text, None
    count = text.removeprefix("blocks")
    if text.startswith("blocks") and count.isdigit() and int(count) > 0:
        return "chunked", int(count)
    stop(
        f"--arrangement {text!r}: expected contiguous, transposed or "
        "blocks<N> (e.g. blocks256)"
    )


def on_lustre(directory: pl.Path) -> bool:
    if platform.system() != "Linux":
        return False
    done = subprocess.run(
        ["stat", "-f", "-c", "%T", str(directory)],
        capture_output=True,
        text=True,
    )
    return done.stdout.strip() == "lustre"


def stripe_count(path: pl.Path) -> int:
    done = subprocess.run(
        ["lfs", "getstripe", "-c", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(done.stdout.strip())


class FileSource:
    """The 2-D `time` variables of an existing netCDF, each handed over
    as a (time, hru) array whatever order the file stores."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.nc = netCDF4.Dataset(path)
        self.nc.set_auto_maskandscale(False)
        self.names = [
            name
            for name, var in self.nc.variables.items()
            if len(var.dimensions) == 2 and "time" in var.dimensions
        ]
        if not self.names:
            stop(f"{path}: no 2-D variable with a `time` dimension")
        dims = self.nc[self.names[0]].dimensions
        for name in self.names:
            if self.nc[name].dimensions != dims:
                stop(
                    f"{path}: {name!r} is on {self.nc[name].dimensions}, "
                    f"{self.names[0]!r} on {dims}; they must agree"
                )
        self.time_first = dims[0] == "time"
        self.hru_dim = dims[1] if self.time_first else dims[0]
        self.n_time = self.nc.dimensions["time"].size
        self.n_hru = self.nc.dimensions[self.hru_dim].size

    def describe(self) -> str:
        return self.path

    def is_time_first_contiguous(self) -> bool:
        return self.time_first and all(
            self.nc[name].chunking() == "contiguous" for name in self.names
        )

    def dimensions(self) -> dict[str, int]:
        return {name: dim.size for name, dim in self.nc.dimensions.items()}

    def global_attributes(self) -> dict[str, Any]:
        return {name: self.nc.getncattr(name) for name in self.nc.ncattrs()}

    def dtype(self, name: str) -> Any:
        return self.nc[name].dtype

    def attributes(self, name: str) -> dict[str, Any]:
        var = self.nc[name]
        return {att: var.getncattr(att) for att in var.ncattrs()}

    def values(self, name: str) -> np.ndarray:
        values = self.nc[name][:]
        return values if self.time_first else values.T

    def write_other_variables(self, out: netCDF4.Dataset) -> None:
        """Every variable that is not a 2-D `time` variable, unchanged
        (the `time` coordinate, for one)."""
        for name, var in self.nc.variables.items():
            if name in self.names:
                continue
            attributes = self.attributes(name)
            fill = attributes.pop("_FillValue", None)
            new = out.createVariable(
                name, var.dtype, var.dimensions, fill_value=fill
            )
            new.set_auto_maskandscale(False)
            new.setncatts(attributes)
            new[...] = var[...]

    def close(self) -> None:
        self.nc.close()


class SyntheticSource:
    """float32 random numbers on (time, nhru), one day per time."""

    def __init__(self, n_time: int, n_hru: int) -> None:
        self.names = list(SYNTHETIC_NAMES)
        self.time_first = True
        self.hru_dim = "nhru"
        self.n_time = n_time
        self.n_hru = n_hru
        self._rng = np.random.default_rng(0)

    def describe(self) -> str:
        return f"synthesized, {self.n_time} days x {self.n_hru} hrus"

    def is_time_first_contiguous(self) -> bool:
        return False  # there are no bytes to copy

    def dimensions(self) -> dict[str, int]:
        return {"time": self.n_time, "nhru": self.n_hru}

    def global_attributes(self) -> dict[str, Any]:
        return {}

    def dtype(self, name: str) -> Any:
        return np.float32

    def attributes(self, name: str) -> dict[str, Any]:
        return {}

    def values(self, name: str) -> np.ndarray:
        return self._rng.random((self.n_time, self.n_hru), dtype=np.float32)

    def write_other_variables(self, out: netCDF4.Dataset) -> None:
        var = out.createVariable("time", np.int32, ("time",))
        var.units = "days since 1979-01-01"
        var.calendar = "standard"
        var[:] = np.arange(self.n_time, dtype=np.int32)

    def close(self) -> None:
        pass


def write_netcdf(
    path: pl.Path,
    source: FileSource | SyntheticSource,
    kind: str,
    chunk_hrus: int | None,
    arrangement: str,
) -> None:
    if kind == "transposed":
        dims = (source.hru_dim, "time")
    else:
        dims = ("time", source.hru_dim)
    if chunk_hrus is None:
        storage: dict[str, Any] = {"contiguous": True}
    else:
        storage = {"chunksizes": (source.n_time, chunk_hrus)}
    with netCDF4.Dataset(path, "w", format="NETCDF4") as out:
        # no fill pass: every value is written below
        out.set_fill_off()
        out.setncatts(source.global_attributes())
        out.setncattr(
            "machine_probes_layout",
            f"{arrangement}, written by machine_probes/make_layout.py "
            f"from {source.describe()}",
        )
        for dim_name, size in source.dimensions().items():
            out.createDimension(dim_name, size)
        source.write_other_variables(out)
        for name in source.names:
            attributes = source.attributes(name)
            fill = attributes.pop("_FillValue", None)
            var = out.createVariable(
                name, source.dtype(name), dims, fill_value=fill, **storage
            )
            var.set_auto_maskandscale(False)
            var.setncatts(attributes)
            t0 = time.perf_counter()
            # one whole variable in memory, as (time, hru)
            values = source.values(name)
            t1 = time.perf_counter()
            # the transposed write makes netCDF4 copy the array once
            var[:] = values.T if kind == "transposed" else values
            t2 = time.perf_counter()
            print(
                f"# {name}: {values.nbytes / 1e6:.1f} MB, got in "
                f"{t1 - t0:.1f} s, written in {t2 - t1:.1f} s",
                flush=True,
            )
            del values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make one layout of a forcing netCDF; see the "
        "module docstring."
    )
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--source", help="the netCDF to rearrange")
    origin.add_argument(
        "--synthesize",
        nargs=2,
        type=int,
        metavar=("NTIME", "NHRU"),
        help="write random numbers of this size instead",
    )
    parser.add_argument("--out", required=True, type=pl.Path)
    parser.add_argument("--arrangement", required=True)
    parser.add_argument("--stripes", type=int, default=1)
    parser.add_argument("--stripe-size", default="1M")
    args = parser.parse_args()

    out: pl.Path = args.out
    kind, n_blocks = parse_arrangement(args.arrangement)
    if out.exists():
        stop(f"{out} exists; it is never overwritten")
    work_dir = out.with_name(out.name + ".partial")
    if work_dir.exists():
        stop(
            f"{work_dir} is left from an earlier run that did not "
            "finish; remove it and run again"
        )

    if args.source is not None:
        source: FileSource | SyntheticSource = FileSource(args.source)
    else:
        source = SyntheticSource(*args.synthesize)

    chunk_hrus = None
    if n_blocks is not None:
        # rounded up: the width of mpixarray's blocks at n_blocks ranks
        # (read_probe.block_bounds)
        chunk_hrus = -(-source.n_hru // n_blocks)
        item_bytes = max(
            np.dtype(source.dtype(name)).itemsize for name in source.names
        )
        chunk_bytes = source.n_time * chunk_hrus * item_bytes
        if chunk_bytes > MAX_CHUNK_BYTES:
            stop(
                f"blocks{n_blocks}: a chunk of {source.n_time} days x "
                f"{chunk_hrus} hrus is {chunk_bytes / 1e9:.1f} GB; HDF5 "
                "allows less than 4 GiB. Use more blocks."
            )
        n_chunks = -(-source.n_hru // chunk_hrus)
        print(
            f"# chunks: {source.n_time} days x {chunk_hrus} hrus = "
            f"{chunk_bytes / 1e6:.1f} MB each, {n_chunks} per variable"
        )
    print(f"# source: {source.describe()}")
    print(f"# arrangement: {args.arrangement}")

    lustre = on_lustre(out.parent)
    if lustre and shutil.which("lfs") is None:
        stop(
            f"{out.parent} is Lustre but this node has no `lfs` "
            "command: the stripes cannot be set"
        )
    work_dir.mkdir()
    if lustre:
        subprocess.run(
            [
                "lfs",
                "setstripe",
                "-c",
                str(args.stripes),
                "-S",
                args.stripe_size,
                str(work_dir),
            ],
            check=True,
        )
        print(f"# stripes: {args.stripes} x {args.stripe_size}")
    else:
        print("# stripes: ignored, the file system is not Lustre")

    partial = work_dir / out.name
    t_start = time.perf_counter()
    if kind == "contiguous" and source.is_time_first_contiguous():
        print("# the source is (time, hru) contiguous: copying its bytes")
        shutil.copyfile(args.source, partial)
    else:
        write_netcdf(partial, source, kind, chunk_hrus, args.arrangement)
    source.close()
    seconds = time.perf_counter() - t_start

    if lustre and args.stripes > 0 and stripe_count(partial) != args.stripes:
        stop(
            f"{partial} has {stripe_count(partial)} stripes, not the "
            f"{args.stripes} asked for; it is left there to look at"
        )
    os.replace(partial, out)
    work_dir.rmdir()
    print(
        f"# wrote {out}: {out.stat().st_size / 1e9:.2f} GB in {seconds:.1f} s"
    )


if __name__ == "__main__":
    main()
