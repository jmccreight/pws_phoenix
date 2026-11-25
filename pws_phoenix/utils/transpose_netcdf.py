"""Copy a netCDF with the two dims of its time-varying variables
swapped: ``(time, nhru)`` -> ``(nhru, time)``, or back.

Both models take an input in either order and give the same answers.
What differs is the cost of a read from disk: one day of all hrus is
one unbroken run of a ``(time, nhru)`` file, many days of a rank's
block of hrus is one unbroken run of a ``(nhru, time)`` file
(machine_probes/README.md "The order of the dims decides the read").

Command line::

    python -m pws_phoenix.utils.transpose_netcdf SOURCE PATH [--force]
"""

import argparse
import pathlib as pl
from collections.abc import Sequence
from typing import Any

import netCDF4
import numpy as np

# The copy is written this many bytes at a time, beside the one whole
# variable held in memory.
SLAB_BYTES = 64 * 1024**2


def _is_swapped(var: netCDF4.Variable) -> bool:
    """A 2-D variable with a `time` dim: the ones whose dims swap."""
    return len(var.dimensions) == 2 and "time" in var.dimensions


def _signature(path: pl.Path, swap: bool) -> dict[str, Any]:
    """What must agree for an existing copy to be reused: each
    variable's name, dims (swapped as transpose_netcdf would when
    `swap`), shape along those dims, and ``source_file`` attribute."""
    signature: dict[str, Any] = {}
    with netCDF4.Dataset(path) as nc:
        for name, var in nc.variables.items():
            dims = tuple(var.dimensions)
            if swap and _is_swapped(var):
                dims = dims[::-1]
            signature[name] = (
                dims,
                tuple(nc.dimensions[dd].size for dd in dims),
                getattr(var, "source_file", None),
            )
    return signature


def _copy_attributes(source: Any, target: Any) -> None:
    target.setncatts(
        {
            name: source.getncattr(name)
            for name in source.ncattrs()
            if name != "_FillValue"
        }
    )


def transpose_netcdf(
    source: str | pl.Path, path: str | pl.Path, force: bool = False
) -> pl.Path:
    """Write `path`, a copy of the netCDF `source` in which every 2-D
    variable with a `time` dim has its two dims swapped. Returns
    `path`.

    Everything else is copied as it is: the other variables (the
    `time` coordinate among them), every attribute (a forcing file's
    ``source_file`` attributes too), the dtypes, the values bit for
    bit. The swapped variables are stored without chunks and without
    compression, whatever `source` uses.

    Memory: ONE whole variable at a time (7.9 GB for a CONUS forcing
    variable), plus a slab of SLAB_BYTES of it in the new order. The
    variable is read in one piece, the order `source` stores it in,
    and written slab by slab.

    An existing `path` is REUSED (nothing is read or written) when it
    holds the variables of `source` with the swapped dims, the same
    sizes and the same ``source_file`` attributes; when it does not,
    it raises. ``force=True`` rewrites it in either case.

    The file is written under ``<path>.partial`` and renamed at the
    end, so a file at `path` is always a complete one."""
    source = pl.Path(source)
    path = pl.Path(path)
    if path.resolve() == source.resolve():
        raise ValueError(
            f"transpose_netcdf: source and path are both {source}."
        )
    wanted = _signature(source, swap=True)
    if not any(
        "time" in dims and len(dims) == 2 for dims, *_ in wanted.values()
    ):
        raise ValueError(
            f"{source} has no 2-D variable with a 'time' dim: nothing "
            "to transpose."
        )
    if path.exists() and not force:
        found = _signature(path, swap=False)
        if found == wanted:
            return path
        raise FileExistsError(
            f"{path} exists and is not the transposed copy of {source}. "
            f"Its variables (dims, sizes, source_file): {found}. "
            f"Expected: {wanted}. Pass force=True to rewrite it."
        )

    partial = path.with_name(path.name + ".partial")
    with (
        netCDF4.Dataset(source) as nc,
        netCDF4.Dataset(partial, "w", format="NETCDF4") as out,
    ):
        # the values as stored: no masking, no scaling
        nc.set_auto_maskandscale(False)
        swapped = [nn for nn, vv in nc.variables.items() if _is_swapped(vv)]
        # no fill pass: every value is written below
        out.set_fill_off()
        _copy_attributes(nc, out)
        for name, dim in nc.dimensions.items():
            out.createDimension(name, dim.size)
        for name, var in nc.variables.items():
            dims = var.dimensions[::-1] if name in swapped else var.dimensions
            new = out.createVariable(
                name,
                var.dtype,
                dims,
                fill_value=getattr(var, "_FillValue", None),
                # False is netCDF4's default: the library chooses
                contiguous=name in swapped,
            )
            new.set_auto_maskandscale(False)
            _copy_attributes(var, new)
            if name not in swapped:
                new[...] = var[...]
                continue
            # a translation-time allocation: one whole variable, in
            # the order `source` stores it
            values = var[:]
            # rows of the copy = columns of `values`
            n_rows = values.shape[1]
            row_bytes = values.shape[0] * values.dtype.itemsize
            n_slab = max(1, SLAB_BYTES // row_bytes)
            for row0 in range(0, n_rows, n_slab):
                row1 = min(row0 + n_slab, n_rows)
                # the slab: a C-ordered copy of these columns
                new[row0:row1, :] = np.ascontiguousarray(
                    values[:, row0:row1].T
                )
            del values
    partial.replace(path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pws_phoenix.utils.transpose_netcdf",
        description="Copy a netCDF with the two dims of its 2-D time "
        "variables swapped: (time, nhru) -> (nhru, time), or back.",
    )
    parser.add_argument("source", type=pl.Path)
    parser.add_argument("path", type=pl.Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite an existing PATH",
    )
    args = parser.parse_args(argv)
    print(transpose_netcdf(args.source, args.path, force=args.force))


if __name__ == "__main__":
    main()
