"""Cut a translated domain to some of its hrus: a translated domain
(``translate_domain``) -> a translated domain of the hrus asked for,
in another directory.

    python -m pws_phoenix.prms_translate.cut_translation \\
        TRANSLATION_YAML --out DOMAIN_DIR \\
        (--nhm-ids-file FILE | --positions-file FILE)

The cut is a domain of HRUS ONLY. It has the files and the yaml keys
of a translated domain, so ``assemble_from_translation`` and
``verify_translation`` read it as they read any other, with the
processes on the hru grid named in ``processes=``:

    DOMAIN_DIR/
        cbh.nc                      (the source's forcing file name)
        cbh_transpose.nc            (when the source has one)
        <stem>.control              a copy of the source's control copy
        <stem>_parameters.nc
        <stem>_preprocessed.nc      (when the source has one)
        <stem>_translation.yaml

- ``<stem>_parameters.nc``: every variable on ``nhru`` cut to the
  hrus, in the source's order; the variables on the dims of
  ``WHOLE_DIMS`` alone as they are; every other variable (the
  segments, the gages) DROPPED, its dims listed in the file's
  ``dropped_dims`` attribute. Two additions: the variable
  ``source_position`` (each hru's 0-based position in the source
  domain) and the attributes ``cut_from`` (the source yaml) and
  ``source_nhru`` (the source's number of hrus). No process declares
  ``source_position``, so a model does not read it.
- ``<stem>_preprocessed.nc``: the derived arrays on ``nhru`` (the
  solar tables, ``hru_in_to_cf``) CUT from the source's, not derived
  again: the cut holds the source's values exactly. Each is stamped
  again, against the cut parameters. The segment order and the map
  weights are dropped (the weights index hrus by position).
- The forcing files: every variable cut to the hrus, all of its
  times, in the source's dtype.
- The yaml: the source's keys, and ``cut_from``.

Why hrus only: a basin's channel needs its segments cut and numbered
again, and the map weights rebuilt. That is not done here. A build of
a cut that keeps a process off the hru grid raises
(``assemble_from_translation``).

A model of the cut gives those hrus' columns of a model of the
source, identical (tests/test_cut_translation.py): an hru's answers
depend on its own parameters and forcing alone.
"""

import argparse
import pathlib as pl
import shutil
from collections.abc import Sequence

import numpy as np
import xarray as xr
import yaml

from pws_phoenix.prms_translate.preprocess import cut_preprocessed
from pws_phoenix.utils.transpose_netcdf import transpose_netcdf

# the dims a variable may have beside nhru and still be kept; a
# variable on these alone is kept whole
WHOLE_DIMS = ("one", "nmonths", "ndeplval")


def _positions(
    params: xr.Dataset,
    hru_positions: Sequence[int] | np.ndarray | None,
    nhm_ids: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    """The hrus asked for as ascending positions in the source domain
    (see cut_translation for what raises)."""
    if (hru_positions is None) == (nhm_ids is None):
        raise ValueError(
            "cut_translation: give hru_positions or nhm_ids, one of them."
        )
    nhru = params.sizes["nhru"]
    if hru_positions is not None:
        positions = np.asarray(hru_positions, dtype=np.int64)
        if (
            positions.ndim != 1
            or positions.size == 0
            or positions[0] < 0
            or positions[-1] >= nhru
            or (np.diff(positions) <= 0).any()
        ):
            raise ValueError(
                "cut_translation: hru_positions must be ascending "
                f"positions without repeats, each from 0 to {nhru - 1} "
                "(the source domain's last hru)."
            )
        return positions

    if "nhm_id" not in params:
        raise ValueError(
            "cut_translation: the source's parameter file has no "
            "nhm_id; give hru_positions."
        )
    asked = np.asarray(nhm_ids, dtype=np.int64)
    source_ids = params["nhm_id"].values
    if asked.ndim != 1 or asked.size == 0:
        raise ValueError("cut_translation: nhm_ids is empty or not 1-D.")
    if np.unique(asked).size != asked.size:
        raise ValueError("cut_translation: nhm_ids holds repeats.")
    if np.unique(source_ids).size != source_ids.size:
        raise ValueError(
            "cut_translation: the source's nhm_id holds repeats, so an "
            "nhm_id does not name one hru; give hru_positions."
        )
    wanted = np.isin(source_ids, asked)
    if wanted.sum() != asked.size:
        missing = np.setdiff1d(asked, source_ids)
        raise ValueError(
            f"cut_translation: {missing.size} of the {asked.size} "
            "nhm_ids are not in the source domain; the first are "
            f"{missing[:10].tolist()}."
        )
    return np.flatnonzero(wanted)


def _cut_parameters(
    whole: xr.Dataset, positions: np.ndarray, source_yaml: pl.Path
) -> xr.Dataset:
    """The parameter file's content for the cut (module docstring)."""
    kept_dims = {"nhru", *WHOLE_DIMS}
    kept = [
        name for name in whole.data_vars if set(whole[name].dims) <= kept_dims
    ]
    dropped_dims = sorted(
        {
            str(dim)
            for name in whole.data_vars
            if name not in kept
            for dim in whole[name].dims
        }
        - kept_dims
    )
    # one read of the span from the first position to the last, then
    # the hrus out of it in memory
    first = int(positions[0])
    span = slice(first, int(positions[-1]) + 1)
    cut = whole[kept].isel(nhru=span).load().isel(nhru=positions - first)
    cut = cut.drop_vars(list(cut.coords))
    cut["source_position"] = xr.DataArray(
        positions,
        dims=("nhru",),
        attrs={
            "description": "0-based position of the hru in the domain "
            "this one was cut from (the cut_from attribute)"
        },
    )
    cut.attrs["cut_from"] = str(source_yaml)
    cut.attrs["source_nhru"] = whole.sizes["nhru"]
    cut.attrs["dropped_dims"] = " ".join(dropped_dims)
    return cut


def _cut_forcing(
    source: pl.Path, path: pl.Path, positions: np.ndarray
) -> None:
    """Write `path`, the forcing file `source` (either order of its
    two dims) cut to the hrus, on (time, nhru) as write_cbh_netcdf
    writes it: no chunks, no compression, the two coordinates, each
    variable's dtype and attributes. One variable is in memory at a
    time, over the hrus from the first position to the last."""
    first = int(positions[0])
    span = slice(first, int(positions[-1]) + 1)
    with xr.open_dataset(source) as whole:
        for count, name in enumerate(whole.data_vars):
            # one read of the span, then the hrus out of it in memory
            da = whole[name].isel(nhru=span).load()
            da = da.isel(nhru=positions - first).transpose("time", "nhru")
            da = da.drop_vars([cc for cc in da.coords if cc != "time"])
            da = da.drop_encoding()
            if count == 0:
                da.assign_coords(
                    nhru=np.arange(positions.size)
                ).to_dataset().to_netcdf(path, mode="w")
            else:
                da.drop_vars("time").to_dataset().to_netcdf(path, mode="a")


def cut_translation(
    translation_yaml: str | pl.Path,
    out_dir: str | pl.Path,
    hru_positions: Sequence[int] | np.ndarray | None = None,
    nhm_ids: Sequence[int] | np.ndarray | None = None,
) -> pl.Path:
    """Cut the translated domain of `translation_yaml` to some of its
    hrus and write the cut, a translated domain of hrus only, in
    `out_dir` (see the module docstring for its files). Returns the
    path of the cut's ``<stem>_translation.yaml``.

    The hrus are named ONE of two ways:

    - `hru_positions`: 0-based positions in the source domain,
      ascending, without repeats;
    - `nhm_ids`: values of the source parameter file's ``nhm_id``, in
      any order, without repeats.

    In both cases the cut keeps the SOURCE's order of the hrus.

    Raises ValueError for neither or both of them, for positions out
    of order, repeated or outside the domain, and for an ``nhm_id``
    that the source does not hold; FileExistsError when `out_dir`
    already holds one of the files (a cut is written once);
    FileNotFoundError when the source yaml names no parameter file or
    no forcing file.

    Memory: everything on ``nhru`` is read over the SPAN of hrus from
    the first position to the last, in one piece, and the hrus are
    taken out of it in memory: one forcing variable at a time, its
    times by the span. The forcing is read from the source's
    transposed copy when it has one: there the span is one piece of
    the file."""
    source_yaml = pl.Path(translation_yaml).resolve()
    out_dir = pl.Path(out_dir)
    with open(source_yaml) as ff:
        info = yaml.safe_load(ff)
    source_dir = source_yaml.parent
    not_run = [
        key
        for key in ("forcing_file", "parameter_file")
        if info.get(key) is None
    ]
    if not_run:
        raise FileNotFoundError(
            f"{source_yaml} names no {' and no '.join(not_run)}: run "
            "translate_domain with those steps."
        )
    file_keys = (
        "control_copy",
        "parameter_file",
        "preprocessed_file",
        "forcing_file",
        "forcing_file_transpose",
    )
    targets = {
        key: out_dir / info[key]
        for key in file_keys
        if info.get(key) is not None
    }
    yaml_path = out_dir / source_yaml.name
    existing = [
        str(path) for path in (*targets.values(), yaml_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"cut_translation: {existing} exist. A cut is written once, "
            "into a directory that holds none of its files."
        )

    with xr.open_dataset(source_dir / info["parameter_file"]) as whole:
        positions = _positions(whole, hru_positions, nhm_ids)
        params = _cut_parameters(whole, positions, source_yaml)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / info["control_copy"], targets["control_copy"])
    params.to_netcdf(targets["parameter_file"])

    if "preprocessed_file" in targets:
        with xr.open_dataset(source_dir / info["preprocessed_file"]) as whole:
            # stamped against the parameter FILE just written: the
            # arrays a run verifies the stamps against
            cut_preprocessed(
                whole, xr.load_dataset(targets["parameter_file"]), positions
            ).to_netcdf(targets["preprocessed_file"])

    read_key = (
        "forcing_file_transpose"
        if "forcing_file_transpose" in targets
        else "forcing_file"
    )
    _cut_forcing(
        source_dir / info[read_key], targets["forcing_file"], positions
    )
    if "forcing_file_transpose" in targets:
        transpose_netcdf(
            targets["forcing_file"], targets["forcing_file_transpose"]
        )

    with open(yaml_path, "w") as ff:
        yaml.safe_dump(
            info | {"cut_from": str(source_yaml)}, ff, sort_keys=False
        )
    return yaml_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pws_phoenix.prms_translate.cut_translation",
        description="Cut a translated domain to some of its hrus: a "
        "translated domain of hrus only, in another directory.",
    )
    parser.add_argument("translation_yaml", type=pl.Path)
    parser.add_argument(
        "--out",
        dest="out_dir",
        type=pl.Path,
        required=True,
        help="the cut's domain directory (created if missing)",
    )
    named = parser.add_mutually_exclusive_group(required=True)
    named.add_argument(
        "--nhm-ids-file",
        type=pl.Path,
        help="a text file of nhm_ids, one integer per line",
    )
    named.add_argument(
        "--positions-file",
        type=pl.Path,
        help="a text file of 0-based positions in the source domain, "
        "ascending, one integer per line",
    )
    args = parser.parse_args(argv)
    path = args.nhm_ids_file or args.positions_file
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    print(
        cut_translation(
            args.translation_yaml,
            args.out_dir,
            hru_positions=values if args.positions_file else None,
            nhm_ids=values if args.nhm_ids_file else None,
        )
    )


if __name__ == "__main__":
    main()
