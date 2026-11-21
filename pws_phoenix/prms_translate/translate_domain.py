"""Translate one PRMS domain ONCE, ahead of any run: a PRMS control
file -> pws_phoenix input files in a flat domain directory.

    python -m pws_phoenix.prms_translate.translate_domain \\
        CONTROL_FILE --out DOMAIN_DIR [--cbh-name NAME] \\
        [--cbh-transpose] \\
        [--only {cbh,parameters,preprocessed} ...] [--force]

The domain directory is FLAT (the layout of test_data/<domain>/): one
directory holds the files of every control translated into it, each
control's own files named by the control file's stem.

    DOMAIN_DIR/
        cbh.nc                      (or cbh_<NAME>.nc)
        cbh_transpose.nc            (or cbh_<NAME>_transpose.nc; optional)
        <stem>.control              a copy of the control file
        <stem>_parameters.nc
        <stem>_preprocessed.nc
        <stem>_translation.yaml

- ``cbh.nc``: the control's CBH files in one netCDF
  (``cbh.write_cbh_netcdf``). Controls that name the same CBH files
  share it; a control with other CBH files needs its own name
  (``cbh_name``).
- ``cbh_transpose.nc``: written only when asked for
  (``cbh_transpose``): a copy of ``cbh.nc`` with every variable on
  ``(nhru, time)`` instead of ``(time, nhru)``
  (``utils.transpose_netcdf``), the same ``source_file`` attributes.
  Both models read either file and give the same answers; which one
  reads faster depends on the run (machine_probes/README.md "The
  order of the dims decides the read"). The yaml lists it as
  ``forcing_file_transpose``; a run is pointed at it by the user.
- ``<stem>_parameters.nc``: everything in the control's parameter
  file(s), as ``readers.load_parameters`` returns it. Its
  ``source_files`` attribute holds their resolved paths, one per line.
- ``<stem>_preprocessed.nc``: what a run would otherwise derive from
  the parameters at every start (solar tables, segment order,
  ``hru_in_to_cf``, the map weights), stamped
  (``preprocess.derive_preprocessed``). It is derived from
  ``<stem>_parameters.nc``, the arrays a run verifies the stamps
  against, and is written again whenever that file is. It carries the
  same ``source_files`` attribute.
- ``<stem>.control``: a byte-for-byte copy. It is a RECORD and the
  source of OPTIONS (modules, flags, the simulation window, the
  requested output variables). It is not a runnable PRMS control: its
  file entries (``param_file``, ``precip_day``, ...) are relative to
  the ORIGINAL's directory and must never be resolved from the copy.
- ``<stem>_translation.yaml``: which FILES the control maps to. The
  products are file names relative to the yaml's own directory (the
  domain directory can be moved); the files that were read are
  absolute paths. A run takes its file paths from here and its
  options from the control copy.

An existing product is reused when it was written from the files the
control names, and raises when it was written from others (see
``translate_domain`` for ``force``).
"""

import argparse
import hashlib
import pathlib as pl
from collections.abc import Sequence
from typing import Any

import xarray as xr
import yaml

from pws_phoenix.prms_translate.cbh import (
    cbh_netcdf_sources,
    write_cbh_netcdf,
)
from pws_phoenix.prms_translate.control import from_control
from pws_phoenix.prms_translate.preprocess import derive_preprocessed
from pws_phoenix.prms_translate.readers import load_control, load_parameters
from pws_phoenix.utils.transpose_netcdf import transpose_netcdf

STEPS = ("cbh", "parameters", "preprocessed")


def _raise_if_other_sources(
    path: pl.Path, found: Any, wanted: Any, hint: str = ""
) -> None:
    if found != wanted:
        raise FileExistsError(
            f"{path} exists and was not written from the files this "
            f"control names. Its sources: {found}. The control's: "
            f"{wanted}. Rewrite it (include its step and pass "
            f"force=True, --force){hint}."
        )


def translate_domain(
    control_file: str | pl.Path,
    out_dir: str | pl.Path,
    cbh_name: str | None = None,
    only: Sequence[str] | None = None,
    force: bool = False,
    cbh_transpose: bool = False,
) -> pl.Path:
    """Translate the files `control_file` names into `out_dir` (see
    the module docstring for the products). Returns the path of
    ``<stem>_translation.yaml``.

    `cbh_name`: the forcing file is ``cbh_<cbh_name>.nc`` instead of
    ``cbh.nc``. A name that is ``transpose`` or ends in ``_transpose``
    raises: it is the name of a transposed copy.

    `cbh_transpose`: the ``cbh`` step also writes the transposed copy
    of the forcing file, ``<forcing stem>_transpose.nc``
    (``utils.transpose_netcdf``; one whole variable in memory at a
    time). It raises when the ``cbh`` step is not selected. Without
    it an existing transposed copy is left as it is and stays listed
    in the yaml; a ``force`` that rewrites the forcing file raises
    when that would leave such a copy behind.

    `only`: the steps to run, of ``STEPS`` (default: all). The control
    copy and the yaml are written in every case. The yaml lists a
    product whose file does not exist as null.

    An existing product written from the files the control names is
    reused, and nothing is parsed or derived for it. One written from
    OTHER files raises, whether or not its step is selected.
    ``force=True`` rewrites the products of the selected steps.

    The ``preprocessed`` step derives from ``<stem>_parameters.nc``:
    selected without that file and without the ``parameters`` step it
    raises, and it runs again whenever this call writes the parameter
    file. A preprocessed file left behind by a later rewrite of the
    parameter file alone is NOT detected here: the run that reads it
    raises (``preprocess.verify_preprocessed``).

    An existing control copy that differs from `control_file` (the
    control file changed since the domain was translated) raises;
    ``force=True`` replaces the copy."""
    control_file = pl.Path(control_file).resolve()
    out_dir = pl.Path(out_dir)
    steps = STEPS if only is None else tuple(only)
    unknown = [ss for ss in steps if ss not in STEPS]
    if unknown:
        raise ValueError(
            f"translate_domain: unknown steps {unknown}; the steps are "
            f"{list(STEPS)}."
        )

    if cbh_name is not None and (
        cbh_name == "transpose" or cbh_name.endswith("_transpose")
    ):
        raise ValueError(
            f"translate_domain: cbh_name {cbh_name!r} is the name of a "
            "transposed copy (cbh_transpose.nc, "
            "cbh_<NAME>_transpose.nc): choose another."
        )
    if cbh_transpose and "cbh" not in steps:
        raise ValueError(
            "translate_domain: cbh_transpose (--cbh-transpose) is part "
            f"of the cbh step, which is not among the steps {list(steps)}."
        )

    cfg = from_control(load_control(control_file), control_file)
    if cfg.dyn_param_paths:
        raise NotImplementedError(
            f"{control_file} names dynamic-parameter files "
            f"({sorted(cfg.dyn_param_paths)}): translate_domain does "
            "not translate them yet."
        )
    stem = control_file.stem
    parameters = out_dir / f"{stem}_parameters.nc"
    if (
        "preprocessed" in steps
        and "parameters" not in steps
        and not parameters.exists()
    ):
        raise FileNotFoundError(
            f"{parameters} does not exist and the preprocessed step "
            "derives from it: run the parameters step too."
        )
    out_dir.mkdir(exist_ok=True)

    # -- the control copy --
    control_bytes = control_file.read_bytes()
    control_copy = out_dir / f"{stem}.control"
    # a control file translated inside its own directory is its own
    # copy
    if control_copy.resolve() != control_file:
        differs = (
            control_copy.exists()
            and control_copy.read_bytes() != control_bytes
        )
        if differs and not force:
            raise FileExistsError(
                f"{control_copy} differs from {control_file}: the "
                "control file changed since this domain was "
                "translated. Pass force=True (--force) to replace the "
                "copy."
            )
        if differs or not control_copy.exists():
            control_copy.write_bytes(control_bytes)

    # -- the forcing file --
    forcing_name = "cbh.nc" if cbh_name is None else f"cbh_{cbh_name}.nc"
    forcing = out_dir / forcing_name
    forcing_sources = {name: str(pp) for name, pp in cfg.cbh_paths.items()}
    rewrite = force and "cbh" in steps
    if forcing.exists() and not rewrite:
        _raise_if_other_sources(
            forcing,
            cbh_netcdf_sources(forcing),
            forcing_sources,
            hint=", or give this control its own forcing file "
            "(cbh_name, --cbh-name)",
        )
    # its transposed copy, when there is or will be one
    transposed = forcing.with_name(f"{forcing.stem}_transpose.nc")
    if transposed.exists() and not (rewrite and cbh_transpose):
        _raise_if_other_sources(
            transposed,
            cbh_netcdf_sources(transposed),
            forcing_sources,
            hint=", or give this control its own forcing file "
            "(cbh_name, --cbh-name)",
        )
        if rewrite:
            raise FileExistsError(
                f"{transposed} exists and would be left as the copy of "
                f"the {forcing.name} this call rewrites: pass "
                "cbh_transpose=True (--cbh-transpose) to rewrite both, "
                "or remove it."
            )
    if "cbh" in steps:
        write_cbh_netcdf(cfg.cbh_paths, forcing, force=force)
    if cbh_transpose:
        transpose_netcdf(forcing, transposed, force=force)

    # -- the parameter file --
    parameter_sources = [str(pp) for pp in cfg.param_files]
    rewrite = force and "parameters" in steps
    if parameters.exists() and not rewrite:
        with xr.open_dataset(parameters) as existing:
            found = existing.attrs.get("source_files")
        _raise_if_other_sources(
            parameters, found, "\n".join(parameter_sources)
        )
    parameters_written = "parameters" in steps and (
        rewrite or not parameters.exists()
    )
    if parameters_written:
        ds = load_parameters(cfg.param_files)
        ds.attrs["source_files"] = "\n".join(parameter_sources)
        partial = parameters.with_name(parameters.name + ".partial")
        ds.to_netcdf(partial)
        partial.replace(parameters)

    # -- the preprocessed file: derived from the parameter FILE (the
    # arrays a run verifies its stamps against), so written again
    # whenever that file is --
    preprocessed = out_dir / f"{stem}_preprocessed.nc"
    rewrite = (force or parameters_written) and "preprocessed" in steps
    if preprocessed.exists() and not rewrite:
        with xr.open_dataset(preprocessed) as existing:
            found = existing.attrs.get("source_files")
        _raise_if_other_sources(
            preprocessed, found, "\n".join(parameter_sources)
        )
    if "preprocessed" in steps and (rewrite or not preprocessed.exists()):
        ds = derive_preprocessed(
            xr.load_dataset(parameters), "prms_stream_temp" in cfg.classes
        )
        ds.attrs["source_files"] = "\n".join(parameter_sources)
        partial = preprocessed.with_name(preprocessed.name + ".partial")
        ds.to_netcdf(partial)
        partial.replace(preprocessed)

    # -- the yaml --
    info = {
        "control_file": str(control_file),
        "control_sha256": hashlib.sha256(control_bytes).hexdigest(),
        "control_copy": control_copy.name,
        "forcing_file": forcing.name if forcing.exists() else None,
        "forcing_file_transpose": (
            transposed.name if transposed.exists() else None
        ),
        "forcing_sources": forcing_sources,
        "parameter_file": parameters.name if parameters.exists() else None,
        "parameter_sources": parameter_sources,
        "preprocessed_file": (
            preprocessed.name if preprocessed.exists() else None
        ),
    }
    yaml_path = out_dir / f"{stem}_translation.yaml"
    with open(yaml_path, "w") as ff:
        yaml.safe_dump(info, ff, sort_keys=False)
    return yaml_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pws_phoenix.prms_translate.translate_domain",
        description="Translate the files a PRMS control file names "
        "into pws_phoenix input files in a flat domain directory.",
    )
    parser.add_argument("control_file", type=pl.Path)
    parser.add_argument(
        "--out",
        dest="out_dir",
        type=pl.Path,
        required=True,
        help="the domain directory (created if missing)",
    )
    parser.add_argument(
        "--cbh-name",
        help="write the forcing file as cbh_<CBH_NAME>.nc, not cbh.nc",
    )
    parser.add_argument(
        "--cbh-transpose",
        action="store_true",
        help="also write the forcing file's transposed copy, "
        "(nhru, time): cbh_transpose.nc, or cbh_<CBH_NAME>_transpose.nc",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=STEPS,
        help="the steps to run (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite the products of the selected steps, and replace "
        "a control copy that differs from the control file",
    )
    args = parser.parse_args(argv)
    print(
        translate_domain(
            args.control_file,
            args.out_dir,
            cbh_name=args.cbh_name,
            only=args.only,
            force=args.force,
            cbh_transpose=args.cbh_transpose,
        )
    )


if __name__ == "__main__":
    main()
