#!/usr/bin/env python
"""Populate <root>/<domain> from a pws-prms-oracle answer set and
derive the answer variables PRMS does not write next to it.

    PWS_ORACLE_DIR=<oracle clone>/domains \\
    python test_data/generate/oracle_populate.py [--root <dir>] \\
        [--domain drb_2yr ...] [--control_pattern nhm.control] \\
        [--no-derive] [--force]

Ported from pywatershed's test_data/generate/oracle_populate.py
(feat_oracle, Nov 2026); the tree it writes has the same layout and
markers, so tests/conftest.py reads either. Differences: --root (default
pws_phoenix's test_data/), PWS_ORACLE_DIR is required (there is no
release-asset fallback yet), and the control files are read with
pws_phoenix.prms_translate, not pywatershed.

Without --domain every domain in the oracle is populated. The oracle
directory is laid out like the oracle's domains/: <domain>/<inputs> and
<domain>/<output dir named by the control>/<var>.nc. It is never
written to. Each file is symlinked into <root>/<domain>/ (copied where
symlinks are unavailable); the key markers (.oracle_ikey, .oracle_okey)
are always copied, so a later regeneration in the oracle is detected as
a mismatch. The derived variables (prms_diagnostic_variables.py) are
written as regular files alongside; .derived_okey records the answers
they were derived from.
"""

import argparse
import os
import pathlib as pl
import shutil
import sys
import warnings

from prms_diagnostic_variables import derive_all

from pws_phoenix.prms_translate.readers import load_control

REPO_ROOT = pl.Path(__file__).parents[2]
TEST_DATA_DIR = REPO_ROOT / "test_data"

# oracle files the derive step overwrites: never link these, a write
# through the link would land in the oracle (the derive step also
# unlinks before every write)
DERIVE_OVERWRITES = {"seg_lateral_inflow.nc"}
# oracle run products that are not inputs
SKIP_INPUTS = ("output", "soltab_debug", ".DS_Store")
SKIP_INPUT_SUFFIXES = (".out", ".day", ".log", ".msgs")
MARKERS = (".oracle_ikey", ".oracle_okey")
OUTPUT_DIR_ENTRIES = ("nhruOutBaseFileName", "nsegmentOutBaseFileName")


def oracle_root() -> pl.Path:
    """PWS_ORACLE_DIR: a pws-prms-oracle clone's domains/."""
    env = os.environ.get("PWS_ORACLE_DIR")
    if env is None:
        raise SystemExit(
            "set PWS_ORACLE_DIR to a pws-prms-oracle domains/ directory"
        )
    root = pl.Path(env).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"PWS_ORACLE_DIR={root} is not a directory")
    return root


def unlink(path: pl.Path) -> None:
    """Remove path if present. Copies of the oracle's files are read-only
    (copy2 keeps the 0o444), and Windows refuses to unlink a read-only
    file, so make it writable first."""
    if path.is_symlink() or path.exists():
        if not path.is_symlink():
            path.chmod(0o644)
        path.unlink()


def link_or_copy(src: pl.Path, dst: pl.Path, force: bool) -> str:
    """Symlink src at dst, copying when symlinks are unavailable.
    Returns what was done. An existing regular file is left alone unless
    force."""
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "kept"
        dst.unlink()
    elif dst.exists():
        if not force:
            return "regular file kept"
        unlink(dst)
    try:
        os.symlink(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def copy_marker(src_dir: pl.Path, dst_dir: pl.Path, name: str) -> None:
    src = src_dir / name
    # a second populate must remove the read-only copy before rewriting it
    unlink(dst_dir / name)
    if src.exists():
        shutil.copy2(src, dst_dir / name)
    else:
        warnings.warn(f"{src_dir}: no {name}; the oracle predates key markers")


def populate_inputs(src_dom: pl.Path, dst_dom: pl.Path, force: bool):
    dst_dom.mkdir(parents=True, exist_ok=True)
    counts = {}
    for src in sorted(src_dom.iterdir()):
        if src.name.startswith(SKIP_INPUTS) or src.name in MARKERS:
            continue
        if src.suffix in SKIP_INPUT_SUFFIXES or src.is_dir():
            continue
        done = link_or_copy(src, dst_dom / src.name, force)
        counts[done] = counts.get(done, 0) + 1
    copy_marker(src_dom, dst_dom, ".oracle_ikey")
    print(f"{dst_dom}: {counts}")


def populate_outputs(src_out: pl.Path, dst_out: pl.Path):
    dst_out.mkdir(parents=True, exist_ok=True)
    counts = {}
    # the netCDF answers, PRMS's raw solar table (print_debug 5 controls)
    # and the few csv files kept for test_preprocess_csv
    srcs = (
        list(src_out.glob("*.nc"))
        + list(src_out.glob("*.csv"))
        + list(src_out.glob("soltab_debug*"))
    )
    for src in sorted(srcs):
        if src.name in DERIVE_OVERWRITES:
            continue
        done = link_or_copy(src, dst_out / src.name, force=True)
        counts[done] = counts.get(done, 0) + 1
    copy_marker(src_out, dst_out, ".oracle_okey")
    print(f"{dst_out}: {counts}")


def output_dir_name(control_file: pl.Path) -> str:
    """The directory PRMS wrote this control's answers to
    (nhruOutBaseFileName / nsegmentOutBaseFileName; the oracle requires
    exactly one), as a name under the domain."""
    ctl = load_control(control_file)
    dirs = {
        pl.Path(str(ctl.get(entry).values).strip()).name
        for entry in OUTPUT_DIR_ENTRIES
        if ctl.exists(entry)
    }
    if len(dirs) != 1:
        raise SystemExit(
            f"{control_file}: expected one output directory, found {dirs}"
        )
    return dirs.pop()


def derive(control_file: pl.Path, out_dir: pl.Path) -> None:
    """Write the derived variables into out_dir (the oracle's files are
    linked there, the derived ones are regular)."""
    derive_all(control_file, out_dir)
    # what the derived files were made from; "unkeyed" when the oracle
    # wrote no marker
    marker = out_dir / ".oracle_okey"
    key = marker.read_text().strip() if marker.exists() else "unkeyed"
    (out_dir / ".derived_okey").write_text(key + "\n")


def select_controls(dom_dir: pl.Path, patterns: list[str]) -> list[pl.Path]:
    controls = [
        cc
        for cc in sorted(dom_dir.glob("*.control"))
        if "make_cbh_only" not in cc.name
    ]
    if patterns:
        controls = [
            cc for cc in controls if any(pp in cc.name for pp in patterns)
        ]
    return controls


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=pl.Path,
        default=TEST_DATA_DIR,
        help=f"the tree to populate; default: {TEST_DATA_DIR}",
    )
    parser.add_argument(
        "--domain", action="append", help="default: every domain in the oracle"
    )
    parser.add_argument("--control_pattern", action="append", default=[])
    parser.add_argument("--no-derive", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace regular input files with links",
    )
    args = parser.parse_args(argv)
    src_root = oracle_root()
    dst_root = args.root.expanduser().resolve()
    domains = args.domain or sorted(
        dd.name
        for dd in src_root.iterdir()
        if dd.is_dir() and not dd.name.startswith(".")
    )

    for domain in domains:
        src_dom, dst_dom = src_root / domain, dst_root / domain
        if not src_dom.is_dir():
            raise SystemExit(f"{src_dom}: no such domain under {src_root}")
        populate_inputs(src_dom, dst_dom, args.force)
        for control_file in select_controls(src_dom, args.control_pattern):
            out_name = output_dir_name(control_file)
            src_out = src_dom / out_name
            if not src_out.is_dir():
                raise SystemExit(
                    f"{domain}:{control_file.stem}: no answers at {src_out}; "
                    "generate them in the oracle clone first"
                )
            dst_out = dst_dom / out_name
            populate_outputs(src_out, dst_out)
            if not args.no_derive:
                print(f"{dst_out}: deriving from {control_file.name}")
                derive(dst_dom / control_file.name, dst_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
