#!/usr/bin/env python
"""Compare the derived variables of two populated test-data trees, file
by file and value by value (nan equal to nan).

    python test_data/generate/compare_derived.py --root <tree> \\
        [--reference <tree>] [--domain drb_2yr ...]

The derived files are the regular files in each answer directory (one
with a .derived_okey); the oracle's answers are symlinks and are not
compared. Default reference: pywatershed's tree, the sibling clone's
test_data/. Kept until pywatershed is retired, to show the derive step
here still matches pywatershed's. Exit status 1 when anything differs.
"""

import argparse
import pathlib as pl
import sys

import numpy as np
import xarray as xr

REPO_ROOT = pl.Path(__file__).parents[2]
REFERENCE_DEFAULT = REPO_ROOT.parent / "pywatershed" / "test_data"

# Differences that are the reference's defects, each checked to still be
# exactly as described: every value equal except the first time, where
# the reference is nan and this tree is not.
_OBS_ET_INIT = (
    "pywatershed PRMSSoilzoneAgObsET._set_initial_conditions rebinds "
    "self.ssres_stor (prms_soilzone_ag_obs_et.py:548) instead of "
    "filling it, so its derive step reads the nan initial value"
)
KNOWN_DIFFERENCES = {
    "fgr_ag_2yr/output_analysis/ssres_stor_prev.nc": _OBS_ET_INIT,
    "fgr_ag_2yr/output_analysis/ssres_stor_change.nc": _OBS_ET_INIT,
}


def derived_files(out_dir: pl.Path) -> set[str]:
    return {
        pp.name
        for pp in out_dir.glob("*.nc")
        if pp.is_file() and not pp.is_symlink()
    }


def compare_file(path: pl.Path, ref_path: pl.Path) -> str | None:
    """None when equal, else what differs."""
    test = xr.load_dataset(path)
    ref = xr.load_dataset(ref_path)
    if set(test.data_vars) != set(ref.data_vars):
        return f"variables {sorted(test.data_vars)} vs {sorted(ref.data_vars)}"
    for name in ref.data_vars:
        aa, bb = test[name], ref[name]
        if aa.dims != bb.dims or aa.shape != bb.shape:
            return f"{name}: dims {aa.sizes} vs {bb.sizes}"
        for coord in bb.coords:
            if coord not in aa.coords or not np.array_equal(
                aa[coord].values, bb[coord].values
            ):
                return f"{name}: coordinate {coord} differs"
        if not np.array_equal(aa.values, bb.values, equal_nan=True):
            both_nan = np.isnan(aa.values) & np.isnan(bb.values)
            differs = (aa.values != bb.values) & ~both_nan
            diff = np.abs(aa.values - bb.values)[differs]
            finite = diff[~np.isnan(diff)]
            max_diff = f"{finite.max():.3e}" if finite.size else "none"
            return (
                f"{name}: values differ at {int(differs.sum())} of "
                f"{aa.size} (max abs diff {max_diff}; "
                f"{int(np.isnan(diff).sum())} nan in one only)"
            )
    return None


def check_reference_nan_at_start(
    path: pl.Path, ref_path: pl.Path
) -> str | None:
    """None when the files are equal except at the first time, where
    the reference is nan and path is not; else what differs."""
    test = xr.load_dataset(path)
    ref = xr.load_dataset(ref_path)
    for name in ref.data_vars:
        aa, bb = test[name].values, ref[name].values
        if not np.isnan(bb[0]).all() or np.isnan(aa[0]).any():
            return f"{name}: the first time is not nan only in the reference"
        if not np.array_equal(aa[1:], bb[1:], equal_nan=True):
            return f"{name}: values differ after the first time"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=pl.Path, required=True)
    parser.add_argument(
        "--reference",
        type=pl.Path,
        default=REFERENCE_DEFAULT,
        help=f"default: {REFERENCE_DEFAULT}",
    )
    parser.add_argument("--domain", action="append")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    reference = args.reference.expanduser().resolve()

    n_bad = 0
    n_same = 0
    n_known = 0
    out_dirs = sorted(pp.parent for pp in root.glob("*/*/.derived_okey"))
    if args.domain:
        out_dirs = [dd for dd in out_dirs if dd.parent.name in args.domain]
    if not out_dirs:
        raise SystemExit(f"{root}: no derived answer directories")
    for out_dir in out_dirs:
        rel = out_dir.relative_to(root)
        ref_dir = reference / rel
        if not ref_dir.is_dir():
            print(f"{rel}: NOT IN REFERENCE")
            n_bad += 1
            continue
        mine, theirs = derived_files(out_dir), derived_files(ref_dir)
        for name in sorted(theirs - mine):
            print(f"{rel}/{name}: missing here")
            n_bad += 1
        for name in sorted(mine - theirs):
            print(f"{rel}/{name}: not derived in the reference")
            n_bad += 1
        for name in sorted(mine & theirs):
            known = KNOWN_DIFFERENCES.get(f"{rel}/{name}")
            if known is not None:
                problem = check_reference_nan_at_start(
                    out_dir / name, ref_dir / name
                )
                if problem is None:
                    print(f"{rel}/{name}: known difference ({known})")
                    n_known += 1
                    continue
            else:
                problem = compare_file(out_dir / name, ref_dir / name)
            if problem is None:
                n_same += 1
            else:
                print(f"{rel}/{name}: {problem}")
                n_bad += 1
    print(
        f"{n_same} files identical, {n_known} known differences, "
        f"{n_bad} problems"
    )
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
