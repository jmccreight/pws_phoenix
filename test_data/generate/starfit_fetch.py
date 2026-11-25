#!/usr/bin/env python
"""Fetch pywatershed's STARFIT test data into <root>/starfit/.

    python test_data/generate/starfit_fetch.py [--root <dir>] [--force]

The data are tracked in pywatershed (test_data/starfit/, unchanged since
2.0.0, the release that introduced STARFIT) and published as an asset of
that release. The download is checked against the pinned sha256 and its
member list before anything is unpacked. Without --force an existing
<root>/starfit/ is left alone.
"""

import argparse
import hashlib
import pathlib as pl
import sys
import tarfile
import tempfile
import urllib.request

REPO_ROOT = pl.Path(__file__).parents[2]
TEST_DATA_DIR = REPO_ROOT / "test_data"

URL = (
    "https://github.com/DOI-USGS/pywatershed/releases/download/2.0.0/"
    "starfit_test_data.tar.gz"
)
SHA256 = "b461cbdfdf4b720f1d9047262391b2c4c470ec3a217e2cbbbe7f3e135aa11e0f"
MEMBERS = {
    "starfit",
    "starfit/istarf_conus_grand.nc",
    "starfit/lake_inflow.nc",
    "starfit/starfit_mean_output_1995-2001.nc",
    "starfit/starfit_original_parameters.nc",
}


def sha256_of(path: pl.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as ff:
        for block in iter(lambda: ff.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(root: pl.Path, force: bool) -> pl.Path:
    dest = root / "starfit"
    if dest.exists() and not force:
        print(f"{dest} exists; kept (--force to fetch again)")
        return dest
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = pl.Path(tmp) / "starfit_test_data.tar.gz"
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, tar_path)
        digest = sha256_of(tar_path)
        if digest != SHA256:
            raise SystemExit(
                f"{URL}: sha256 {digest} is not the pinned {SHA256}"
            )
        with tarfile.open(tar_path) as tf:
            names = set(tf.getnames())
            if names != MEMBERS:
                raise SystemExit(
                    f"{URL}: members {sorted(names)} are not the pinned "
                    f"{sorted(MEMBERS)}"
                )
            tf.extractall(root, filter="data")
    print(f"{dest}: {len(MEMBERS) - 1} files")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=pl.Path,
        default=TEST_DATA_DIR,
        help=f"default: {TEST_DATA_DIR}",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    fetch(args.root.expanduser().resolve(), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
