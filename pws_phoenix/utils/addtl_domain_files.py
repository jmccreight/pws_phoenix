"""Fetch the additional example domains (fgr_2yr, fgr_ag_2yr,
sagehen_cbh_16yr).

These domains are too large to live in the repository; they are a
release asset (a zip) that this module downloads, checks, and unpacks
into ``pws_phoenix/data/`` (gitignored). Moved from pywatershed's
``utils/addtl_domain_files.py``; the asset is still pywatershed's
release asset until a data repository replaces it.

Command line: ``python -m pws_phoenix.utils.addtl_domain_files
[--force]``.
"""

import argparse
import hashlib
import pathlib as pl
import urllib.request as request
import zipfile
from shutil import rmtree

DATA_DIR = pl.Path(__file__).parents[1] / "data"
ADDTL_DOMAINS_DIR = DATA_DIR / "pywatershed_addtl_domains"
ADDTL_DOMAINS_ZIP = DATA_DIR / "pywatershed_addtl_domains.zip"

# URL and MD5 must be updated together when a new asset is released
ADDTL_DOMAINS_URL = (
    "https://github.com/DOI-USGS/pywatershed/"
    "releases/download/2.0.3/pywatershed_addtl_domains.zip"
)
ADDTL_DOMAINS_MD5 = "ae3b933fb733ea79818d67822cfb5d85"


def compute_md5(file_path: pl.Path) -> str:
    """MD5 checksum of a file, as a hex string."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as ff:
        for chunk in iter(lambda: ff.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _remove_all() -> None:
    if ADDTL_DOMAINS_DIR.exists():
        rmtree(ADDTL_DOMAINS_DIR)
    if ADDTL_DOMAINS_ZIP.exists():
        ADDTL_DOMAINS_ZIP.unlink()


def _fetch_zip() -> None:
    print(f"Downloading {ADDTL_DOMAINS_URL} to {ADDTL_DOMAINS_ZIP}")
    DATA_DIR.mkdir(exist_ok=True)
    request.urlretrieve(ADDTL_DOMAINS_URL, ADDTL_DOMAINS_ZIP)
    downloaded_md5 = compute_md5(ADDTL_DOMAINS_ZIP)
    if downloaded_md5 != ADDTL_DOMAINS_MD5:
        raise ValueError(
            "Downloaded file MD5 mismatch: "
            f"expected {ADDTL_DOMAINS_MD5}, got {downloaded_md5}"
        )


def _extract_zip() -> None:
    print(f"Extracting {ADDTL_DOMAINS_ZIP} to {DATA_DIR}")
    with zipfile.ZipFile(ADDTL_DOMAINS_ZIP, "r") as zz:
        zz.extractall(DATA_DIR)
    assert ADDTL_DOMAINS_DIR.exists()


def download_addtl_domains(force: bool = False) -> pl.Path:
    """Make the additional domains available locally; returns their
    root directory. A zip that is present but fails its checksum is
    removed together with anything unpacked from it; ``force``
    removes both first and re-downloads."""
    if force:
        _remove_all()
    elif ADDTL_DOMAINS_ZIP.exists():
        current_md5 = compute_md5(ADDTL_DOMAINS_ZIP)
        if current_md5 != ADDTL_DOMAINS_MD5:
            print(
                f"MD5 mismatch: expected {ADDTL_DOMAINS_MD5}, got "
                f"{current_md5}; removing {ADDTL_DOMAINS_DIR} and "
                f"{ADDTL_DOMAINS_ZIP}"
            )
            _remove_all()

    if not ADDTL_DOMAINS_ZIP.exists():
        _fetch_zip()
    if not ADDTL_DOMAINS_DIR.exists():
        _extract_zip()
    return ADDTL_DOMAINS_DIR


def get_addtl_domains_dir(domain: str | None = None) -> pl.Path:
    """The additional-domains root, or ``root / domain`` (e.g.
    ``"fgr_2yr"``). Does not download; see `download_addtl_domains`."""
    if domain is None:
        return ADDTL_DOMAINS_DIR
    return ADDTL_DOMAINS_DIR / domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the additional example domains"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove existing files first and re-download",
    )
    args = parser.parse_args()
    download_addtl_domains(force=args.force)
