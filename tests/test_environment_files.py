"""The two environment files agree outside the parallel-io block.

environment.yaml (env pwpx) is the serial environment;
environment_mpi.yaml (env pwpx_mpi) is the same file plus the marked
`BEGIN/END parallel-io` block. Every shared pin must match, so the
MPI file minus that block and its `name:` line must equal the serial
file minus its `name:` line.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SERIAL_FILE = REPO_ROOT / "environment.yaml"
MPI_FILE = REPO_ROOT / "environment_mpi.yaml"


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _drop_block(lines: list[str], begin: str, end: str) -> list[str]:
    kept = []
    inside = False
    for line in lines:
        if begin in line:
            inside = True
        if not inside:
            kept.append(line)
        if end in line:
            inside = False
    return kept


def test_env_names() -> None:
    assert _lines(SERIAL_FILE)[0] == "name: pwpx"
    assert _lines(MPI_FILE)[0] == "name: pwpx_mpi"


def test_serial_file_has_no_parallel_block() -> None:
    text = SERIAL_FILE.read_text()
    assert "BEGIN parallel-io" not in text
    assert "END parallel-io" not in text


def test_mpi_file_has_blocks_once() -> None:
    text = MPI_FILE.read_text()
    for marker in (
        "BEGIN parallel-io",
        "END parallel-io",
        "BEGIN mpi-binaries",
        "END mpi-binaries",
    ):
        assert text.count(marker) == 1, marker


def test_mpi_file_is_serial_file_plus_block() -> None:
    mpi_lines = _drop_block(
        _lines(MPI_FILE), "BEGIN parallel-io", "END parallel-io"
    )
    assert mpi_lines[1:] == _lines(SERIAL_FILE)[1:]
