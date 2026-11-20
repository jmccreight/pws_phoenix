"""Table of read-probe logs.

    python summarize.py LOG_DIRECTORY_OR_FILE [...]

Reads the `rank ...` lines read_probe.py prints (and the Nov 20, 2026
Hovenweep logs, which have the same fields). One log file is one
launch. Per launch and per (file, ranks, reader, variable, attempt):

  open      the slowest rank's open, seconds
  fastest / mean / slowest
            the read, seconds, over the ranks. The slowest rank is the
            time of the launch: a model waits for it.
  MB/s      megabytes read by all ranks / the slowest rank's seconds

Launches of the same case (repeats) are then joined: every column is
the MEDIAN over the launches, and `slowest range` is the least and the
greatest slowest-rank time among them. The variable `(all)` adds up
the variables of a launch.

A `*` marks a case where a launch printed fewer lines than it has
ranks (it did not finish).
"""

import argparse
import pathlib as pl
import statistics
from collections import defaultdict

ALL_VARIABLES = "(all)"
ATTEMPT_ORDER = {"first": 0, "second": 1}
COLUMNS = (
    "file",
    "ranks",
    "reader",
    "variable",
    "attempt",
    "launches",
    "open",
    "fastest",
    "mean",
    "slowest",
    "slowest range",
    "MB/s",
)

# (megabytes, open seconds, read seconds) of one rank
RankRead = tuple[float, float, float]
# (file, ranks, reader, variable, attempt)
Case = tuple[str, int, str, str, str]


def log_files(paths: list[pl.Path]) -> list[pl.Path]:
    files: list[pl.Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    pp for pp in path.iterdir() if pp.suffix.lower() == ".log"
                )
            )
        else:
            files.append(path)
    return files


def parse_log(path: pl.Path) -> dict[Case, list[RankRead]]:
    """The rank lines of one launch, by case. The fields, counted from
    0: 3 ranks, 5 reader, 6 variable, 7 attempt, 12 megabytes, 15 open
    seconds, 17 read seconds, 20 file name."""
    reads: dict[Case, list[RankRead]] = defaultdict(list)
    with open(path, errors="replace") as log:
        for line in log:
            parts = line.split()
            if len(parts) != 21 or parts[0] != "rank" or parts[2] != "of":
                continue
            case = (parts[20], int(parts[3]), parts[5], parts[6], parts[7])
            reads[case].append(
                (float(parts[12]), float(parts[15]), float(parts[17]))
            )
    return reads


def launch_statistics(
    reads: dict[Case, list[RankRead]],
) -> dict[Case, dict[str, float]]:
    """Per case of one launch: the columns, plus `complete` (1.0 when
    every rank printed its line)."""
    stats: dict[Case, dict[str, float]] = {}
    for case, rank_reads in reads.items():
        megabytes = [rr[0] for rr in rank_reads]
        seconds = [rr[2] for rr in rank_reads]
        stats[case] = {
            "megabytes": sum(megabytes),
            "open": max(rr[1] for rr in rank_reads),
            "fastest": min(seconds),
            "mean": statistics.fmean(seconds),
            "slowest": max(seconds),
            "complete": float(len(rank_reads) == case[1]),
        }
    # the variables of the launch added up
    totals: dict[Case, dict[str, float]] = {}
    for case, stat in stats.items():
        file_name, n_ranks, reader, _, attempt = case
        total_case = (file_name, n_ranks, reader, ALL_VARIABLES, attempt)
        total = totals.setdefault(
            total_case,
            {
                "megabytes": 0.0,
                "open": 0.0,
                "fastest": 0.0,
                "mean": 0.0,
                "slowest": 0.0,
                "complete": 1.0,
            },
        )
        for key in ("megabytes", "fastest", "mean", "slowest"):
            total[key] += stat[key]
        total["open"] = max(total["open"], stat["open"])
        total["complete"] = min(total["complete"], stat["complete"])
    stats.update(totals)
    return stats


def table_rows(files: list[pl.Path]) -> list[tuple[str, ...]]:
    launches: dict[Case, list[dict[str, float]]] = defaultdict(list)
    for path in files:
        for case, stat in launch_statistics(parse_log(path)).items():
            launches[case].append(stat)

    def order(case: Case) -> tuple:
        file_name, n_ranks, reader, variable, attempt = case
        return (
            file_name,
            n_ranks,
            reader,
            variable == ALL_VARIABLES,
            variable,
            ATTEMPT_ORDER.get(attempt, 2),
        )

    rows: list[tuple[str, ...]] = []
    for case in sorted(launches, key=order):
        stats = launches[case]
        file_name, n_ranks, reader, variable, attempt = case

        def median(key: str) -> float:
            return statistics.median(stat[key] for stat in stats)

        slowest = [stat["slowest"] for stat in stats]
        # a read too short for the log's digits has no rate
        rates = [
            stat["megabytes"] / stat["slowest"]
            for stat in stats
            if stat["slowest"] > 0.0
        ]
        rate = f"{statistics.median(rates):.1f}" if rates else "-"
        complete = all(stat["complete"] == 1.0 for stat in stats)
        rows.append(
            (
                file_name,
                str(n_ranks),
                reader,
                variable,
                attempt,
                f"{len(stats)}{'' if complete else '*'}",
                f"{median('open'):.2f}",
                f"{median('fastest'):.2f}",
                f"{median('mean'):.2f}",
                f"{median('slowest'):.2f}",
                f"{min(slowest):.2f}-{max(slowest):.2f}",
                rate,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table of read-probe logs; see the module docstring."
    )
    parser.add_argument("paths", nargs="+", type=pl.Path)
    args = parser.parse_args()

    files = log_files(args.paths)
    rows = table_rows(files)
    if not rows:
        raise SystemExit(
            f"summarize: no `rank` lines in {len(files)} log file(s)"
        )
    widths = [
        max(len(row[cc]) for row in [COLUMNS] + rows)
        for cc in range(len(COLUMNS))
    ]
    text_columns = 5  # left-justified; the numbers are right-justified
    for row in [COLUMNS] + rows:
        cells = [
            cell.ljust(width) if cc < text_columns else cell.rjust(width)
            for cc, (cell, width) in enumerate(zip(row, widths))
        ]
        print("  ".join(cells).rstrip())
    print()
    print("seconds; every number is the median over the launches of a case")
    if any(row[5].endswith("*") for row in rows):
        print("* a launch printed fewer lines than it has ranks")


if __name__ == "__main__":
    main()
