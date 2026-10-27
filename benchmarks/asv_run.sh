#!/usr/bin/env bash
# Benchmark a series of commits in the FIXED current env -- the
# hand-rolled `asv continuous` for ASV's existing-environment mode,
# which refuses commit ranges (benchmarks/README.md section 1).
#
#   benchmarks/asv_run.sh <hash> [<hash> ...] [-- <extra asv run args>]
#
# For each hash, in order: `git checkout <hash>`, `asv run
# --set-commit-hash <hash>` (results saved under .asv/results/), and
# `conda list --export` into .asv/env_lists/<hash>.txt (the env is
# NOT recorded by ASV in this mode; this is the dependency-drift
# record). The original checkout is restored at the end, also on
# error. With two or more hashes, finishes with `asv compare` of the
# first and last.
#
# Requires a CLEAN working tree (the checkouts would otherwise carry
# or clobber local edits). Run from anywhere; operates on the repo
# that contains this script.
set -euo pipefail

cd "$(dirname "$0")/.."

hashes=()
extra_args=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        extra_args=("$@")
        break
    fi
    hashes+=("$1")
    shift
done
if [[ ${#hashes[@]} -eq 0 ]]; then
    echo "usage: $0 <hash> [<hash> ...] [-- <extra asv run args>]" >&2
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree is not clean; commit or stash first" >&2
    exit 1
fi

# branch name if on one, else the detached commit
orig=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
restore() { git checkout -q "$orig"; }
trap restore EXIT

mkdir -p .asv/env_lists
full=()
for hh in "${hashes[@]}"; do
    hash=$(git rev-parse --verify "${hh}^{commit}")
    full+=("$hash")
    echo "==> $hh ($hash)"
    git checkout -q "$hash"
    asv run --show-stderr --set-commit-hash "$hash" "${extra_args[@]}"
    conda list --export > ".asv/env_lists/${hash}.txt"
done

if [[ ${#full[@]} -ge 2 ]]; then
    asv compare --factor 1.1 "${full[0]}" "${full[-1]}"
fi
