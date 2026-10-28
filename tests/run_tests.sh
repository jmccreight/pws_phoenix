#!/usr/bin/env bash
# Full pws_phoenix test sweep: the serial suite, then each
# MPI test file under ${MPI_CMD}.
#
#   ./tests/run_tests.sh          # from the repo root (or anywhere)
#   ./tests/run_tests.sh -vv -x   # extra args REPLACE the default -q on
#                                 # every pytest invocation
#   MPI_CMD="srun -n 4" ./tests/run_tests.sh   # HPC (see mpix/detect_hpc.sh)
#
# MPI files run ONE PER LAUNCH (module-scoped fixtures stay isolated and
# a failure names its file). Exit code = number of failed suites.
#
# NO SKIPS: every pytest invocation runs with --error-for-skips
# (pytest-error-for-skips, environment.yaml), so a skipped test fails
# its suite -- including the module-level skipifs of the
# data-dependent tests (pywatershed domain data missing = failure, not
# a quiet skip). The serial invocation ignores the *_mpi.py files
# (pytest-mpi would skip every test in them without --with-mpi); the
# loop below runs each of them under ${MPI_CMD}.
set -uo pipefail

cd "$(dirname "$0")/.."
start_s=${SECONDS}

# ---- env check: WARN (never fail) unless the active env's pws_phoenix
# imports from THIS clone (its editable install). The former name check
# against environment.yaml's `name:` broke with a second env per clone
# (pwpx2); HPC/CI contexts may legitimately differ.
installed=$(python -c "import pathlib, pws_phoenix; \
print(pathlib.Path(pws_phoenix.__file__).resolve().parents[1])" 2>/dev/null)
env_warning=""
if [ "${installed}" != "$(pwd -P)" ]; then
    env_warning="WARNING: active env '${CONDA_DEFAULT_ENV:-?}' imports \
pws_phoenix from '${installed:-<not importable>}', not this clone \
$(pwd -P) -- suggest: pip install -e . --no-deps in the right env"
    echo "${env_warning}"
    echo
fi

MPI_CMD=${MPI_CMD:-"mpirun -n 4"}
if [ "$#" -gt 0 ]; then
    pytest_opts=("$@")
else
    pytest_opts=(-q)
fi
failures=0

serial_opts=(--error-for-skips --ignore-glob='tests/*_mpi.py')
echo "==== serial: pytest tests/ ${serial_opts[*]} ${pytest_opts[*]}"
pytest tests/ "${serial_opts[@]}" "${pytest_opts[@]}" \
    || failures=$((failures + 1))

mpi_opts=(--with-mpi --error-for-skips)
for ff in tests/test_*_mpi.py; do
    echo
    echo "==== MPI: ${MPI_CMD} pytest ${mpi_opts[*]} ${ff} ${pytest_opts[*]}"
    ${MPI_CMD} pytest "${mpi_opts[@]}" "${ff}" "${pytest_opts[@]}" \
        || failures=$((failures + 1))
done

echo
elapsed=$(( SECONDS - start_s ))
elapsed="$(( elapsed / 60 ))m$(( elapsed % 60 ))s"
if [ "${failures}" -eq 0 ]; then
    echo "==== ALL SUITES PASSED (in ${elapsed})"
else
    echo "==== ${failures} SUITE(S) FAILED (in ${elapsed})"
fi
# repeat the env warning: the top of the output is long gone by now
if [ -n "${env_warning}" ]; then
    echo "${env_warning}"
fi
exit "${failures}"
