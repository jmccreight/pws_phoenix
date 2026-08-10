#!/usr/bin/env bash
# Full incarnations/mpixarray test sweep: the serial suite, then each
# MPI test file under ${MPI_CMD}.
#
#   ./tests/run_tests.sh          # from incarnations/mpixarray (or anywhere)
#   ./tests/run_tests.sh -vv -x   # extra args REPLACE the default -q on
#                                 # every pytest invocation
#   MPI_CMD="srun -n 4" ./tests/run_tests.sh   # HPC (see mpix/detect_hpc.sh)
#
# MPI files run ONE PER LAUNCH (module-scoped fixtures stay isolated and
# a failure names its file). Exit code = number of failed suites.
set -uo pipefail

cd "$(dirname "$0")/.."

# ---- env-of-record check: WARN (never fail) if the active conda env
# is not the one pws_phoenix/environment.yaml defines (see CLAUDE.md
# "Environment"). Name-only comparison; HPC/CI contexts may
# legitimately differ, and an unset CONDA_DEFAULT_ENV skips the check.
expected_env=$(grep "^name:" ../../environment.yaml 2>/dev/null |
    head -n 1 | awk '{print $2}')
env_warning=""
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ -n "${expected_env}" ] &&
    [ "${CONDA_DEFAULT_ENV}" != "${expected_env}" ]; then
    env_warning="WARNING: active env '${CONDA_DEFAULT_ENV}' is not the \
env of record '${expected_env}' (pws_phoenix/environment.yaml) -- \
suggest: conda activate ${expected_env}"
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

echo "==== serial: pytest tests/ ${pytest_opts[*]}"
pytest tests/ "${pytest_opts[@]}" || failures=$((failures + 1))

for ff in tests/test_*_mpi.py; do
    echo
    echo "==== MPI: ${MPI_CMD} pytest --with-mpi ${ff} ${pytest_opts[*]}"
    ${MPI_CMD} pytest --with-mpi "${ff}" "${pytest_opts[@]}" \
        || failures=$((failures + 1))
done

echo
if [ "${failures}" -eq 0 ]; then
    echo "==== ALL SUITES PASSED"
else
    echo "==== ${failures} SUITE(S) FAILED"
fi
# repeat the env warning: the top of the output is long gone by now
if [ -n "${env_warning}" ]; then
    echo "${env_warning}"
fi
exit "${failures}"
