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
exit "${failures}"
