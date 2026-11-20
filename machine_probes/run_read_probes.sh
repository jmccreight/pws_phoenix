#!/usr/bin/env bash
# Read probes for one machine and one domain: how long does a cold
# read of a (time, hru) forcing file take, per file layout, per rank
# count and per reader?
#
#   ./machine_probes/run_read_probes.sh screen
#   ./machine_probes/run_read_probes.sh matrix
#
#   screen  every reader ONCE at one rank count (the last of
#           RANKS_LIST): which readers differ on this machine?
#   matrix  the reader ModelMPI uses (mpio), at every rank count,
#           repeated. Add a reader the screen showed to matter with
#           READERS="mpio netCDF4".
#
# Activate the python environment first. Under Slurm, run it inside an
# allocation (salloc).
#
# One run of this script is one experiment, with one log directory:
#   logs/<cluster>/<domain>/<date_time>_<stage>/
#     experiment.txt                      the variables below, the commit
#     make_layout.<layout>.log
#     <layout>.r<ranks>.<reader>.<repeat>.log
#     summary.txt
#
# What it does:
#   1. makes each layout file that does not exist yet (make_layout.py)
#   2. for each repeat, reader, rank count and layout:
#        empties the file cache of that file on every node of the
#        allocation, checks it with fincore, stops if it is not empty;
#        then runs read_probe.py, which reads every variable twice
#        (first = cold, second = warm) in the one launch
#   3. prints the table of the logs (summarize.py)
#
# Every variable below can be set in the environment instead, e.g.
#   READERS="mpio rows" ./machine_probes/run_read_probes.sh matrix
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-${SLURM_CLUSTER_NAME:-$(hostname -s)}}"

# ---- the stage ------------------------------------------------------
# Readers not in either list (run them by setting READERS): xarray
# (Nov 20, 2026: the same as netCDF4) and step (one day of all hrus per
# read, what the serial file-backed Input does).
STAGE="${1:-screen}"
case "$STAGE" in
screen)
    : "${READERS:=mpio mpio-aligned mpio-collective netCDF4 rows}"
    : "${REPEATS:=1}"
    ;;
matrix)
    : "${READERS:=mpio}"
    : "${REPEATS:=3}"
    ;;
*)
    echo "run_read_probes.sh: the stage is screen or matrix, not '$STAGE'" >&2
    exit 1
    ;;
esac

# ---- per machine ----------------------------------------------------
# SOURCE_FILE   the file the layouts are copied from; empty = synthesize
# LAYOUT_DIR    where the layout files live (cbh_<layout name>.nc)
# RANKS_LIST    the rank counts of every reader but `step` (serial)
# LAUNCH_MPI    starts N processes; the script appends N
# LAUNCH_ONE    starts one process ON A COMPUTE NODE (make a layout,
#               the `step` reader)
# LAUNCH_NODES  starts one process on EVERY node of the allocation
#               (empty the file cache, fincore)
# EMPTY_CACHE   1 = empty and check the file cache before every launch
#               0 = do not (the first read is then of unknown state)
case "$CLUSTER" in
hovenweep*)
    : "${SOURCE_FILE:=/caldera/hovenweep/projects/usgs/water/impd/jmccreight/domains_pws_phoenix/conus_gf_v1.1_gridmet/cbh.nc}"
    : "${LAYOUT_DIR:=$(dirname "$SOURCE_FILE")}"
    : "${RANKS_LIST:=64 128}"
    : "${LAUNCH_MPI:=srun -c 1 --mem=0 -n}"
    : "${LAUNCH_ONE:=srun -N 1 -n 1 -c 1 --mem=0}"
    : "${LAUNCH_NODES:=srun -N ${SLURM_JOB_NUM_NODES:-1} --ntasks-per-node=1 --mem=0}"
    : "${EMPTY_CACHE:=1}"
    ;;
pathfinder*)
    # TODO: not known yet. Set SOURCE_FILE and LAYOUT_DIR once the file
    # is there; the launcher may be mpirun.
    : "${SOURCE_FILE:=}"
    : "${LAYOUT_DIR:=}"
    : "${RANKS_LIST:=64 128}"
    : "${LAUNCH_MPI:=srun -c 1 --mem=0 -n}"
    : "${LAUNCH_ONE:=srun -N 1 -n 1 -c 1 --mem=0}"
    : "${LAUNCH_NODES:=srun -N ${SLURM_JOB_NUM_NODES:-1} --ntasks-per-node=1 --mem=0}"
    : "${EMPTY_CACHE:=1}"
    ;;
*)
    # a laptop: a synthesized file, and the cache is not emptied (macOS
    # has neither posix_fadvise nor fincore)
    : "${SOURCE_FILE:=}"
    : "${LAYOUT_DIR:=$HERE/layouts}"
    : "${RANKS_LIST:=2 4}"
    : "${LAUNCH_MPI:=mpirun -n}"
    : "${LAUNCH_ONE:=}"
    : "${LAUNCH_NODES:=}"
    : "${EMPTY_CACHE:=0}"
    ;;
esac
if [[ "$STAGE" == screen ]]; then
    RANKS_LIST="${RANKS_LIST##* }"
fi

# ---- the measurement ------------------------------------------------
# only when SOURCE_FILE is empty: the size of the synthesized file
: "${SYNTH_NTIME:=3653}"
: "${SYNTH_NHRU:=20000}"
# the name of the domain in the log path
if [[ -n "$SOURCE_FILE" ]]; then
    : "${DOMAIN:=$(basename "$(dirname "$SOURCE_FILE")")}"
else
    : "${DOMAIN:=synthetic_${SYNTH_NTIME}x${SYNTH_NHRU}}"
fi
# One layout per word: name:stripes:stripe_size:arrangement
#   stripes, stripe_size  lfs setstripe -c / -S (ignored off Lustre)
#   arrangement           contiguous  (time, hru), no chunks
#                         blocks<N>   (time, hru) in chunks of all days
#                                     x ceil(number of hrus / N) hrus:
#                                     one chunk per rank at N ranks,
#                                     for any domain
#                         transposed  (hru, time), no chunks
# A layout whose file exists is used as it is (cbh_striped8.nc is the
# copy kept from Nov 20, 2026).
: "${LAYOUTS:=striped8:8:4M:contiguous striped8_blocks128:8:4M:blocks128 striped8_transposed:8:4M:transposed}"
# the `step` reader stops after this many
# seconds per variable and reports the days it got through
: "${STEP_MAX_SECONDS:=300}"
# stop when more than this percent of a file is still in a node's file
# cache after emptying
: "${MAX_CACHED_PERCENT:=1}"

LOG_DIR="${LOG_DIR:-$HERE/logs/$CLUSTER/$DOMAIN/$(date +%Y-%m-%d_%H%M%S)_$STAGE}"

# ---- checks ---------------------------------------------------------
stop() {
    echo "run_read_probes.sh: $*" >&2
    exit 1
}

[[ -n "$LAYOUT_DIR" ]] || stop "LAYOUT_DIR is not set for cluster '$CLUSTER'"
if [[ -n "$SOURCE_FILE" && ! -e "$SOURCE_FILE" ]]; then
    stop "SOURCE_FILE does not exist: $SOURCE_FILE"
fi
if [[ "$LAUNCH_MPI" == srun* && -z "${SLURM_JOB_ID:-}" ]]; then
    stop "not inside a Slurm allocation: salloc first"
fi
if [[ "$EMPTY_CACHE" == 1 ]]; then
    $LAUNCH_NODES sh -c 'command -v fincore' >/dev/null ||
        stop "no fincore on a compute node: the emptied cache cannot" \
            "be checked (EMPTY_CACHE=0 runs without it)"
fi
# fail now, not an hour in, when the allocation cannot hold a rank count
for ranks in $RANKS_LIST; do
    $LAUNCH_MPI "$ranks" true ||
        stop "cannot start $ranks processes with: $LAUNCH_MPI $ranks"
done

layout_file() { echo "$LAYOUT_DIR/cbh_$1.nc"; }

# Empty the file cache of one file on every node, then check each node:
# stop when more than MAX_CACHED_PERCENT of the file is still cached.
empty_cache() {
    local file="$1" cached size nodes=0
    $LAUNCH_NODES python "$HERE/read_probe.py" drop-cache "$file"
    while read -r cached size; do
        nodes=$((nodes + 1))
        echo "# cache check: $cached of $size bytes cached: $file"
        if ((cached * 100 > size * MAX_CACHED_PERCENT)); then
            stop "the file cache was not emptied: $file"
        fi
    done < <(
        $LAUNCH_NODES fincore --bytes --noheadings --output RES,SIZE "$file"
    )
    [[ "$nodes" -gt 0 ]] || stop "fincore printed nothing for: $file"
}

mkdir -p "$LAYOUT_DIR" "$LOG_DIR"
{
    echo "date:         $(date)"
    echo "cluster:      $CLUSTER"
    echo "domain:       $DOMAIN"
    echo "commit:       $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "slurm job:    ${SLURM_JOB_ID:-none}, nodes ${SLURM_JOB_NODELIST:-none}"
    for var in STAGE SOURCE_FILE LAYOUT_DIR LAYOUTS READERS RANKS_LIST REPEATS \
        LAUNCH_MPI LAUNCH_ONE LAUNCH_NODES EMPTY_CACHE MAX_CACHED_PERCENT \
        STEP_MAX_SECONDS SYNTH_NTIME SYNTH_NHRU; do
        printf '%-13s %s\n' "$var:" "${!var}"
    done
} | tee "$LOG_DIR/experiment.txt"

# ---- 1. make the missing layouts ------------------------------------
for spec in $LAYOUTS; do
    IFS=: read -r name stripes stripe_size arrangement <<<"$spec"
    file="$(layout_file "$name")"
    if [[ -e "$file" ]]; then
        echo "layout exists, used as it is: $file"
        continue
    fi
    if [[ -n "$SOURCE_FILE" ]]; then
        from="--source $SOURCE_FILE"
    else
        from="--synthesize $SYNTH_NTIME $SYNTH_NHRU"
    fi
    $LAUNCH_ONE python "$HERE/make_layout.py" $from --out "$file" \
        --stripes "$stripes" --stripe-size "$stripe_size" \
        --arrangement "$arrangement" \
        2>&1 | tee "$LOG_DIR/make_layout.$name.log"
done

# ---- 2. the reads ---------------------------------------------------
# Layouts alternate innermost, so a change of the machine during the
# run does not land on one layout only.
for repeat in $(seq 1 "$REPEATS"); do
    for reader in $READERS; do
        if [[ "$reader" == step ]]; then
            rank_counts=1
        else
            rank_counts="$RANKS_LIST"
        fi
        for ranks in $rank_counts; do
            if [[ "$reader" == step ]]; then
                launch="$LAUNCH_ONE"
                extra="--max-seconds $STEP_MAX_SECONDS"
            else
                launch="$LAUNCH_MPI $ranks"
                extra=""
            fi
            for spec in $LAYOUTS; do
                name="${spec%%:*}"
                file="$(layout_file "$name")"
                log="$LOG_DIR/$name.r$ranks.$reader.$repeat.log"
                : >"$log"
                if [[ "$EMPTY_CACHE" == 1 ]]; then
                    empty_cache "$file" 2>&1 | tee -a "$log"
                fi
                # read_probe.py prints the `#` header (host, cluster,
                # domain, layout, file system, stripes, chunks, package
                # versions, launcher), then one line per rank, variable
                # and attempt in the format of the Nov 20, 2026 logs
                $launch python "$HERE/read_probe.py" read "$file" \
                    --reader "$reader" --cluster "$CLUSTER" \
                    --domain "$DOMAIN" --layout "$name" \
                    --launcher "$launch" --cache-emptied "$EMPTY_CACHE" \
                    $extra 2>&1 | tee -a "$log"
            done
        done
    done
done

# ---- 3. the table ---------------------------------------------------
python "$HERE/summarize.py" "$LOG_DIR" | tee "$LOG_DIR/summary.txt"
