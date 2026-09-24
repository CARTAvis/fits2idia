#!/bin/bash
set -euo pipefail

# --- fits2idia parameters (defaults) ---
fitsfile="IMAGE_CUBE.fits"
max_mem_mb=31000
extra_mem_mb=5000 # currently extra is 5GB but could also be 10 or 20 % 
algorithm="channel"
approx_cube_histogram=0
use_ssd=1
remove_ssd=1
work_dir="./"
copy=0
report=0
module="fits2idia/devel"
account="ja3"
extra_options=""

# --- SLURM resource parameters (defaults; these become sbatch flags) ---
cpus_per_task=128
mem="" # this is now calculated based the limit max_mem_mb (see above), but can be overwritten
job_name="fits2idia"
partition="highmem"
time_limit="96:00:00"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
worker_script="${script_dir}/fits2idia_worker.sbatch"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

fits2idia options:
  -f, --file PATH        Path to FITS file to convert
                         (default: ${fitsfile})
  -m, --mem-mb INT       fits2idia memory limit in MB, passed as -M
                         (default: ${max_mem_mb})
  -M, --mem-gb INT       Same, given in GB
                         (default: ${max_mem_mb} MB)
  -a, --algo STRING      Algorithm: channel, spatial, fast, or slow
                         (default: ${algorithm})
  -H, --approx-hist      Calculate approximate XYZ histogram (flag)
  -s, --no-ssd           Disable SSD partition usage (default: SSD enabled)
  -c, --copy             Copy input FITS file to temporary directory (default: disabled)
  -k, --keep             Keep the output file on SSD partition to avoid moving it (default: disabled) 
  -w, --work-dir PATH    Working directory to execute conversion in
                         (default: ${work_dir})
  -R, --report           Only reports on required memory and I/O execution time
  -L, --module           Load specific module
  -A, --account          Account (default: $account)
  -O, --options          Extra options of the converter (default: $extra_options)

SLURM resource options (these become sbatch flags for this submission):
  --cpus INT             --cpus-per-task for the job (default: ${cpus_per_task})
  --slurm-mem STR        --mem for the job, e.g. 400GB (default: $max_mem_mb + $extra_mem_mb [MB])
  --job-name STR         --job-name for the job (default: ${job_name})
  --partition STR        --partition for the job (default: ${partition})
  --time STR             --time for the job (default: ${time_limit})

  -h, --help             Show this help message and exit

EOF
    exit 0
}

# --- Parse Command-Line Options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            fitsfile="$2"; shift 2 ;;
        -m|--mem-mb)
            max_mem_mb="$2"; shift 2 ;;
        -M|--mem-gb)
            max_mem_mb=$(( $2 * 1024 )); shift 2 ;;
        -L|--module)
            module="$2"; shift 2 ;;
        -a|--algo)
            algorithm="$2"; shift 2 ;;
        -H|--approx-hist)
            approx_cube_histogram=1; shift 1 ;;
        -s|--no-ssd)
            use_ssd=0; shift 1 ;;
        -k|--keep)
            remove_ssd=0; shift 1 ;;
        -c|--copy)
            copy=1; shift 1 ;;
        -w|--work-dir)
            work_dir="$2"; shift 2 ;;
        -A|--account)
            account="$2"; shift 2 ;;
        -R|--report)
            report=1; shift 1 ;;
        -O|--options)
            extra_options="$2"; shift 2 ;;
        --cpus)
            cpus_per_task="$2"; shift 2 ;;
        --slurm-mem)
            mem="$2"; shift 2 ;;
        --job-name)
            job_name="$2"; shift 2 ;;
        --partition)
            partition="$2"; shift 2 ;;
        --time)
            time_limit="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            echo "Run '$(basename "$0") --help' for usage." >&2
            exit 1 ;;
    esac
done

# If --slurm-mem option was not provided calculate SLURM limit as max_mem_mb + extra_mem_mb:
slurm_mem_mb=$(($max_mem_mb+$extra_mem_mb))
if [[ ! -n "$mem" ]]; then
   mem=`echo $slurm_mem | awk '{printf("%.5fGB",$1);}'`
fi   

if [[ ! -f "$worker_script" ]]; then
    echo "Error: worker script not found at ${worker_script}" >&2
    exit 1
fi

echo "Submitting fits2idia job:"
echo "  fitsfile              = ${fitsfile}"
echo "  algorithm              = ${algorithm}"
echo "  max_mem_mb             = ${max_mem_mb}"
echo "  extra_mem_mb           = ${extra_mem_mb}"
echo "  approx_cube_histogram = ${approx_cube_histogram}"
echo "  use_ssd                = ${use_ssd}"
echo "  copy                   = ${copy}"
echo "  remove_ssd             = ${remove_ssd}"
echo "  work_dir               = ${work_dir}"
echo "  report                 = ${report}"
echo "  extra_options          = ${extra_options}"
echo "  --- SLURM resources ---"
echo "  cpus-per-task          = ${cpus_per_task}"
echo "  mem                    = ${mem}"
echo "  job-name               = ${job_name}"
echo "  partition              = ${partition}"
echo "  time                   = ${time_limit}"
echo "  account                = ${account}"
echo "  module                 = ${module}"

sbatch \
    --job-name="${job_name}" \
    --partition="${partition}" \
    --time="${time_limit}" \
    --cpus-per-task="${cpus_per_task}" \
    --mem="${mem}" \
    --account=${account} \
    --export=ALL,FITSFILE="${fitsfile}",MAX_MEM_MB="${max_mem_mb}",ALGORITHM="${algorithm}",APPROX_CUBE_HISTOGRAM="${approx_cube_histogram}",USE_SSD="${use_ssd}",WORK_DIR="${work_dir}",COPY="${copy}",REMOVE_SSD="${remove_ssd}",REPORT="${report}",MODULE="${module}",EXTRA_OPTIONS="${extra_options}" \
    "${worker_script}"
