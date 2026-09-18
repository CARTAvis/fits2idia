#!/bin/bash
#SBATCH --cpus-per-task=128
#SBATCH --mem=400GB
#SBATCH --job-name=fits2idia
#SBATCH --output=pawsey_submit_smart_converter_%x-%j.out
#SBATCH --error=pawsey_submit_smart_converter_%x-%j.err
#SBATCH --partition=highmem
#SBATCH --time=96:00:00

set -euo pipefail

# --- Default Parameters ---
fitsfile="image.restored.i.SB9992.contcube.linmos.13arcsec.leakage.zernike.holoI.Rotated-Axis3.fits"
max_mem_mb=31000
algorithm="channel"
approx_cube_histogram=0
use_ssd=1
work_dir="./"
copy=0

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -f, --file PATH        Path to FITS file to convert
                         (default: ${fitsfile})
  -m, --mem-mb INT       Memory limit in MB
                         (default: ${max_mem_mb})
  -M, --mem-gb INT       Memory limit in GB
                         (default: ${max_mem_mb} MB)
  -a, --algo STRING      Algorithm: channel, spatial, fast, or slow
                         (default: ${algorithm})
  -H, --approx-hist      Calculate approximate XYZ histogram (flag)
  -s, --no-ssd           Disable SSD partition usage (default: SSD enabled)
  -c, --copy             Copy input FITS file to temporary directory (default: disabled)
  -w, --work-dir PATH    Working directory to execute conversion in
                         (default: ${work_dir})
  -h, --help             Show this help message and exit

EOF
    exit 0
}

# --- Parse Command-Line Options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            fitsfile="$2"
            shift 2
            ;;
        -m|--mem-mb)
            max_mem_mb="$2"
            shift 2
            ;;
        -M|--mem-gb)
            max_mem_mb=$(( $2 * 1024 ))
            shift 2
            ;;
        -a|--algo)
            algorithm="$2"
            shift 2
            ;;
        -H|--approx-hist)
            approx_cube_histogram=1
            shift 1
            ;;
        -s|--no-ssd)
            use_ssd=0
            shift 1
            ;;
        -c|--copy)
            copy=1
            shift 1
            ;;
        -w|--work-dir)
            work_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            echo "Run '$(basename "$0") --help' for usage." >&2
            exit 1
            ;;
    esac
done

# --- Environment & Modules ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

module use /software/projects/ja3/setonix/2025.08/modules/zen3/gcc/14.2.0/
module load cfitsio/4.4.0 cray-hdf5/1.14.3.7 fits2idia/devel


if [[ -n "$work_dir" && "$work_dir" != "./" ]]; then
   cd "${work_dir}"
fi

base_fitsfile=$(basename "$fitsfile")
template_hdf5_file="${fitsfile%%fits}hdf5"

echo "-------------------------------------------------------------------------------------------------"
echo "PARAMETERS:"
echo "--------------------------"
echo "max_mem_mb            = $max_mem_mb"
echo "fitsfile              = $fitsfile"
echo "algorithm             = $algorithm"
echo "approx_cube_histogram = $approx_cube_histogram"
echo "OMP_NUM_THREADS       = $OMP_NUM_THREADS"
echo "use_ssd               = $use_ssd"
echo "copy                  = $copy"
echo "work_dir              = $work_dir"
echo "SLURM_CPUS_PER_TASK   = $SLURM_CPUS_PER_TASK (OMP_NUM_THREADS = $OMP_NUM_THREADS)"
echo "-------------------------------------------------------------------------------------------------"

date
pwd
start_dir=$(pwd)
temp_dir="./"

start_ux=`date +%s`

if [[ $use_ssd -gt 0 ]]; then
   temp_dir=$(mktemp -d ./tmp_dir_XXXXXX)
   # Generate the name locally
   # temp_dir=$(mktemp -u ./tmp_dir.XXXXXX)
   # Create the directory manually
   # mkdir -p "$temp_dir"
   
   echo "lfs setstripe --pool flash --stripe-count 10 --stripe-size 3G ${temp_dir}/"
   lfs setstripe --pool flash --stripe-count 10 --stripe-size 3G "${temp_dir}/"      

   if [[ $copy -gt 0 ]]; then      
      echo "cp ${fitsfile} ${temp_dir}/"
      cp ${fitsfile} ${temp_dir}/
   else
      echo "ln -sf ${fitsfile} ${temp_dir}/${fitsfile}"
      ln -sf ${fitsfile} ${temp_dir}/${fitsfile}
   fi
   
   echo "cd ${temp_dir}/"
   cd ${temp_dir}/
else
   echo "WARNING : SSD partition will not be used in conversion consider using this option!"
fi

if [[ $algorithm == "slow" ]]; then
   echo
   echo "-------------------------------------------------------------------------------------------------"
   echo "SLOW CONVERTER:"
   date

   outfile="${fitsfile%%.fits}_SLOW.hdf5"

   rm -f "$outfile"
   echo "srun fits2idia $fitsfile -o $outfile -p -s"
   srun fits2idia "$fitsfile" -o "$outfile" -p -s
   date

elif [[ $algorithm == "fast" ]]; then
   echo
   echo "-------------------------------------------------------------------------------------------------"
   echo "FAST CONVERTER:"
   date

   outfile="${fitsfile%%.fits}_FAST.hdf5"

   rm -f "$outfile"
   echo "srun fits2idia $fitsfile -o $outfile -p"
   srun fits2idia "$fitsfile" -o "$outfile" -p
   date

else
   echo
   echo "-------------------------------------------------------------------------------------------------"
   echo "SMART CONVERTER ALGORITHM: $algorithm"
   date
   
   options=""
   if [[ $algorithm == "channel" ]]; then
      options="-T channel"
   elif [[ $algorithm == "spatial" ]]; then   
      options="-T spatial"
   fi
   
   if [[ $approx_cube_histogram -gt 0 ]]; then
      options="$options -A"
   fi

   outfile="${fitsfile%%.fits}_SMARTALGO-${algorithm}_approx${approx_cube_histogram}.hdf5"

   rm -f "$outfile"
   echo "srun fits2idia $fitsfile -o $outfile -p -S -M ${max_mem_mb} -a ${options}"
   srun fits2idia "$fitsfile" -o "$outfile" -p -S -M "${max_mem_mb}" -a ${options}
   date        
fi

if [[ $use_ssd -gt 0 ]]; then
   echo "cd ${start_dir}"
   cd ${start_dir}

   echo "mv ${temp_dir}/${outfile} ${start_dir}"
   mv "${temp_dir}/${outfile}" "${start_dir}"
   
   echo "rm -fr ${temp_dir}/"
   rm -fr "${temp_dir}/"
fi

echo "-------------------------------------------------------------------------------------------------"
echo "Finished all at:"
date

end_ux=`date +%s`
total_time=$(($end_ux-$start_ux))
echo "Total execution time (including copying to and moving from SSD partition) took : $total_time [sec]"
echo "-------------------------------------------------------------------------------------------------"
