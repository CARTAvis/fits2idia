#!/bin/bash
#SBATCH --job-name=fits2idia
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=highmem
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=400GB
#
# This file is a normal, static sbatch script. All of the flags above are
# defaults only -- the launcher (submit_fits2idia.sh) overrides them on the
# `sbatch` command line for each run, and passes the fits2idia-specific
# parameters below through --export. You can also `sbatch` this file
# directly for a quick manual run; anything not exported just falls back
# to the defaults baked in here.

set -euo pipefail

# --- Parameters (come from the launcher's --export, with fallbacks) ---
fitsfile="${FITSFILE:?FITSFILE was not exported by the launcher}"
max_mem_mb="${MAX_MEM_MB:-31000}"
algorithm="${ALGORITHM:-channel}"
approx_cube_histogram="${APPROX_CUBE_HISTOGRAM:-0}"
use_ssd="${USE_SSD:-1}"
remove_ssd="${REMOVE_SSD:1}"
work_dir="${WORK_DIR:-./}"
copy="${COPY:-0}"

# --- Environment & Modules ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

module use /software/projects/ja3/setonix/2025.08/modules/zen3/gcc/14.2.0/
module load cfitsio/4.4.0 cray-hdf5/1.14.3.7 fits2idia/devel

if [[ -n "$work_dir" && "$work_dir" != "./" ]]; then
   cd "${work_dir}"
fi

base_fitsfile=$(basename "$fitsfile")
fullpath_fitsfile=$(realpath "$fitsfile")
template_hdf5_file="${fitsfile%%fits}hdf5"

echo "-------------------------------------------------------------------------------------------------"
echo "PARAMETERS:"
echo "--------------------------"
echo "max_mem_mb            = $max_mem_mb"
echo "fitsfile              = $fitsfile ($fullpath_fitsfile)"
echo "algorithm              = $algorithm"
echo "approx_cube_histogram = $approx_cube_histogram"
echo "OMP_NUM_THREADS       = $OMP_NUM_THREADS"
echo "use_ssd               = $use_ssd"
echo "remove_ssd            = $remove_ssd"
echo "copy                  = $copy"
echo "work_dir              = $work_dir"
echo "SLURM_CPUS_PER_TASK   = ${SLURM_CPUS_PER_TASK:-} (OMP_NUM_THREADS = $OMP_NUM_THREADS)"
echo "-------------------------------------------------------------------------------------------------"

date
pwd
start_dir=$(pwd)
temp_dir="./"

start_ux=`date +%s`

if [[ $use_ssd -gt 0 ]]; then
   temp_dir=$(mktemp -d ./tmp_dir_XXXXXX)

   echo "lfs setstripe --pool flash --stripe-count 10 --stripe-size 3G ${temp_dir}/"
   lfs setstripe --pool flash --stripe-count 10 --stripe-size 3G "${temp_dir}/"

   echo "cd ${temp_dir}/"
   cd ${temp_dir}/

   if [[ $copy -gt 0 ]]; then
      echo "cp ${fullpath_fitsfile} ."
      cp ${fullpath_fitsfile} .
   else
      echo "ln -sf ${fullpath_fitsfile}"
      ln -sf ${fullpath_fitsfile}
   fi
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

   if [[ $remove_ssd -gt 0 ]]; then
      echo "mv ${temp_dir}/${outfile} ${start_dir}"
      mv "${temp_dir}/${outfile}" "${start_dir}"

      echo "rm -fr ${temp_dir}/"
      rm -fr "${temp_dir}/"
   else
      echo "WARNING: Keeping outfile $outfile on SSD partition. Please do not overuse it"
   fi
fi

echo "-------------------------------------------------------------------------------------------------"
echo "Finished all at:"
date

end_ux=`date +%s`
total_time=$(($end_ux-$start_ux))
echo "Total execution time (including copying to and moving from SSD partition) took : $total_time [sec]"
echo "-------------------------------------------------------------------------------------------------"
