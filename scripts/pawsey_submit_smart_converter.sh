#!/bin/bash
#SBATCH --cpus-per-task=128
#SBATCH --mem=400GB
#SBATCH --job-name=fits2idia
#SBATCH --output=pawsey_submit_smart_converter_%x-%j.out
#SBATCH --error=pawsey_submit_smart_converter_%x-%j.err
#SBATCH --partition=highmem
#SBATCH --time=96:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module use TODO 
module load cfitsio/4.4.0 cray-hdf5/1.14.3.7 fits2idia-msok

# FITS FILE TO CONVERT:
fitsfile=image.restored.i.SB9992.contcube.linmos.13arcsec.leakage.zernike.holoI.Rotated-Axis3.fits
if [[ -n "$1" && "$1" != "-" ]]; then
   fitsfile="$1"
fi
base_fitsfile=`basename $fitsfile`
template_hdf5_file=${fitsfile%%fits}hdf5

# limit in MB (default 31 GB)
max_mem_mb=31000
if [[ -n "$2" && "$2" != "-" ]]; then
   max_mem_mb=$2
fi

# slow, channel, spatial, fast
algorithm="channel"
if [[ -n "$3" && "$3" != "-" ]]; then
   algorithm="$3"
fi

# calculate approximate XYZ histogram
approx_cube_histogram=0
if [[ -n "$4" && "$4" != "-" ]]; then
   approx_cube_histogram=$4
fi

use_ssd=1
if [[ -n "$5" && "$5" != "-" ]]; then
   use_ssd=$5
fi

work_dir="./"
if [[ -n "$6" && "$6" != "-" ]]; then
   work_dir="$6"
   cd ${work_dir}
fi

echo "-------------------------------------------------------------------------------------------------"
echo "PARAMETERS:"
echo "--------------------------"
echo "max_mem_mb = $max_mem_mb"
echo "fitsfile   = $fitsfile"
echo "algorithm  = $algorithm"
echo "approx_cube_histogram = $approx_cube_histogram"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "use_ssd = $use_ssd"
echo "work_dir = $work_dir"
echo "--------------------------"

date
pwd
start_dir=`pwd`
temp_dir="./"

if [[ $use_ssd -gt 0 ]]; then
   temp_dir=$(mktemp -d)
   mkdir -p ${temp_dir}
   
   echo "lfs setstripe --pool flash --stripe-count 10 --stripe-size 3G ${temp_dir}/"
   lfs setstripe --pool flash --stripe-count 10 --stripe-size 3G ${temp_dir}/
else
   echo "WARNING : SSD partition will not be used in conversion consider using this option!"
fi

if [[ $algorithm == "slow" ]]; then
   echo
   echo "-------------------------------------------------------------------------------------------------"
   echo "SLOW CONVERTER:"
   date

   outfile=${fitsfile%%.fits}_SLOW.hdf5

   date
   echo "rm -f $outfile"
   rm -f $outfile
   date

   # -m 
   echo "srun fits2idia $fitsfile -o $outfile -p -s"
   srun fits2idia $fitsfile -o $outfile -p -s
   date
elif [[ $algorithm == "fast" ]]; then
   echo
   echo "-------------------------------------------------------------------------------------------------"
   echo "FAST CONVERTER:"
   date

   outfile=${fitsfile%%.fits}_FAST.hdf5

   date
   echo "rm -f $outfile"
   rm -f $outfile
   date

   # -m 
   echo "srun fits2idia $fitsfile -o $outfile -p"
   srun fits2idia $fitsfile -o $outfile -p
   date

else
   echo
   echo "-------------------------------------------------------------------------------------------------"
   echo "SMART CONVERTER ALGORITHM: $algorithm"
   date
   
   options=""
   if [[ $algorithm == "channel" ]]; then
      options="-T fast"
   elif [[ $algorithm == "spatial" ]]; then   
      options="-T spatial"
   fi
   
   if [[ $approx_cube_histogram -gt 0 ]]; then
      options="$options -A"
   fi

   outfile=${fitsfile%%.fits}_SMARTALGO-${algorithm}_approx${approx_cube_histogram}.hdf5

   date
   echo "rm -f $outfile"
   rm -f $outfile
   date

   echo "srun fits2idia $fitsfile -o $outfile -p -S -M ${max_mem_mb} -a ${options}"
   srun fits2idia $fitsfile -o $outfile -p -S -M ${max_mem_mb} -a ${options}
   date        
fi

if [[ $use_ssd -gt 0 ]]; then
   echo "mv ${temp_dir}/${outfile} ${start_dir}"
   mv ${temp_dir}/${outfile} ${start_dir}
   
   echo "rm -fr ${temp_dir}/"
# TODO : uncomment when I am sure it works and will not remove anything wrong !!!
#   rm -fr ${temp_dir}/
fi


echo "-------------------------------------------------------------------------------------------------"
echo "Finished all at:"
date
