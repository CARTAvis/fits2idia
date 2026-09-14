#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --job-name=h5diff
#SBATCH --output=pawsey_h5diff_%x-%j.out
#SBATCH --error=pawsey_h5diff_%x-%j.err
#SBATCH --time=24:00:00

file1="file1.hdf5"
if [[ -n "$1" && "$1" != "-" ]]; then
   file1=$1
fi

file2="file2.hdf5"
if [[ -n "$2" && "$2" != "-" ]]; then
   file2=$2
fi


module load cray-hdf5/1.14.3.7


echo "-------------------------------------------------------------------------------------------------"
echo "PARAMETERS:"
echo "--------------------------"
echo "--------------------------"

date
pwd

echo "srun h5diff -r $file1 $file2"
srun h5diff -r $file1 $file2

echo "Finished all at:"
date
