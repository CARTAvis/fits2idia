#!/bin/bash
#SBATCH --job-name=read_bw_measurement
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB


file_size=1G
test_file=./fio_benchmark_scratch.dat

filefrag -v ${test_file}

# TEST:
# fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=randread --direct=1 --ioengine=libaio --runtime=30 --time_based --unlink=1

rm -f randread_bw_vs_size.txt
echo
echo "TESTING randread started at (next read stars at random position):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_${bs}.out 2>&1
    bw=$(cat randread_${bs}.out | grep WRITE | awk '{gsub("\\(","");gsub("kB\\/s\\),","");print $3;}')
    size=$(cat randread_${bs}.out | grep "iotest" | head -1 | awk '{gsub("B-"," ");print $5;}')
    echo "$size $bw" >> randread_bw_vs_size.txt
    
    sleep 2    
done
echo "TESTING randread complated at:"
date
sleep 30


rm -f read_bw_vs_size.txt
echo
echo "TESTING read started at (next read starts exectly where the previous finished):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_${bs}.out 2>&1
    bw=$(cat read_${bs}.out | grep WRITE | awk '{gsub("\\(","");gsub("kB\\/s\\),","");print $3;}')
    size=$(cat read_${bs}.out | grep "iotest" | head -1 | awk '{gsub("B-"," ");print $5;}')
    echo "$size $bw" >> read_bw_vs_size.txt
    
    sleep 2    
done
echo "TESTING read complated at:"
date
sleep 30


