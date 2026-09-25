#!/bin/bash
#SBATCH --job-name=write_bw_measurement
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB


file_size=1G
test_file=./fio_benchmark_scratch.dat

# TEST:
# fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=randwrite --direct=1 --ioengine=libaio --runtime=30 --time_based --unlink=1

rm -f randwrite_bw_vs_size.txt
echo
echo "TESTING randwrite started at (next write stars at random position):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_${bs}.out 2>&1
    filefrag -v ${test_file}
    bw=$(cat randwrite_${bs}.out | grep WRITE | awk '{gsub("\\(","");gsub("kB\\/s\\),","");print $3;}')
    size=$(cat randwrite_${bs}.out | grep "iotest" | head -1 | awk '{gsub("B-"," ");print $5;}')
    echo "$size $bw" >> randwrite_bw_vs_size.txt
    
    sleep 2    
done
echo "TESTING randwrite complated at:"
date
sleep 30


rm -f write_bw_vs_size.txt
echo
echo "TESTING write started at (next write starts exectly where the previous finished):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_${bs}.out 2>&1
    filefrag -v ${test_file}
    bw=$(cat write_${bs}.out | grep WRITE | awk '{gsub("\\(","");gsub("kB\\/s\\),","");print $3;}')
    size=$(cat write_${bs}.out | grep "iotest" | head -1 | awk '{gsub("B-"," ");print $5;}')
    echo "$size $bw" >> write_bw_vs_size.txt
    
    sleep 2    
done
echo "TESTING write complated at:"
date
sleep 30


