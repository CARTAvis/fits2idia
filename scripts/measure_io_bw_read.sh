#!/bin/bash

file_size=1G
test_file=./fio_benchmark_scratch.dat

# TEST:
# fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=randread --direct=1 --ioengine=libaio --runtime=30 --time_based --unlink=1

echo
echo "TESTING randread started at (next read stars at random position):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randread --direct=1 --ioengine=libaio --runtime=30 --time_based > randread_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randread --direct=1 --ioengine=libaio --runtime=30 --time_based > randread_${bs}.out 2>&1
    
    sleep 2    
done
echo "TESTING randread complated at:"
date
sleep 30


echo
echo "TESTING read started at (next read starts exectly where the previous finished):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=read --direct=1 --ioengine=libaio --runtime=30 --time_based > read_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=read --direct=1 --ioengine=libaio --runtime=30 --time_based > read_${bs}.out 2>&1
    
    sleep 2    
done
echo "TESTING read complated at:"
date
sleep 30


