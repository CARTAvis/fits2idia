#!/bin/bash
#SBATCH --job-name=read_bw_measurement
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB

# for SETONIX ONLY:
export PATH=/software/projects/mwasci/msok/fio/fio:$PATH

file_size=1G
test_file=./fio_benchmark_scratch.dat

filefrag -v ${test_file}

# TEST:
# fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=randread --direct=1 --ioengine=libaio --runtime=30 --time_based --unlink=1

rm -f randread_bw_vs_size.txt
echo
echo "TESTING randread started at (next read stars at random position):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M 256M 512M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_${bs}.out 2>&1
    bw=$(cat randread_${bs}.out | grep READ | awk '{gsub("\\(","");print $3;}')
    size=$(cat randread_${bs}.out | grep "iotest" | head -1 | awk '{gsub("B-"," ");print $5;}')
    echo "$size $bw" >> randread_bw_vs_size.txt
    
    sleep 2    
done

cat randread_bw_vs_size.txt | awk '{if(NR==1){gsub("4096","4*1024",$1);}gsub("Ki","*1024",$1);gsub("Mi","*1024*1024",$1);gsub("\\.0","",$1);$1=$1;gsub("MB/s),","e6",$2);if(index($2,"kB/s")>0){gsub("kB/s),","",$2);$2=$2/1000.00;$2=$2"e6";}print "{ "$1" , "$2" },";}' > randread_bw_vs_size.cc
cat randread_bw_vs_size.txt | awk '{if($1~/Ki/){sub("Ki","",$1);$1=$1*1024}else if($1~/Mi/){sub("Mi","",$1);$1=$1*1024*1024}; gsub("MB/s),","",$2); if(index($2,"kB/s")>0){gsub("kB/s),","",$2);$2=$2/1000.00}; print $1" "$2;}' > randread_bw_vs_size_PLOT.txt
echo "TESTING randread complated at:"
date
sleep 30


rm -f read_bw_vs_size.txt
echo
echo "TESTING read started at (next read starts exectly where the previous finished):"
date
for bs in `echo "4k 64k 1M 16M 32M 64M 128M 256M 512M"`
do
    echo "rm -f ${test_file}"
    rm -f ${test_file}

    # keeping the file for read tests -> no option --unlink=1
    echo "fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_${bs}.out 2>&1"
    fio --name=iotest --filename=${test_file} --size=${file_size} --bs=${bs} --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_${bs}.out 2>&1
    bw=$(cat read_${bs}.out | grep READ | awk '{gsub("\\(","");print $3;}')
    size=$(cat read_${bs}.out | grep "iotest" | head -1 | awk '{gsub("B-"," ");print $5;}')
    echo "$size $bw" >> read_bw_vs_size.txt
    
    sleep 2    
done

cat read_bw_vs_size.txt | awk '{if(NR==1){gsub("4096","4*1024",$1);}gsub("Ki","*1024",$1);gsub("Mi","*1024*1024",$1);gsub("\\.0","",$1);$1=$1;gsub("MB/s),","e6",$2);if(index($2,"kB/s")>0){gsub("kB/s),","",$2);$2=$2/1000.00;$2=$2"e6";}print "{ "$1" , "$2" },";}' > read_bw_vs_size.cc
cat read_bw_vs_size.txt | awk '{if($1~/Ki/){sub("Ki","",$1);$1=$1*1024}else if($1~/Mi/){sub("Mi","",$1);$1=$1*1024*1024}; gsub("MB/s),","",$2); if(index($2,"kB/s")>0){gsub("kB/s),","",$2);$2=$2/1000.00}; print $1" "$2;}' > read_bw_vs_size_PLOT.txt
echo "TESTING read complated at:"
date
sleep 30


