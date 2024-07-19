SIZES=(2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456)
mkdir results
for i in {1..10}; do

	rm results/gpu_$i.txt
	touch results/gpu_$i.txt
	for size in "${SIZES[@]}"; do
		echo $size
		./stepcounter $size | grep "Total time" |& tee -a results/gpu_$i.txt
	done


	rm results/cpu_$i.txt
	touch results/cpu_$i.txt
	for size in "${SIZES[@]}"; do
		echo $size
		./sequential $size | grep "Total time" |& tee -a results/cpu_$i.txt
	done
done
