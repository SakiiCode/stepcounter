SIZES=(2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456)
rm -r results_rocprof
mkdir results_rocprof
for i in {1..10}; do
	for size in "${SIZES[@]}"; do
		touch results_rocprof/gpu_$size.csv
		echo $size
		rocprof --stats stepcounter $size
		cat results.stats.csv >> results_rocprof/gpu_$size.csv
	done
done
