# Step Counter in CUDA

- `stepcounter.cu` - Parallel version
- `sequential.cpp` - Sequential version
- Accelerometer input data can be collected with the [senslogs](https://github.com/tyrex-team/senslogs) application

## Compiling

### NVIDIA

```bash
nvcc stepcounter.cu -o stepcounter
```

### AMD

```bash
hipify-clang stepcounter.cu -o stepcounter.cpp && hipcc stepcounter.cpp -o stepcounter
```

## Benchmarking

> [!TIP]
> Real-life sample data can be found in accelerometer.txt

### Fixed data size

`./stepcounter accelerometer.txt` or `./stepcounter 10000`

### Variable data size

See `run_tests.sh` and `run_tests_rocprof.sh`

