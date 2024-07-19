#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define MEASURE_TIME

#define cudaCheckErrors(call)                                 \
  do                                                          \
  {                                                           \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess)                                   \
    {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

struct vector
{
  float x;
  float y;
  float z;
};

// previously the work was unneccessarily split into 3 kernels
/*
// naive method
__global__ void detectPeaks1(float *input, char *output, int N)
{
  int elementId = blockIdx.x * blockDim.x + threadIdx.x;
  if (elementId == 0 || elementId == N - 1)
  {
    output[elementId] = 0;
  }
  else if (input[elementId - 1] < input[elementId] && input[elementId + 1] < input[elementId])
  {
    output[elementId] = 1;
  }
  else
  {
    output[elementId] = 0;
  }
}

__global__ void calcLength(float *x, float *y, float *z, float *output)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  output[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
}

__global__ void calcLength2(vector *vec, float *output)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  output[i] = sqrt(vec[i].x * vec[i].x + vec[i].y * vec[i].y + vec[i].z * vec[i].z);
}

// using cuda shuffle
__global__ void detectPeaks2(float *input, char *output)
{
  int elementId = blockIdx.x * blockDim.x + threadIdx.x;
  int lineId = elementId % warpSize;
  int warpId = elementId / warpSize;
  int index = warpId * (warpSize - 2) + lineId;
  float value = input[index];

  float right = __shfl_down(value, 1);
  float left = __shfl_up(value, 1);

  if (lineId != 0 && lineId != warpSize - 1)
  {
    output[index] = right < value && left < value;
  }
}

// reduction sum
__global__ void sum(char* input, int* output, int length){
  extern __shared__ int sdata[];

  int threadId = threadIdx.x;
  int elementId = blockDim.x * blockIdx.x + threadIdx.x;

  if(elementId == 0 || elementId >= length-1){
    sdata[threadId] = 0;
  }else{
    sdata[threadId] = (int)input[elementId];
  }
  __syncthreads();

  for(int s=blockDim.x / 2; s > 0; s /= 2){
    if(threadId < s){
      sdata[threadId] += sdata[threadId+s];

    }
    __syncthreads();
  }

  if(threadId == 0){
    atomicAdd(output,sdata[0]);
  }
}*/

__global__ void merged(vector *vec, int *output, int length)
{
  extern __shared__ int sum[];
  int threadId = threadIdx.x;
  int elementId = blockIdx.x * blockDim.x + threadIdx.x;
  int lineId = elementId % warpSize;
  int warpId = elementId / warpSize;
  int index = warpId * (warpSize - 2) + lineId;
  int warpsPerBlock = blockDim.x / warpSize;
  float value = vec[index].x * vec[index].x + vec[index].y * vec[index].y + vec[index].z * vec[index].z;
  float right = __shfl_down(value, 1);
  float left = __shfl_up(value, 1);

  int result = 0;
  if (lineId != 0 && lineId != warpSize - 1)
  {
    result = right < value && left < value;
  }

  for (unsigned int s = warpSize / 2; s > 0; s >>= 1)
  {
    result += __shfl_down(result, s);
  }

  if (lineId == 0)
  {
    sum[warpId % warpsPerBlock] = result;
  }
  __syncthreads();

  if (threadId < warpsPerBlock)
  {
    result = sum[threadId];
    for (unsigned int s = warpSize / 2; s > 0; s >>= 1)
    {
      result += __shfl_down(result, s);
    }
    if (threadId == 0)
    {
      atomicAdd(output, result);
    }
  }
}

int main(int argc, char *argv[])
{

  int N;
  vector *vec;
  // float *x, *y, *z;
  int corrected, length;
  int warpSize, blockSize;
  cudaCheckErrors(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0));
  cudaCheckErrors(cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0));
  cout << "Warp size: " << warpSize << ", block size: " << blockSize << endl;

  if (argc > 1)
  {
    length = atoi(argv[1]);
    cout << "Original size: " << length << endl;

    corrected = ceil(((float)length) * 1.0 / ((float)warpSize - 2) * (float)warpSize);
    cout << "Corrected size: " << corrected << endl;

    if (corrected % blockSize == 0)
    {
      N = corrected;
    }
    else
    {
      N = (corrected / blockSize + 1) * blockSize;
    }
    cout << "Total size:" << N << endl;

    cudaCheckErrors(cudaMallocHost(&vec, N * sizeof(vector)));
    // cudaCheckErrors(cudaMallocHost(&x,N*sizeof(float)));
    // cudaCheckErrors(cudaMallocHost(&y,N*sizeof(float)));
    // cudaCheckErrors(cudaMallocHost(&z,N*sizeof(float)));

    srand(0);
    for (int i = 0; i < N; i++)
    {
      if (i >= length)
      {
        // x[i]=0;
        // y[i]=0;
        // z[i]=0;
        vec[i].x = 0;
        vec[i].y = 0;
        vec[i].z = 0;
        continue;
      }
      vec[i].x = rand() % 10;
      vec[i].y = rand() % 10;
      vec[i].z = rand() % 10;
      // x[i] = rand()*10;
      // y[i] = rand()*10;
      // z[i] = rand()*10;
      // cout << vec[i].x << " " << vec[i].y << " " << vec[i].z << endl;
    }
  }
  else
  {
    ifstream inputFile("accelerometer.txt");
    if (inputFile.is_open())
    {
      inputFile >> length;
      cout << "Original size: " << length << endl;
      corrected = ceil((float)(length - 2) / (warpSize - 2) * warpSize);
      cout << "Corrected size: " << corrected << endl;
      if (corrected % blockSize == 0)
      {
        N = corrected;
      }
      else
      {
        N = (corrected / blockSize + 1) * blockSize;
      }
      cout << "Total size:" << N << endl;
      cudaCheckErrors(cudaMallocHost(&vec, N * sizeof(vector)));
      // cudaCheckErrors(cudaMallocHost(&x,N*sizeof(float)));
      // cudaCheckErrors(cudaMallocHost(&y,N*sizeof(float)));
      // cudaCheckErrors(cudaMallocHost(&z,N*sizeof(float)));

      for (int i = 0; i < N; i++)
      {
        if (i >= length)
        {
          // x[i]=0;
          // y[i]=0;
          // z[i]=0;
          vec[i].x = 0;
          vec[i].y = 0;
          vec[i].z = 0;
          continue;
        }
        float elapsedTimeSystem, elapsedTimeSensor, xVal, yVal, zVal;
        inputFile >> elapsedTimeSystem;
        inputFile >> elapsedTimeSensor;
        inputFile >> xVal;
        inputFile >> yVal;
        inputFile >> zVal;
        // cout << xVal << " " << yVal << " " << zVal << endl;
        vec[i].x = xVal;
        vec[i].y = yVal;
        vec[i].z = zVal;
      }
    }
    else
    {
      cerr << "File not open\n";
      return -1;
    }
  }

  auto timeStart = high_resolution_clock::now();

#ifdef MEASURE_TIME
  cudaEvent_t cudaStart, cudaMemcpyStart, cudaMemcpyEnd, cudaKernelEnd, cudaEnd;
  cudaCheckErrors(cudaEventCreate(&cudaStart));
  cudaCheckErrors(cudaEventCreate(&cudaMemcpyStart));
  cudaCheckErrors(cudaEventCreate(&cudaMemcpyEnd));
  cudaCheckErrors(cudaEventCreate(&cudaKernelEnd));
  cudaCheckErrors(cudaEventCreate(&cudaEnd));

  cudaCheckErrors(cudaEventRecord(cudaStart));
#endif

  int blockDim = blockSize, gridDim = N / blockSize;

  // float *d_x, *d_y, *d_z, *d_length;
  float *d_length;
  vector *d_vec;
  char *d_output;
  int *d_sum;
  char *h_output;
  int *h_sum = new int(0);
  cudaCheckErrors(cudaMallocHost(&h_output, N * sizeof(char)));
  /*cudaCheckErrors(cudaMalloc(&d_x, N * sizeof(float)));
  cudaCheckErrors(cudaMalloc(&d_y, N * sizeof(float)));
  cudaCheckErrors(cudaMalloc(&d_z, N * sizeof(float)));*/
  cudaCheckErrors(cudaMalloc(&d_vec, N * sizeof(vector)));
  cudaCheckErrors(cudaMalloc(&d_length, N * sizeof(float)));
  cudaCheckErrors(cudaMemset(d_length, 0, N * sizeof(char)));
  cudaCheckErrors(cudaMalloc(&d_output, N * sizeof(char)));
  cudaCheckErrors(cudaMemset(d_output, 0, N * sizeof(char)));
  cudaCheckErrors(cudaMalloc(&d_sum, sizeof(int)));
  cudaCheckErrors(cudaMemset(d_sum, 0, sizeof(int)));

  cudaStream_t streamX, streamY, streamZ;
  cudaCheckErrors(cudaStreamCreate(&streamX));
  cudaCheckErrors(cudaStreamCreate(&streamY));
  cudaCheckErrors(cudaStreamCreate(&streamZ));

#ifdef MEASURE_TIME
  auto memcpyStart = high_resolution_clock::now();
  cudaCheckErrors(cudaEventRecord(cudaMemcpyStart));
#endif

  /*cudaCheckErrors(cudaMemsetAsync(d_x, 0, N * sizeof(float), streamX));
  cudaCheckErrors(cudaMemcpyAsync(d_x, x, length * sizeof(float), cudaMemcpyHostToDevice,streamX));
  cudaCheckErrors(cudaMemsetAsync(d_y, 0, N * sizeof(float), streamY));
  cudaCheckErrors(cudaMemcpyAsync(d_y, y, length * sizeof(float), cudaMemcpyHostToDevice,streamY));
  cudaCheckErrors(cudaMemsetAsync(d_z, 0, N * sizeof(float), streamZ));
  cudaCheckErrors(cudaMemcpyAsync(d_z, z, length * sizeof(float), cudaMemcpyHostToDevice,streamZ));*/
  cudaCheckErrors(cudaMemsetAsync(d_vec, 0, N * sizeof(vector)));
  cudaCheckErrors(cudaMemcpyAsync(d_vec, vec, length * sizeof(vector), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

#ifdef MEASURE_TIME
  auto memcpyEnd = high_resolution_clock::now();
  cudaCheckErrors(cudaEventRecord(cudaMemcpyEnd));
#endif

  // calcLength<<<gridDim,blockDim>>>(d_x, d_y, d_z, d_length);
  // calcLength2<<<gridDim,blockDim>>>(d_vec, d_length);
  // detectPeaks1<<<gridDim,blockDim>>>(d_length,d_output,N);
  // detectPeaks2<<<gridDim,blockDim>>>(d_length,d_output);
  // sum<<<gridDim,blockDim,blockDim*sizeof(int)>>>(d_output,d_sum,length);
  // printf("0 0\n");
  merged<<<gridDim, blockDim, (blockSize / warpSize) * sizeof(int)>>>(d_vec, d_sum, corrected);
  cudaDeviceSynchronize();

#ifdef MEASURE_TIME
  auto kernelEnd = high_resolution_clock::now();
  cudaCheckErrors(cudaEventRecord(cudaKernelEnd));
#endif

  cudaCheckErrors(cudaMemcpy(h_output, d_output, N * sizeof(char), cudaMemcpyDeviceToHost));
  cudaCheckErrors(cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));

#ifdef MEASURE_TIME
  cudaCheckErrors(cudaEventRecord(cudaEnd));
#endif

  auto timeEnd = high_resolution_clock::now();

  cout << "Result: " << *h_sum << endl;
  cout << "Total time: " << duration_cast<milliseconds>(timeEnd - timeStart).count() << " ms" << endl;
#ifdef MEASURE_TIME
  cout << "STD::CHRONO TIMES:" << endl;
  cout << "Memory & stream allocation time: " << duration_cast<milliseconds>(memcpyStart - timeStart).count() << " ms" << endl;
  cout << "Parallel memcpy time: " << duration_cast<microseconds>(memcpyEnd - memcpyStart).count() << " us" << endl;
  cout << "Kernel execution time: " << duration_cast<microseconds>(kernelEnd - memcpyEnd).count() << " us" << endl;
  cout << "Results copy-back time: " << duration_cast<microseconds>(timeEnd - kernelEnd).count() << " us" << endl;

  cout << "CUDA EVENT TIMES:" << endl;
  cudaEventSynchronize(cudaEnd);
  float initTime, memcpyTime, kernelTime, resultTime;
  cudaEventElapsedTime(&initTime, cudaStart, cudaMemcpyStart);
  cudaEventElapsedTime(&memcpyTime, cudaMemcpyStart, cudaMemcpyEnd);
  cudaEventElapsedTime(&kernelTime, cudaMemcpyEnd, cudaKernelEnd);
  cudaEventElapsedTime(&resultTime, cudaKernelEnd, cudaEnd);  
  cout << "Memory & stream allocation time: " << initTime << " ms" << endl;
  cout << "Parallel memcpy time: " << memcpyTime * 1000 << " us" << endl;
  cout << "Kernel execution time: " << kernelTime * 1000 << " us" << endl;
  cout << "Results copy-back time: " << resultTime * 1000 << " us" << endl;
#endif

  /*cudaFree(x);
  cudaFree(y);
  cudaFree(z);*/
  cudaFree(h_output);
  delete h_sum;
  /*cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);*/
  cudaFree(d_length);
  cudaFree(d_output);
  cudaFree(d_sum);

  return 0;
}
