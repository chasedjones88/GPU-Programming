#include <wb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);

  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaError_t err  = cudaMalloc((void **) &deviceInput1, inputLength * sizeof(float));
  if (err == cudaSuccess)
	err = cudaMalloc((void **) &deviceInput2, inputLength * sizeof(float));
  if (err == cudaSuccess)
	err = cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));
  if (err != cudaSuccess) {
	  printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
  }

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  err = cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
	  printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
	  printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
  }

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);
  printf("input length is: %d", inputLength);
  dim3 DimBlock(256, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd<<<DimGrid,DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  err = cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
	  printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
  }

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
