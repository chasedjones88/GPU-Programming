#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH+MASK_WIDTH-1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values
__global__ void tiled2DConvolution(float *d_in, float *d_out, const float* __restrict__ mask,int height, int width, int channels) {
	__shared__ float ds_in[BLOCK_WIDTH][BLOCK_WIDTH * 3];

	int tx = threadIdx.x;	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = (blockIdx.x*O_TILE_WIDTH + tx)*channels;

	int row_i = row_o - 2;
	int col_i = col_o - 2*channels;

	
	if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width*channels))
		for (int k = 0; k < channels; k++)
			ds_in[ty][tx*channels+k] = d_in[row_i*width*channels + col_i + k];
	else 
		for (int k = 0; k < channels; k++)
			ds_in[ty][tx*channels+k] = 0.0f;
	__syncthreads();
	
	float output[3] = { 0.0f , 0.0f, 0.0f};
	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
		for (int i = 0; i < MASK_WIDTH; i++) // per row
			for (int j = 0; j < MASK_WIDTH; j++) // per column
				for (int k = 0; k < channels; k++) // per channel index
					output[k] += mask[i*MASK_WIDTH+j] * ds_in[i + ty][(j + tx)*channels + k];
		__syncthreads();

		if (row_o < height && col_o < width*channels)
			for (int k = 0; k < channels; k++)
				d_out[row_o*width*channels + col_o + k] = clamp(output[k]);
		__syncthreads();
	}

	
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  int size = imageWidth*imageHeight*imageChannels * sizeof(float);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //allocate device memory
  cudaMalloc((void**)&deviceInputImageData, size);
  cudaMalloc((void**)&deviceOutputImageData, size);
  cudaMalloc((void**)&deviceMaskData, MASK_WIDTH*MASK_WIDTH *sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //copy host memory to device
  cudaMemcpy(deviceInputImageData, hostInputImageData, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, MASK_WIDTH*MASK_WIDTH*sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  //initialize thread block and kernel grid dimensions
  //invoke CUDA kernel
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
  tiled2DConvolution <<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageHeight, imageWidth, imageChannels);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  //copy results from device to host
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, size, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ INSERT CODE HERE
  //deallocate device memory	
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
