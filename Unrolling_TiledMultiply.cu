#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

# define TILE_WIDTH 32

// Input matrix unrolling & tiled matrix multiplication using shared memory

/*
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;

    // From textbook
    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K-1;
    extern __shared__ float shmem[];

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base out data index for the block
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0.0f;
    for (int c = 0; c < Channel; c++) { // sum over all input channels
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
            for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
                shmem[((i-h_base) * X_tile_width) + (j-w_base)] = in_4d(n, c, i, j);
            }
        }
        __syncthreads();
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += shmem[w0 + q + (h0 + p) * X_tile_width] * mask_4d(m, c, p, q);
            }
        }
        __syncthreads();
    }

    if (h < Height_out && w < Width_out) {
        out_4d(n, m, h, w) = acc;
    }


    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
*/

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    // Loop over the M and N tiles required to compute the P element
    // The code assumes that the Width is a multiple of TILE_WIDTH!
    for (int q = 0; q < (numAColumns - 1)/TILE_WIDTH + 1; ++q)  {
        // Collaborative loading of M and N tiles into shared memory
        if (Row < numCRows && q*TILE_WIDTH+tx < numAColumns) {
            // as before
            subTileM[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
        } else {
            subTileM[ty][tx] = 0;
        }
        if (q*TILE_WIDTH+ty < numBRows && Col < numCColumns ) {
            // as before
            subTileN[ty][tx] = B[(q*TILE_WIDTH+ty)*numCColumns+Col];
        } else {
            subTileN[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns) {
        // as before
        C[Row*numCColumns+Col] = Pvalue;
    }



}

__host__ void unroll_input_fx(const int Channel, const int Height, const int Width, const int K, const float* X, float* X_unroll, const int Batch) {    
    #define in_4d(i3, i2, i1, i0) X[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    int c, h, w, p, q, w_base, w_unroll, h_unroll;
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    for(c = 0; c < Channel; c++) {
        // beginning of unrolled
        w_base = c * (K*K);
        for(p = 0; p < K; p++) {
            for(q = 0; q < K; q++) {
                for(h = 0; h < H_out; h++) {
                    for(w = 0; w < W_out; w ++) {
                        // horizontal matrix index
                        w_unroll = w_base + (p * K) + q;
                        // vertical matrix index
                        h_unroll = (h * W_out) + w;

                        X_unroll[(w_unroll * W_out * H_out) +  h_unroll] = in_4d(Batch, c, h + p, w + q);
                    }
                }
            }
        }
    }
    #undef in_4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    //cudaMemcpy(*device_output_ptr, host_output, Batch * Map_out * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    



}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    int sizeOfMask = Map_out * K * K * Channel;
    int sizeOfInput = Channel * K * K * H_out * W_out;
    int sizeOfOutput = H_out * W_out * Map_out;

    float* host_input = (float*)malloc(Batch * Channel * Height * Width * sizeof(float));
    float* host_mask = (float*)malloc(Map_out * K * K * Channel * sizeof(float));
    float* host_output = (float*)malloc(Batch * Map_out * H_out * W_out * sizeof(float));

    cudaMemcpy(host_input, device_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mask, device_mask, Map_out * K * K * Channel * sizeof(float), cudaMemcpyDeviceToHost);

    float *device_output_ptr; 
    float *device_input_ptr;
    float *device_mask_ptr;

    cudaMalloc((void**)&device_output_ptr, Map_out * H_out * W_out * sizeof(float));
    cudaMalloc((void**)&device_input_ptr, Channel * K * K * H_out * W_out * sizeof(float));
    cudaMalloc((void**)&device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(device_mask_ptr, host_mask, sizeOfMask*sizeof(float), cudaMemcpyHostToDevice);

    float* unroll_input = (float*)malloc(sizeOfInput * sizeof(float));

    int numARows = Map_out;
    int numAColumns = Channel*K*K;
    int numBRows = Channel*K*K;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(1.0*numCColumns/TILE_WIDTH), ceil(1.0*numCRows/TILE_WIDTH), 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    int offset = 0;

    for (int b = 0; b < Batch; b++) {
        offset = b*H_out*W_out*Map_out;
        // Print the value of 'b'
        unroll_input_fx(Channel, Height, Width, K, host_input, unroll_input, b);
        cudaMemcpy(device_input_ptr, unroll_input, sizeOfInput*sizeof(float), cudaMemcpyHostToDevice);
        matrixMultiplyShared<<<DimGrid, DimBlock>>>(device_mask_ptr, device_input_ptr, device_output_ptr, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
        cudaMemcpy((void*)(host_output + offset), device_output_ptr, sizeOfOutput * sizeof(float), cudaMemcpyDeviceToHost);

    }
    cudaDeviceSynchronize();

    cudaMemcpy(device_output, host_output, sizeOfOutput * Batch* sizeof(float), cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // in prolog

    cudaMemcpy(host_output, device_output, Batch*Map_out*(Width-K+1)*(Height-K+1)*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
    return;

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}