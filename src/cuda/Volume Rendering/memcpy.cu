#include "VolumeRendering/memcpy.h"

template<class T> __device__ float Multiplier()	{ return 1.0f; }
template<> __device__ float Multiplier<unsigned char>()	{ return 255.0f; }
template<> __device__ float Multiplier<signed char>()	{ return 127.0f; }
template<> __device__ float Multiplier<unsigned short>(){ return 65535.0f; }
template<> __device__ float Multiplier<short>()	{ return 32767.0f; }

inline __device__ __host__ uint PowTwoDivider2(uint n)
{
	if (n == 0) return 0;
	uint divider = 1;
	while ((n & divider) == 0) divider <<= 1; 
	return divider;
}

template<class T> __global__ void CopyCast(unsigned char* destination, T* source, unsigned int pitch, unsigned int width)
{
	uint2 index = make_uint2(
		__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
		__umul24(blockIdx.y, blockDim.y) + threadIdx.y);

	float* dest = (float*)(destination + index.y * pitch) + index.x;
	*dest = (1.0f/Multiplier<T>()) * (float)(source[index.y * width + index.x]);
}

template<class T> __global__ void CopyCastBack(T* destination, uchar* source, uint pitch, uint width)
{
	uint2 index = make_uint2(
		__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
		__umul24(blockIdx.y, blockDim.y) + threadIdx.y);

	float* src = (float*)(source + index.y * pitch) + index.x;
	destination[index.y * width + index.x] = (T)(Multiplier<T>() * *src);
}

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! and cast it to the normalized floating point format
//! @return the pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class T> extern cudaPitchedPtr CastVolumeHostToDevice2(T* host, unsigned int width, unsigned int height, unsigned int depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMalloc3D(&device, extent);
	const size_t pitchedBytesPerSlice = device.pitch * device.ysize;
	
	T* temp = 0;
	const unsigned int voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * sizeof(T);
	cudaMalloc((void**)&temp, nrOfBytesTemp);

	unsigned int dimX = min(PowTwoDivider2(width), 64);
	dim3 dimBlock(dimX, min(PowTwoDivider2(height), 512 / dimX));
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
	size_t offsetHost = 0;
	size_t offsetDevice = 0;
	
	for (unsigned int slice = 0; slice < depth; slice++)
	{
		cudaMemcpy(temp, host + offsetHost, nrOfBytesTemp, cudaMemcpyHostToDevice);
		CopyCast<T><<<dimGrid, dimBlock>>>((unsigned char*)device.ptr + offsetDevice, temp, (unsigned int)device.pitch, width);
		//CUT_CHECK_ERROR("Cast kernel failed");
		offsetHost += voxelsPerSlice;
		offsetDevice += pitchedBytesPerSlice;
	}

	cudaFree(temp);  //free the temp GPU volume
	return device;
}

//! Copy a voxel volume from GPU to CPU memory
//! while casting it to the desired format
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param device  pitched pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note The \host CPU memory should be pre-allocated
template<class T> extern void CastVolumeDeviceToHost2(T* host, const cudaPitchedPtr device, uint width, uint height, uint depth)
{
	T* temp = 0;
	const uint voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * sizeof(T);
	cudaMalloc((void**)&temp, nrOfBytesTemp);

	uint dimX = min(PowTwoDivider2(width), 64);
	dim3 dimBlock(dimX, min(PowTwoDivider2(height), 512 / dimX));
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
	const size_t pitchedBytesPerSlice = device.pitch * device.ysize;
	size_t offsetHost = 0;
	size_t offsetDevice = 0;
	
	for (uint slice = 0; slice < depth; slice++)
	{
		CopyCastBack<T><<<dimGrid, dimBlock>>>(temp, (uchar*)device.ptr + offsetDevice, (uint)device.pitch, width);
		//CUT_CHECK_ERROR("Cast kernel failed");
		cudaMemcpy(host + offsetHost, temp, nrOfBytesTemp, cudaMemcpyDeviceToHost);
		offsetHost += voxelsPerSlice;
		offsetDevice += pitchedBytesPerSlice;
	}

	cudaFree(temp);  //free the temp GPU volume
}

cudaPitchedPtr CastVolumeHostToDevice(unsigned char* host, unsigned int width, unsigned int height, unsigned int depth) {
	return CastVolumeHostToDevice2(host, width, height, depth);
}

void CastVolumeDeviceToHost(unsigned char* host, const cudaPitchedPtr device, uint width, uint height, uint depth) {
	CastVolumeDeviceToHost2(host, device, width, height, depth);
}