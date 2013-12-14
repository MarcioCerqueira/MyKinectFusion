#ifndef MEMCPY_CUDA_H
#define MEMCPY_CUDA_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;

cudaPitchedPtr CastVolumeHostToDevice(unsigned char* host, unsigned int width, unsigned int height, unsigned int depth);
void CastVolumeDeviceToHost(unsigned char* host, const cudaPitchedPtr device, uint width, uint height, uint depth);

#endif