#include "VolumeRendering/TriCubicInterpolationPreFilter.h"

inline __device__ __host__ uint UMIN(uint a, uint b)
{
	return a < b ? a : b;
}

inline __device__ __host__ uint PowTwoDivider(uint n)
{
	if (n == 0) return 0;
	uint divider = 1;
	while ((n & divider) == 0) divider <<= 1; 
	return divider;
}

#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline

template<class floatN>
__host__ __device__ floatN InitialCausalCoefficient(
	floatN* c,			// coefficients
	uint DataLength,	// number of coefficients
	int step)			// element interleave in bytes
{
	const uint Horizon = UMIN(12, DataLength);

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	float zn = Pole;
	floatN Sum = *c;
	for (uint n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= Pole;
		c = (floatN*)((uchar*)c + step);
	}
	return(Sum);
}

template<class floatN>
__host__ __device__ floatN InitialAntiCausalCoefficient(
	floatN* c,			// last coefficient
	uint DataLength,	// number of samples or coefficients
	int step)			// element interleave in bytes
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - 1.0f)) * *c);
}

template<class floatN>
__host__ __device__ void ConvertToInterpolationCoefficients(
	floatN* coeffs,		// input samples --> output coefficients
	uint DataLength,	// number of samples or coefficients
	int step)			// element interleave in bytes
{
	// compute the overall gain
	const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

	// causal initialization
	floatN* c = coeffs;
	floatN previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (uint n = 1; n < DataLength; n++) {
		c = (floatN*)((uchar*)c + step);
		*c = previous_c = Lambda * *c + Pole * previous_c;
	}
	// anticausal initialization
	*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
	// anticausal recursion
	for (int n = DataLength - 2; 0 <= n; n--) {
		c = (floatN*)((uchar*)c - step);
		*c = previous_c = Pole * (previous_c - *c);
	}
}

template<class floatN>
__global__ void SamplesToCoefficients3DX(
	floatN* volume,		// in-place processing
	uint pitch,			// width in bytes
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in x-direction
	const uint y = blockIdx.x * blockDim.x + threadIdx.x;
	const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = (z * height + y) * pitch;

	floatN* ptr = (floatN*)((uchar*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr, width, sizeof(floatN));
}

template<class floatN>
__global__ void SamplesToCoefficients3DY(
	floatN* volume,		// in-place processing
	uint pitch,			// width in bytes
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in y-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = z * height * pitch;

	floatN* ptr = (floatN*)((uchar*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr + x, height, pitch);
}

template<class floatN>
__global__ void SamplesToCoefficients3DZ(
	floatN* volume,		// in-place processing
	uint pitch,			// width in bytes
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in z-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = y * pitch;
	const uint slice = height * pitch;

	floatN* ptr = (floatN*)((uchar*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr + x, depth, slice);
}


//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class floatN>
extern void CubicBSplinePrefilter3D2(floatN* volume, uint pitch, uint width, uint height, uint depth)
{
	// Try to determine the optimal block dimensions
	uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
	uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(height / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DX<floatN><<<dimGridX, dimBlock>>>(volume, pitch, width, height, depth);
	//CUT_CHECK_ERROR("SamplesToCoefficients3DX kernel failed");

	dim3 dimGridY(width / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DY<floatN><<<dimGridY, dimBlock>>>(volume, pitch, width, height, depth);
	//CUT_CHECK_ERROR("SamplesToCoefficients3DY kernel failed");

	dim3 dimGridZ(width / dimBlock.x, height / dimBlock.y);
	SamplesToCoefficients3DZ<floatN><<<dimGridZ, dimBlock>>>(volume, pitch, width, height, depth);
	//CUT_CHECK_ERROR("SamplesToCoefficients3DZ kernel failed");
}

void CubicBSplinePrefilter3D(float* volume, uint pitch, uint width, uint height, uint depth) {
	CubicBSplinePrefilter3D2(volume, pitch, width, height, depth);
}