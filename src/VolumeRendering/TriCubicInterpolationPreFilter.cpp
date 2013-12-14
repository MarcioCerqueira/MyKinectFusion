#include "VolumeRendering/TriCubicInterpolationPreFilter.h"

void TriCubicInterpolationPreFilter::applyPreFilterForAccurateCubicBSplineInterpolation(unsigned char *volume, int width, int height, int depth) {
	
	//We will use only the alpha-channel (or the original intensity)
	unsigned char *intensity = (unsigned char*)malloc(width * height * depth);
	for(int voxel = 0; voxel < width * height * depth; voxel++)
		intensity[voxel] = volume[voxel * 4 + 3];

	cudaPitchedPtr bsplineCoeffs = CastVolumeHostToDevice(intensity, width, height, depth);
	CubicBSplinePrefilter3D((float*)bsplineCoeffs.ptr, (uint)bsplineCoeffs.pitch, width, height, depth);
	CastVolumeDeviceToHost(intensity, bsplineCoeffs, width, height, depth);

	float opacity;
	for(int voxel = 0; voxel < width * height * depth; voxel++) {
		opacity = (float)(intensity[voxel]/255.f);
		volume[voxel * 4 + 0] = intensity[voxel] * opacity;
		volume[voxel * 4 + 1] = intensity[voxel] * opacity;
		volume[voxel * 4 + 2] = intensity[voxel] * opacity;
		volume[voxel * 4 + 3] = intensity[voxel];
	}

	delete [] intensity;

}