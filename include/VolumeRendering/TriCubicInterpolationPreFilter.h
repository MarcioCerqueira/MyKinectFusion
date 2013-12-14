#ifndef TRICUBICINTERPOLATIONPREFILTER_H
#define TRICUBICINTERPOLATIONPREFILTER_H

#include "memcpy.h"
#include <stdio.h>
#include <cuda.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;

void CubicBSplinePrefilter3D(float* volume, uint pitch, uint width, uint height, uint depth);

class TriCubicInterpolationPreFilter
{
public:
	void applyPreFilterForAccurateCubicBSplineInterpolation(unsigned char *volume, int width, int height, int depth);
};

#endif