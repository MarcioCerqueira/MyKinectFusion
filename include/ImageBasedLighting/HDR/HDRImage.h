#ifndef HDRIMAGE_H
#define HDRIMAGE_H

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda_runtime.h>
#include "ImageBasedLighting/HDR/SH.h"
#include "ImageBasedLighting/HDR/HDRParams.h"

#define PI 3.1415

class HDRImage
{
public:
	HDRImage(int width, int height);
	~HDRImage();
	void computeCoordinates();
	void computeDomegaProduct();
	void computeSHCoeffs();
	void computeSphericalMap();
	void computeDominantLightDirection();
	void computeDominantLightColor();
	void load(float *image);
	void load(unsigned char *image);
	void load(HDRParams *params);
	float* getImage() { return image; }
	float* getSHCoeffs() { return SHCoeffs; }
	float* getDominantLightDirection() { return dominantLightDirection; }
	float* getDominantLightColor() { return dominantLightColor; }
	void setScale(float scale) { this->scale = scale; }

private:
	float *image;
	float *cartesianCoord;
	float *sphericalCoord;
	float *domegaProduct;
	float SHCoeffs[27];
	float dominantLightDirection[3];
	float dominantLightColor[3];
	int width;
	int height;
	float scale;

};

#endif