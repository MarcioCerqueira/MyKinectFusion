#ifndef AR_OCCLUSION_PARAMS_H
#define AR_OCCLUSION_PARAMS_H

typedef struct AROcclusionParams
{
	GLuint *texVBO;
	int realRGBIndex;
	int realDepthIndex;
	int virtualRGBIndex;
	int virtualDepthIndex;
	int curvatureMapIndex;
	int contoursMapIndex; //ghostViewBasedOnClipping
	int windowWidth;
	int windowHeight;
	bool ARPolygonal;
	bool ARFromKinectFusionVolume;
	bool ARFromVolumeRendering;
	bool alphaBlending;
	bool ghostViewBasedOnCurvatureMap;
	bool ghostViewBasedOnDistanceFalloff;
	bool ghostViewBasedOnClipping;
	float curvatureWeight;
	float distanceFalloffWeight;
	float clippingWeight;
	float focusPoint[2];
	float focusRadius;
} AROcclusionParams;

#endif