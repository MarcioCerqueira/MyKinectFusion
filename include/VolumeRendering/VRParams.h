#ifndef VRPARAMS_H
#define VRPARAMS_H

typedef struct VRParams
{
	//transfer function
	char transferFunctionPath[200];
	//step size for raycasting
	float stepSize;
	//early ray termination threshold
	float earlyRayTerminationThreshold;
	//parameter for illustrative context-preserving
	float kt;
	//parameter for illustrative context-preserving
	float ks;
	//scale factors
	float scaleWidth;
	float scaleHeight;
	float scaleDepth;
	//rotation axis
	int rotationX;
	int rotationY;
	int rotationZ;
	bool stochasticJithering;
	bool triCubicInterpolation;
	bool MIP;
	bool NonPolygonalIsoSurface;
	float isoSurfaceThreshold;
	bool gradientByForwardDifferences;
	//Clipping Planes
	bool clippingPlane;
	bool inverseClipping;
	bool clippingOcclusion;
	float clippingPlaneLeftX;
	float clippingPlaneRightX;
	float clippingPlaneUpY;
	float clippingPlaneDownY;
	float clippingPlaneFrontZ;
	float clippingPlaneBackZ;
} VRParams;

#endif