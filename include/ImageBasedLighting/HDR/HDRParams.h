#ifndef HDRPARAMS_H
#define HDRPARAMS_H

typedef struct HDRParams
{
	float SHCoeffs[27];
	float dominantLightDirection[3];
	float dominantLightColor[3];
	float specularScaleFactor;
	float diffuseScaleFactor;
	float shininess;
	int cameraID;
};

#endif