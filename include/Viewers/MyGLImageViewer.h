#ifndef MYGLIMAGEVIEWER_H
#define MYGLIMAGEVIEWER_H

#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <time.h>
#include "VolumeRendering/VRParams.h"

class MyGLImageViewer
{
public:
	MyGLImageViewer();
	~MyGLImageViewer();
	void loadDepthTexture(unsigned short *data, GLuint *texVBO, int index, int threshold, int imageWidth, int imageHeight);
	void loadDepthBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height);
	void loadDepthComponentTexture(unsigned short *data, GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void loadRGBTexture(const unsigned char *data, GLuint *texVBO, int index, int imageWidth, int imageHeight);
	void loadRGBATexture(unsigned char *data, GLuint *texVBO, int index, int width, int height);
	void loadFrameBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height);
	void loadARTexture(const unsigned char *rgbMap, unsigned char *data, GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void load2DNoiseTexture(GLuint *texVBO, int index, int width, int height);
	void load3DTexture(unsigned char *data, GLuint *texVBO, int index, int volumeWidth, int volumeHeight, int volumeDepth);
	void drawDepthTexture(GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void drawRGBTexture(GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void drawARTextureWithOcclusion(GLuint *texVBO, int realRGBIndex, int realDepthIndex, int virtualRGBIndex, int virtualDepthIndex, 
		int windowWidth, int windowHeight, bool ARPolygonal = false);
	void draw3DTexture(GLuint *texVBO, int index, int octreeIndex, VRParams params, int frontFBOIndex, int backFBOIndex, float* cameraPos, 
		int windowWidth, int windowHeight, int transferFunctionIndex = 0, int noiseIndex = 0);
	void drawQuads(float x, float y, float z, GLenum target = GL_TEXTURE0);
	void setProgram(GLuint shaderProg);
	void setScale(float scale);
private:
	GLuint shaderProg;
	unsigned char *depthData;
	unsigned char *frameBuffer;
};
#endif