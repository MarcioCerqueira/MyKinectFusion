#ifndef MYGLIMAGEVIEWER_H
#define MYGLIMAGEVIEWER_H

#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>

class MyGLImageViewer
{
public:
	MyGLImageViewer();
	~MyGLImageViewer();
	void loadDepthTexture(unsigned short *data, GLuint *texVBO, int index, int threshold, int windowWidth, int windowHeight);
	void loadDepthBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height);
	void loadDepthComponentTexture(unsigned short *data, GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void loadRGBTexture(const unsigned char *data, GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void loadFrameBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height);
	void loadARTexture(const unsigned char *rgbMap, unsigned char *data, GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void load3DTexture(const unsigned char *data, GLuint *texVBO, int index, int volumeWidth, int volumeHeight, int volumeDepth);
	void drawDepthTexture(GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void drawRGBTexture(GLuint *texVBO, int index, int windowWidth, int windowHeight);
	void drawARTextureWithOcclusion(GLuint *texVBO, int realRGBIndex, int realDepthIndex, int virtualRGBIndex, int virtualDepthIndex, 
		int windowWidth, int windowHeight, bool ARPolygonal = false);
	void draw3DTexture(GLuint *texVBO, int index);
	void setProgram(GLuint shaderProg);
	void setScale(float scale);
private:
	GLuint shaderProg;
	unsigned char *depthData;
	unsigned char *frameBuffer;
};
#endif