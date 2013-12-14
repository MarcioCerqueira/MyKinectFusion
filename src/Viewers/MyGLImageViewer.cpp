#include "Viewers/MyGLImageViewer.h"

MyGLImageViewer::MyGLImageViewer()
{
	depthData = (unsigned char*)malloc(640 * 480 * 3 * sizeof(unsigned char));
	frameBuffer = (unsigned char*)malloc(640 * 480 * 3 * sizeof(unsigned char));
}

MyGLImageViewer::~MyGLImageViewer()
{
	delete [] depthData;
	delete [] frameBuffer;
}

void MyGLImageViewer::loadDepthTexture(unsigned short *data, GLuint *texVBO, int index, int threshold, int imageWidth, int imageHeight)
{
	
	//Normalize to (0..255)
	for(int p = 0; p < (640 * 480); p++) {
		depthData[p * 3 + 0] = data[p] / 20;
		depthData[p * 3 + 1] = depthData[p * 3 + 0];
		depthData[p * 3 + 2] = depthData[p * 3 + 0];
	}

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, depthData);
}
	
void MyGLImageViewer::loadDepthComponentTexture(unsigned short *data, GLuint *texVBO, int index, int windowWidth, int windowHeight)
{

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, windowWidth/2, windowHeight/2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, data);

}

void MyGLImageViewer::loadRGBTexture(const unsigned char *data, GLuint *texVBO, int index, int imageWidth, int imageHeight)
{

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

}

void MyGLImageViewer::loadRGBATexture(unsigned char *data, GLuint *texVBO, int index, int width, int height) {
	
	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

}

void MyGLImageViewer::loadFrameBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height)
{

	glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
	
	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);

}

void MyGLImageViewer::loadARTexture(const unsigned char *rgbMap, unsigned char *data, GLuint *texVBO, int index, int windowWidth, 
	int windowHeight) {
	
	for(int pixel = 0; pixel < (640 * 480); pixel++) {
		if(data[pixel * 3 + 0] == 0)
			for(int color = 0; color < 3; color++)
				data[pixel * 3 + color] = rgbMap[pixel * 3 + color];
		else 
			for(int color = 0; color < 3; color++)
				data[pixel * 3 + color] = data[pixel * 3 + color] * 0.5 + rgbMap[pixel * 3 + color] * 0.5;
	}

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth/2, windowHeight/2, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

}

void MyGLImageViewer::load2DNoiseTexture(GLuint *texVBO, int index, int width, int height) {
	
	unsigned char *noise = (unsigned char*)malloc(width * height * sizeof(unsigned char));

	srand((unsigned)time(NULL));
	for(int pixel = 0; pixel < (width * height); pixel++)
		noise[pixel] = 255.f * rand()/(float)RAND_MAX;

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, noise);

	delete [] noise;

}

void MyGLImageViewer::load3DTexture(unsigned char *data, GLuint *texVBO, int index, int volumeWidth, int volumeHeight, int volumeDepth)
{
	
	glBindTexture(GL_TEXTURE_3D, texVBO[index]);
	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, volumeWidth, volumeHeight, volumeDepth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_3D, 0);
}

void MyGLImageViewer::drawDepthTexture(GLuint *texVBO, int index, int windowWidth, int windowHeight)
{
	gluOrtho2D( 0, windowWidth, windowHeight/2, 0 ); 
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texVBO[index]);

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); 
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); 
		glVertex2f(windowWidth, 0.0f);
		glTexCoord2f(1.0f, 1.0f); 
		glVertex2f(windowWidth, windowHeight);
		glTexCoord2f(0.0f, 1.0f); 
		glVertex2f(0.0f, windowHeight);
	glEnd();

	glDisable(GL_TEXTURE_2D);
}

void MyGLImageViewer::drawRGBTexture(GLuint *texVBO, int index, int windowWidth, int windowHeight)
{

	//if(index == 2)
	//	glUseProgram(shaderProg);
	
	gluOrtho2D( 0, windowWidth/2, windowHeight/2, 0 ); 
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); 
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); 
		glVertex2f(windowWidth/2, 0.0f);
		glTexCoord2f(1.0f, 1.0f); 
		glVertex2f(windowWidth/2, windowHeight/2);
		glTexCoord2f(0.0f, 1.0f); 
		glVertex2f(0.0f, windowHeight/2);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	
	//if(index == 2)
	//	glUseProgram(0);
}


void MyGLImageViewer::drawARTextureWithOcclusion(GLuint *texVBO, int realRGBIndex, int realDepthIndex, int virtualRGBIndex, int virtualDepthIndex, 
		int windowWidth, int windowHeight, bool ARPolygonal)
{
	
	glUseProgram(shaderProg);
	
	gluOrtho2D( 0, windowWidth/2, windowHeight/2, 0 ); 
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	
	GLuint texLoc = glGetUniformLocation(shaderProg, "realRGB");
	glUniform1i(texLoc, 0);
	texLoc = glGetUniformLocation(shaderProg, "realDepth");
	glUniform1i(texLoc, 1);
	texLoc = glGetUniformLocation(shaderProg, "virtualRGB");
	glUniform1i(texLoc, 2);
	texLoc = glGetUniformLocation(shaderProg, "virtualDepth");
	glUniform1i(texLoc, 3);
	texLoc = glGetUniformLocation(shaderProg, "ARPolygonal");
	glUniform1i(texLoc, (int)ARPolygonal);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texVBO[realRGBIndex]);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texVBO[realDepthIndex]);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, texVBO[virtualRGBIndex]);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, texVBO[virtualDepthIndex]);

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); 
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); 
		glVertex2f(windowWidth/2, 0.0f);
		glTexCoord2f(1.0f, 1.0f); 
		glVertex2f(windowWidth/2, windowHeight/2);
		glTexCoord2f(0.0f, 1.0f); 
		glVertex2f(0.0f, windowHeight/2);
	glEnd();

	glUseProgram(0);

	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE2);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3);
	glDisable(GL_TEXTURE_2D);

}

void MyGLImageViewer::draw3DTexture(GLuint *texVBO, int index, int octreeIndex, VRParams params, int frontFBOIndex, int backFBOIndex, 
	float* cameraPos, int windowWidth, int windowHeight, int transferFunctionIndex, int noiseIndex)
{	
	
	glUseProgram(shaderProg);
	
	GLuint texLoc = glGetUniformLocation(shaderProg, "volume");
	glUniform1i(texLoc, 7);
	
	glActiveTexture(GL_TEXTURE7);
	glEnable(GL_TEXTURE_3D);
	glBindTexture(GL_TEXTURE_3D, texVBO[index]);
	
	texLoc = glGetUniformLocation(shaderProg, "minMaxOctree");
	glUniform1i(texLoc, 2);

	glActiveTexture(GL_TEXTURE2);
	glEnable(GL_TEXTURE_3D);
	glBindTexture(GL_TEXTURE_3D, texVBO[octreeIndex]);
	
	if(params.stepSize >= 0) {

		GLuint texLoc = glGetUniformLocation(shaderProg, "stepSize");
		glUniform1f(texLoc, params.stepSize);

		texLoc = glGetUniformLocation(shaderProg, "earlyRayTerminationThreshold");
		glUniform1f(texLoc, params.earlyRayTerminationThreshold);

		texLoc = glGetUniformLocation(shaderProg, "camera");
		glUniform3f(texLoc, cameraPos[0], cameraPos[1], cameraPos[2]); 

		texLoc = glGetUniformLocation(shaderProg, "kt");
		glUniform1f(texLoc, params.kt);

		texLoc = glGetUniformLocation(shaderProg, "ks");
		glUniform1f(texLoc, params.ks);

		if(params.stochasticJithering) {
			texLoc = glGetUniformLocation(shaderProg, "stochasticJithering");
			glUniform1i(texLoc, 1);
		} else {
			texLoc = glGetUniformLocation(shaderProg, "stochasticJithering");
			glUniform1i(texLoc, 0);
		}

		if(params.triCubicInterpolation) {
			texLoc = glGetUniformLocation(shaderProg, "triCubicInterpolation");
			glUniform1i(texLoc, 1);
		} else {
			texLoc = glGetUniformLocation(shaderProg, "triCubicInterpolation");
			glUniform1i(texLoc, 0);
		}

		if(params.MIP) {
			texLoc = glGetUniformLocation(shaderProg, "MIP");
			glUniform1i(texLoc, 1);
		} else {
			texLoc = glGetUniformLocation(shaderProg, "MIP");
			glUniform1i(texLoc, 0);
		}

		if(params.gradientByForwardDifferences) {
			texLoc = glGetUniformLocation(shaderProg, "forwardDifference");
			glUniform1i(texLoc, 1);
		} else { 
			texLoc = glGetUniformLocation(shaderProg, "forwardDifference");
			glUniform1i(texLoc, 0);
		}

		texLoc = glGetUniformLocation(shaderProg, "isosurfaceThreshold");
		glUniform1f(texLoc, params.isoSurfaceThreshold);

		texLoc = glGetUniformLocation(shaderProg, "windowWidth");
		glUniform1i(texLoc, windowWidth);

		texLoc = glGetUniformLocation(shaderProg, "windowHeight");
		glUniform1i(texLoc, windowHeight);

	}
	
	drawQuads(1.0f/params.scaleWidth, 1.0f/params.scaleHeight, 1.0f/params.scaleDepth);	
	
	if(transferFunctionIndex != 0) {
	
		texLoc = glGetUniformLocation(shaderProg, "transferFunction");
		glUniform1i(texLoc, 4);

		glActiveTexture(GL_TEXTURE4);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texVBO[transferFunctionIndex]);

	}

	if(noiseIndex != 0) {
	
		texLoc = glGetUniformLocation(shaderProg, "noise");
		glUniform1i(texLoc, 8);

		glActiveTexture(GL_TEXTURE8);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texVBO[noiseIndex]);

	}
	
	texLoc = glGetUniformLocation(shaderProg, "backFrameBuffer");
	glUniform1i(texLoc, 5);

	glActiveTexture(GL_TEXTURE5);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texVBO[backFBOIndex]);
	
	texLoc = glGetUniformLocation(shaderProg, "frontFrameBuffer");
	glUniform1i(texLoc, 6);

	glActiveTexture(GL_TEXTURE6);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texVBO[frontFBOIndex]);
	
	glUseProgram(0);

	glActiveTexture(GL_TEXTURE2);
	glDisable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE8);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE4);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE5);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE6);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE7);
	glDisable(GL_TEXTURE_3D);

}

void MyGLImageViewer::drawQuads(float x, float y, float z, GLenum target) {


	bool color = true;
	glBegin(GL_QUADS);
	//front
	if(color) glColor3f(0, 1, 1);
	glMultiTexCoord3f(target, 0.0f, 1.0f, 1.0f);
	glVertex3f(-x, y, z);
	if(color) glColor3f(0, 0, 1);
	glMultiTexCoord3f(target, 0.0f, 0.0f, 1.0f);
	glVertex3f(-x, -y, z);
	if(color) glColor3f(1, 0, 1);
	glMultiTexCoord3f(target, 1.0f, 0.0f, 1.0f);
	glVertex3f(x, -y, z);
	if(color) glColor3f(1, 1, 1);
	glMultiTexCoord3f(target, 1.0f, 1.0f, 1.0f);
	glVertex3f(x, y, z);
	
	//left
	if(color) glColor3f(0, 1, 0);
	glMultiTexCoord3f(target, 0.0f, 1.0f, 0.0f);
	glVertex3f(-x, y, -z);
	if(color) glColor3f(0, 0, 0);
	glMultiTexCoord3f(target, 0.0f, 0.0f, 0.0f);
	glVertex3f(-x, -y, -z);
	if(color) glColor3f(0, 0, 1);
	glMultiTexCoord3f(target, 0.0f, 0.0f, 1.0f);
	glVertex3f(-x, -y, z);
	if(color) glColor3f(0, 1, 1);
	glMultiTexCoord3f(target, 0.0f, 1.0f, 1.0f);
	glVertex3f(-x, y, z);

	//back
	if(color) glColor3f(1, 1, 0);
	glMultiTexCoord3f(target, 1.0f, 1.0f, 0.0f);
	glVertex3f(x, y, -z);
	if(color) glColor3f(1, 0, 0);
	glMultiTexCoord3f(target, 1.0f, 0.0f, 0.0f);
	glVertex3f(x, -y, -z);
	if(color) glColor3f(0, 0, 0);
	glMultiTexCoord3f(target, 0.0f, 0.0f, 0.0f);
	glVertex3f(-x, -y, -z);
	if(color) glColor3f(0, 1, 0);
	glMultiTexCoord3f(target, 0.0f, 1.0f, 0.0f);
	glVertex3f(-x, y, -z);

	//right
	if(color) glColor3f(1, 1, 1);
	glMultiTexCoord3f(target, 1.0f, 1.0f, 1.0f);
	glVertex3f(x, y, z);
	if(color) glColor3f(1, 0, 1);
	glMultiTexCoord3f(target, 1.0f, 0.0f, 1.0f);
	glVertex3f(x, -y, z);
	if(color) glColor3f(1, 0, 0);
	glMultiTexCoord3f(target, 1.0f, 0.0f, 0.0f);
	glVertex3f(x, -y, -z);
	if(color) glColor3f(1, 1, 0);
	glMultiTexCoord3f(target, 1.0f, 1.0f, 0.0f);
	glVertex3f(x, y, -z);

	//top
	if(color) glColor3f(0, 1, 0);
	glMultiTexCoord3f(target, 0.0f, 1.0f, 0.0f);
	glVertex3f(-x, y, -z);
	if(color) glColor3f(0, 1, 1);
	glMultiTexCoord3f(target, 0.0f, 1.0f, 1.0f);
	glVertex3f(-x, y, z);
	if(color) glColor3f(1, 1, 1);
	glMultiTexCoord3f(target, 1.0f, 1.0f, 1.0f);
	glVertex3f(x, y, z);
	if(color) glColor3f(1, 1, 0);
	glMultiTexCoord3f(target, 1.0f, 1.0f, 0.0f);
	glVertex3f(x, y, -z);

	//bottom
	if(color) glColor3f(1, 0, 0);
	glMultiTexCoord3f(target, 1.0f, 0.0f, 0.0f);
	glVertex3f(x, -y, -z);
	if(color) glColor3f(1, 0, 1);
	glMultiTexCoord3f(target, 1.0f, 0.0f, 1.0f);
	glVertex3f(x, -y, z);
	if(color) glColor3f(0, 0, 1);
	glMultiTexCoord3f(target, 0.0f, 0.0f, 1.0f);
	glVertex3f(-x, -y, z);
	if(color) glColor3f(0, 0, 0);
	glMultiTexCoord3f(target, 0.0f, 0.0f, 0.0f);
	glVertex3f(-x, -y, -z);
	
	glEnd();


}

void MyGLImageViewer::setProgram(GLuint shaderProg) 
{
	this->shaderProg = shaderProg;
}
