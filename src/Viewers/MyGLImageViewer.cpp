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
	
void MyGLImageViewer::loadDepthComponentTexture(float *data, GLuint *texVBO, int index, int windowWidth, int windowHeight)
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

void MyGLImageViewer::loadRGBTextureWithMipMaps(const unsigned char *data, GLuint *texVBO, int index, int imageWidth, int imageHeight)
{

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, data);

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

void MyGLImageViewer::loadFrameBufferTexture(int x, int y, int width, int height)
{
	glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);	
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

void MyGLImageViewer::drawRGBTextureOnShader(GLuint *texVBO, int index, int windowWidth, int windowHeight)
{

	glUseProgram(shaderProg);
	
	gluOrtho2D( 0, windowWidth/2, windowHeight/2, 0 ); 
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	GLuint texLoc = glGetUniformLocation(shaderProg, "image");
	glUniform1i(texLoc, 0);
	
	glActiveTexture(GL_TEXTURE0);
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

	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D);
	
	glUseProgram(0);

}

void MyGLImageViewer::drawARTextureWithOcclusion(AROcclusionParams occlusionParams)
{
	
	glUseProgram(shaderProg);
	
	gluOrtho2D( 0, occlusionParams.windowWidth/2, occlusionParams.windowHeight/2, 0 ); 
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	
	GLuint texLoc = glGetUniformLocation(shaderProg, "realRGBTexture");
	glUniform1i(texLoc, 0);
	texLoc = glGetUniformLocation(shaderProg, "realDepthTexture");
	glUniform1i(texLoc, 1);
	texLoc = glGetUniformLocation(shaderProg, "virtualRGBTexture");
	glUniform1i(texLoc, 2);
	texLoc = glGetUniformLocation(shaderProg, "virtualDepthTexture");
	glUniform1i(texLoc, 3);
	texLoc = glGetUniformLocation(shaderProg, "ARPolygonal");
	glUniform1i(texLoc, (int)occlusionParams.ARPolygonal);
	texLoc = glGetUniformLocation(shaderProg, "ARFromKinectFusionVolume");
	glUniform1i(texLoc, (int)occlusionParams.ARFromKinectFusionVolume);
	texLoc = glGetUniformLocation(shaderProg, "ARFromVolumeRendering");
	glUniform1i(texLoc, (int)occlusionParams.ARFromVolumeRendering);
	texLoc = glGetUniformLocation(shaderProg, "alphaBlending");
	glUniform1i(texLoc, (int)occlusionParams.alphaBlending);
	texLoc = glGetUniformLocation(shaderProg, "ghostViewBasedOnCurvatureMap");
	glUniform1i(texLoc, (int)occlusionParams.ghostViewBasedOnCurvatureMap);
	texLoc = glGetUniformLocation(shaderProg, "ghostViewBasedOnDistanceFalloff");
	glUniform1i(texLoc, (int)occlusionParams.ghostViewBasedOnDistanceFalloff);
	texLoc = glGetUniformLocation(shaderProg, "ghostViewBasedOnClipping");
	glUniform1i(texLoc, (int)occlusionParams.ghostViewBasedOnClipping);

	if(occlusionParams.ARFromVolumeRendering)
	{
		
		if(occlusionParams.ghostViewBasedOnCurvatureMap) {
			texLoc = glGetUniformLocation(shaderProg, "curvatureMap");
			glUniform1i(texLoc, 9);
			texLoc = glGetUniformLocation(shaderProg, "curvatureWeight");
			glUniform1f(texLoc, occlusionParams.curvatureWeight);
		}

		if(occlusionParams.ghostViewBasedOnDistanceFalloff) {
			texLoc = glGetUniformLocation(shaderProg, "distanceFalloffWeight");
			glUniform1f(texLoc, occlusionParams.distanceFalloffWeight);
		}
	
		if(occlusionParams.ghostViewBasedOnClipping) {
			texLoc = glGetUniformLocation(shaderProg, "contoursMap");
			glUniform1i(texLoc, 10);
			texLoc = glGetUniformLocation(shaderProg, "clippingWeight");
			glUniform1f(texLoc, occlusionParams.clippingWeight);
		
		}

		texLoc = glGetUniformLocation(shaderProg, "focusPoint");
		glUniform2f(texLoc, occlusionParams.focusPoint[0], occlusionParams.focusPoint[1]);
		texLoc = glGetUniformLocation(shaderProg, "focusRadius");
		glUniform1f(texLoc, occlusionParams.focusRadius);
	
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, occlusionParams.texVBO[occlusionParams.realRGBIndex]);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, occlusionParams.texVBO[occlusionParams.realDepthIndex]);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, occlusionParams.texVBO[occlusionParams.virtualRGBIndex]);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, occlusionParams.texVBO[occlusionParams.virtualDepthIndex]);
	if(occlusionParams.ARFromVolumeRendering)
	{
		if(occlusionParams.ghostViewBasedOnCurvatureMap) {
			glActiveTexture(GL_TEXTURE9);
			glBindTexture(GL_TEXTURE_2D, occlusionParams.texVBO[occlusionParams.curvatureMapIndex]);
		}

		if(occlusionParams.ghostViewBasedOnClipping) {
			glActiveTexture(GL_TEXTURE10);
			glBindTexture(GL_TEXTURE_2D, occlusionParams.texVBO[occlusionParams.contoursMapIndex]);
		}
	}

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); 
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); 
		glVertex2f(occlusionParams.windowWidth/2, 0.0f);
		glTexCoord2f(1.0f, 1.0f); 
		glVertex2f(occlusionParams.windowWidth/2, occlusionParams.windowHeight/2);
		glTexCoord2f(0.0f, 1.0f); 
		glVertex2f(0.0f, occlusionParams.windowHeight/2);
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
	glActiveTexture(GL_TEXTURE9);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE10);
	glDisable(GL_TEXTURE_2D);

}

void MyGLImageViewer::draw3DTexture(GLuint *texVBO, int index, int octreeIndex, VRParams params, int frontFBOIndex, int backFBOIndex, 
	float* cameraPos, int windowWidth, int windowHeight, MyGLCloudViewer *myGLCloudViewer, GLuint *quadVBO, int transferFunctionIndex, 
	int noiseIndex)
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

		texLoc = glGetUniformLocation(shaderProg, "clippingPlane");
		glUniform1i(texLoc, (int)params.clippingPlane);

		texLoc = glGetUniformLocation(shaderProg, "clippingPlaneLeftX");
		glUniform1f(texLoc, params.clippingPlaneLeftX);
		
		texLoc = glGetUniformLocation(shaderProg, "clippingPlaneRightX");
		glUniform1f(texLoc, params.clippingPlaneRightX);

		texLoc = glGetUniformLocation(shaderProg, "clippingPlaneUpY");
		glUniform1f(texLoc, params.clippingPlaneUpY);
		
		texLoc = glGetUniformLocation(shaderProg, "clippingPlaneDownY");
		glUniform1f(texLoc, params.clippingPlaneDownY);
		
		texLoc = glGetUniformLocation(shaderProg, "clippingPlaneFrontZ");
		glUniform1f(texLoc, params.clippingPlaneFrontZ);
		
		texLoc = glGetUniformLocation(shaderProg, "clippingPlaneBackZ");
		glUniform1f(texLoc, params.clippingPlaneBackZ);

		texLoc = glGetUniformLocation(shaderProg, "isosurfaceThreshold");
		glUniform1f(texLoc, params.isoSurfaceThreshold);

		texLoc = glGetUniformLocation(shaderProg, "windowWidth");
		glUniform1i(texLoc, windowWidth);

		texLoc = glGetUniformLocation(shaderProg, "windowHeight");
		glUniform1i(texLoc, windowHeight);

	}
	
	myGLCloudViewer->drawQuad(quadVBO);
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

void MyGLImageViewer::setProgram(GLuint shaderProg) 
{
	this->shaderProg = shaderProg;
}
