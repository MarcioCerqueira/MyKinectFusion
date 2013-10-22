#include "Viewers/MyGLImageViewer.h"

MyGLImageViewer::MyGLImageViewer()
{
	depthData = (unsigned char*)malloc(640 * 480 * 3 * sizeof(unsigned char));
	frameBuffer = (unsigned char*)malloc(640 * 480 * 3 * sizeof(unsigned char));
	depthBuffer = (float*)malloc(640 * 480 * sizeof(float));
	auxDepthBuffer = (float*)malloc(640 * 480 * sizeof(float));
}

MyGLImageViewer::~MyGLImageViewer()
{
	delete [] depthData;
	delete [] frameBuffer;
	delete [] depthBuffer;
	delete [] auxDepthBuffer;
}

void MyGLImageViewer::loadDepthTexture(unsigned short *data, GLuint *texVBO, int index, int threshold, int windowWidth, int windowHeight)
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
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, windowWidth, windowHeight,  
		//0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth/2, windowHeight/2, 
		0, GL_RGB, GL_UNSIGNED_BYTE, depthData);
}

void MyGLImageViewer::loadDepthBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height)
{

	glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, auxDepthBuffer);


	int xp, yp, inversePixel;
	for(int pixel = 0; pixel < (640 * 480); pixel++) {
		xp = pixel % 640;
		yp = pixel / 640;
		inversePixel = (480 - yp) * 640 + xp;
		auxDepthBuffer[inversePixel] = (auxDepthBuffer[inversePixel] - 0.999) * 1000;
		depthBuffer[pixel] = auxDepthBuffer[inversePixel];
	}

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height,  
		0, GL_LUMINANCE, GL_FLOAT, depthBuffer);

}
	
void MyGLImageViewer::loadRGBTexture(const unsigned char *data, GLuint *texVBO, int index, int windowWidth, int windowHeight)
{

	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth/2, windowHeight/2,  
		0, GL_RGB, GL_UNSIGNED_BYTE, data);

}

void MyGLImageViewer::loadFrameBufferTexture(GLuint *texVBO, int index, int x, int y, int width, int height)
{

	glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, depthData);
	int xp, yp, inversePixel;
	for(int pixel = 0; pixel < (640 * 480); pixel++) {
		xp = pixel % 640;
		yp = pixel / 640;
		inversePixel = (480 - yp) * 640 + xp;
		for(int ch = 0; ch < 3; ch++) {
			frameBuffer[pixel * 3 + ch] = depthData[inversePixel * 3 + ch];
		}
	}
	
	glBindTexture(GL_TEXTURE_2D, texVBO[index]);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,  
		0, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
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
	
void MyGLImageViewer::load3DTexture(const unsigned char *data, GLuint *texVBO, int index, int volumeWidth, 
	int volumeHeight, int volumeDepth)
{
	
	glBindTexture(GL_TEXTURE_3D, texVBO[index]);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, volumeWidth, volumeHeight, volumeDepth, 0, GL_RGBA, 
		GL_UNSIGNED_BYTE, data);
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
		int windowWidth, int windowHeight)
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
}

void MyGLImageViewer::draw3DTexture(GLuint *texVBO, int index)
{	
	
	glUseProgram(shaderProg);

	GLuint texLoc = glGetUniformLocation(shaderProg, "virtualRGB");
	glUniform1i(texLoc, 0);
	
	glActiveTexture(GL_TEXTURE7);
	glEnable(GL_TEXTURE_3D);
	glBindTexture(GL_TEXTURE_3D, texVBO[index]);
	float dOrthoSize = 1.0f;

	/*
	Draw textured quads along the z axis. The x-y vertex coordinates are (–1, –1), (1, –1), (1, 1), (–1, 1). 
	The corresponding x-y texture coordinates are (0, 0), (1, 0), (1, 1), (0, 1). 
	The z vertex and texture coordinates increase uniformly from –1 to 1 and 0 to 1, respectively. 
	*/

	for(float x = -1.0f; x <= 1.0f; x += 0.01f)
	{
		glBegin(GL_QUADS);
			glNormal3f(0.0, 0.0, 1.0);
			//glMultiTexCoord3f(GL_TEXTURE0, 0.0f, 0.0f, ((float)x+1.0f)/2.0f);
			glTexCoord3f(0.0f, 0.0f, ((float)x+1.0f)/2.0f);
			glVertex3f(-dOrthoSize,-dOrthoSize,x);
			glTexCoord3f(1.0f, 0.0f, ((float)x+1.0f)/2.0f);
			glVertex3f(dOrthoSize,-dOrthoSize,x);
			glTexCoord3f(1.0f, 1.0f, ((float)x+1.0f)/2.0f);
			glVertex3f(dOrthoSize,dOrthoSize,x);
			glTexCoord3f(0.0f, 1.0f, ((float)x+1.0f)/2.0f);
			glVertex3f(-dOrthoSize,dOrthoSize,x);
		glEnd();
	}

	glDisable(GL_TEXTURE_3D);
	glUseProgram(0);

}

void MyGLImageViewer::setProgram(GLuint shaderProg) 
{
	this->shaderProg = shaderProg;
}
