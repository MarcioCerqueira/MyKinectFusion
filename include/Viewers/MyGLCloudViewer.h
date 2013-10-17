#ifndef MYGLCLOUDVIEWER_H
#define MYGLCLOUDVIEWER_H

#include <GL/glew.h>
#include <stdlib.h>
#include <iostream>
#include <GL/glut.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Viewers/glm.h"
#include "MyPointCloud.h"

class MyGLCloudViewer
{
public:
	MyGLCloudViewer();
	void configureAmbient(int threshold, float *pointCloud);
	void configureLight();
	void configureOBJAmbient(int threshold);
	void configureARAmbientWithBlending(int threshold);
	void computeARModelCentroid(float *centroid);

	void drawAxis();
	void drawMesh(GLuint* VBOs, Eigen::Vector3f gTrans, Matrix3frm gRot, Eigen::Vector3f initialTranslation, float *rotationAngles, bool useShader, bool globalCoordinates);
	void drawOBJ(float *translationVector, float *rotationAngles, Eigen::Vector3f gTrans, Matrix3frm gRot, Eigen::Vector3f initialTranslation);
	void loadARModel(char *fileName);
	void loadIndices(int *indices, float *pointCloud);
	void loadVBOs(GLuint *meshVBO, int *indices, float *pointCloud, float *normalVector);
	void setEyePosition(int xEye, int yEye, int zEye);
	void setOBJScale(float scale);
	void setProgram(GLuint shaderProg);
	//void configure3DTexture(
	void updateModelViewMatrix(float *translationVector, float *rotationAngles, Eigen::Vector3f gTrans, Matrix3frm gRot, 
		Eigen::Vector3f initialTranslation, bool useTextureRotation = false, float volumeWidth = 0, float volumeHeight = 0, 
		float volumeDepth = 0);
private:
	GLMmodel* ARModel;
	float eyePos[3];
	float eyePosSpecial;
	GLuint shaderProg;
};
#endif
