// TCCKinFu.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <pcl/console/parse.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include "pcl/gpu/containers/initialization.hpp"
#include "Reconstruction.h"
#include "Viewers/MyGLImageViewer.h"
#include "Viewers/MyGLCloudViewer.h"
#include "Viewers/shader.h"
#include "Mediators/MeshGenerationMediator.h"
#include "Mediators/ColoredReconstructionMediator.h"
#include "Mediators/HeadPoseEstimationMediator.h"
#include "VolumeRendering/VRParams.h"
#include "VolumeRendering/MedicalVolume.h"
#include "VolumeRendering/MinMaxOctree.h"
#include "VolumeRendering/TransferFunction.h"
#include "VolumeRendering/TriCubicInterpolationPreFilter.h"
#include "Kinect.h"
#include "FaceDetection.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;

//Window's size
int windowWidth = 1280;
int windowHeight = 960;

//Our Objects
Kinect *kinect;
Reconstruction *reconstruction;
MyGLImageViewer *myGLImageViewer;
MyGLCloudViewer *myGLCloudViewer;
MedicalVolume *medicalVolume;
MinMaxOctree *minMaxOctree;
TransferFunction *transferFunction;
TriCubicInterpolationPreFilter *triCubicInterpolationPreFilter;
ColoredReconstructionMediator *coloredReconstructionMediator;
HeadPoseEstimationMediator *headPoseEstimationMediator;
FaceDetection *faceDetector;
	
//VBOs
GLuint texVBO[15]; 
GLuint spt[2];
GLuint meshVBO[4];
GLuint virtualFrameBuffer;
GLuint realFrameBuffer;
/* texVBO
0 -- Depth
1 -- Real RGB
2 -- Raycast
3 -- ARFromVolume (Grid)
4 -- Virtual DepthBuffer
5 -- Virtual FrameBuffer
6 -- Real DepthBuffer
7 -- ARFromVolume (Volume Rendering)
8 -- Real FrameBuffer
*/

enum
{
	REAL_DEPTH_FROM_DEPTHMAP_BO = 0,
	REAL_RGB_BO = 1,
	RAYCAST_BO = 2,
	AR_FROM_VOLUME_KINECTFUSION_BO = 3,
	VIRTUAL_DEPTH_BO = 4,
	VIRTUAL_RGB_BO = 5,
	REAL_DEPTH_FROM_DEPTHBUFFER_BO = 6,
	AR_FROM_VOLUME_RENDERING_BO = 7,
	REAL_RGB_FROM_FBO = 8, 
	MIN_MAX_OCTREE_BO = 9,
	TRANSFER_FUNCTION_BO = 10,
	NOISE_BO = 11,
	FRONT_FBO = 12,
	BACK_FBO = 13
};

int indices[640 * 480 * 6];
float pointCloud[640 * 480 * 3];
float normalVector[640 * 480 * 3];

//AR (General attributes)
int vel = 15;
float scale[3];
float translationVector[3];
float rotationAngles[3];
bool translationOn = false;
bool rotationOn = false;
bool scaleOn = false;
bool earlyRayTerminationOn = false;
bool stepSizeModificationOn = false;
bool isoSurfaceThresholdModificationOn = false;
bool ksOn = false;
bool ktOn = false;
bool AR = false;
bool ARConfiguration = false;

//AR Configuration (Polygonal model)
bool ARPolygonal = false;
char ARObjectFileName[1000];

//AR Configuration (Volumetric model)
char volumetricPath[1000];
char volumeConfigurationFile[1000];
int firstSlice, lastSlice;
char volumetricPathExtension[3];
bool ARVolumetric = false;
VRParams vrparams;
int VRShaderID = 2;

bool ARKinectFusionVolume = true;

bool integrateColors = false;
bool isHeadPoseEstimationEnabled = false;
bool hasFaceDetection = false;
bool faceDetected = false;
bool shader=true;

bool showCloud = false;
bool showRaycasting = true;
bool showDepthMap = true;
bool showRGBMap = true;

//
// Global handles for the currently active program object, with its two shader objects
//
GLuint ProgramObject = 0;
GLuint VertexShaderObject = 0;
GLuint FragmentShaderObject = 0;

GLuint shaderVS, shaderFS, shaderProg[7];   // handles to objects
GLint  linked;

int w1 = 1, w2 = 0, w3 = 120; 
int workAround = 0;

//  The number of frames
int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

void calculateFPS() {

	frameCount++;
	currentTime = glutGet(GLUT_ELAPSED_TIME);

    int timeInterval = currentTime - previousTime;

    if(timeInterval > 1000) {
        fps = frameCount / (timeInterval / 1000.0f);
        previousTime = currentTime;
        frameCount = 0;
		std::cout << "FPS: " << fps << std::endl;
    }

}

void printHelp() {

	std::cout << "Help " << std::endl;
	std::cout << "--cloud: Show Cloud " << std::endl;
	std::cout << "--mesh: Show Mesh extracted from MC " << std::endl;
	std::cout << "--color: Enable color integration " << std::endl;
	std::cout << "--hpefile config.txt: Head Pose Estimation Config File " << std::endl; 
	std::cout << "--threshold value: Depth Threshold for depth map truncation " << std::endl;
	std::cout << "--face cascades.txt: Enable face detection " << std::endl;
	
	std::cout << "On the fly.. " << std::endl;
	std::cout << "Press 'h' to enable head pose tracking " << std::endl;
	std::cout << "Press 'p' to stop the head pose tracking " << std::endl;
	std::cout << "Press 'c' to continue the head pose tracking " << std::endl;
	std::cout << "Press 's' to save point cloud or mesh (if --mesh is enabled) " << std::endl;
	std::cout << "Press 'a' to enable AR application " << std::endl;

}

void saveModel()
{

	int op;
	std::cout << "Saving Model..." << std::endl;
	std::cout << "Do you want to save a point cloud (.pcd) or a mesh (.ply)? (0: point cloud; 1: mesh)" << std::endl;
	std::cin >> op;
	if(op == 0) {
		if(integrateColors)
			coloredReconstructionMediator->savePointCloud(reconstruction->getTsdfVolume());
		else
			reconstruction->savePointCloud();
	} else {
		MeshGenerationMediator mgm;
		mgm.saveMesh(reconstruction->getTsdfVolume());
	}
}

void setScale(int index, bool up)
{
	
	float invScale[3];
	invScale[0] = 1.f/scale[0];
	invScale[1] = 1.f/scale[1];
	invScale[2] = 1.f/scale[2];

	if(ARPolygonal)
		myGLCloudViewer->setOBJScale(invScale);
	if(up)
		if(scale[index] < 2)
			scale[index] *= 2;
		else
			scale[index] += vel;
	else 
		if(scale[index] < 2)
			scale[index] /= 2;
		else
			scale[index] -= vel;
	if(ARPolygonal)
		myGLCloudViewer->setOBJScale(scale);

}

void positionVirtualObject(int x, int y)
{
	//We compute the translationVector necessary to move the virtual object
	float virtualCentroid[3];
	if(ARPolygonal)
		myGLCloudViewer->computeARModelCentroid(virtualCentroid);
	if(ARVolumetric) {
		virtualCentroid[0] = 0.5f; virtualCentroid[1] = 0.5f; virtualCentroid[2] = 0.5f;
	}

	//We need to update the centroid
	virtualCentroid[0] += translationVector[0];
	virtualCentroid[1] += translationVector[1];
	virtualCentroid[2] += translationVector[2];

	y = windowHeight/2 - (y - windowHeight/2);
	y = windowHeight/2 - y;

	int pixel = y * windowWidth/2 + x;
	
	float cx = reconstruction->getIntrinsics().cx;
	float cy = reconstruction->getIntrinsics().cy;
	float fx = reconstruction->getIntrinsics().fx;
	float fy = reconstruction->getIntrinsics().fy;
	//If the chosen point is visible
	if(pixel >= 0 && pixel < (640 * 480)) {
		if(reconstruction->getCurrentDepthMap()[pixel] != 0) {
	
			float xp = (float)(x - cx) * reconstruction->getCurrentDepthMap()[pixel]/fx;
			float yp = (float)(y - cy) * reconstruction->getCurrentDepthMap()[pixel]/fy;
			float zp = reconstruction->getCurrentDepthMap()[pixel];
					
			//yp *= -1;
				
			translationVector[0] += xp - virtualCentroid[0];
			translationVector[1] += yp - virtualCentroid[1];
			translationVector[2] += zp - virtualCentroid[2];
				
		}
	}

}

void loadArguments(int argc, char **argv, Reconstruction *reconstruction)
{
	//Default arguments
	char fileName[100];
	char hpeConfigFileName[100];
	char cascadeFileName[100];
	char aux[5];
	int begin = 0;
	int end = 0;
	int threshold = 5000;
	
	if(pcl::console::find_argument(argc, argv, "--cloud") >= 0) {
	showCloud = true;
	}
	if(pcl::console::find_argument(argc, argv, "--color") >= 0) {
	integrateColors = true;
	coloredReconstructionMediator = new ColoredReconstructionMediator(reconstruction->getVolumeSize());
	}
	if(pcl::console::find_argument(argc, argv, "-h") >= 0) {
	printHelp();
	}
	if(pcl::console::parse(argc, argv, "--threshold", aux) >= 0) {
	threshold = atoi(aux);
	}
	if(pcl::console::parse(argc, argv, "--hpefile", hpeConfigFileName) >= 0) {
	isHeadPoseEstimationEnabled = true;
	headPoseEstimationMediator = new HeadPoseEstimationMediator(hpeConfigFileName);
	reconstruction->setHeadPoseEstimationMediatorPointer((void*)headPoseEstimationMediator);
	}
	if(pcl::console::parse(argc, argv, "--face", cascadeFileName) >= 0) {
	hasFaceDetection = true;
	faceDetector = new FaceDetection(cascadeFileName);
	}
	if(pcl::console::parse(argc, argv, "--arobject", ARObjectFileName) >= 0) {
		ARPolygonal = true;
		ARVolumetric = false;
		ARKinectFusionVolume = false;
	}
	if(pcl::console::parse(argc, argv, "--volfile", volumeConfigurationFile) >= 0) {
		//sprintf(volumetricPathExtension, "%s", "tif");
		ARVolumetric = true;
		ARPolygonal = false;
		ARKinectFusionVolume = false;
		std::fstream file(volumeConfigurationFile);
		std::string line;
		if(file.is_open()) {
			std::getline(file, line);
			strcpy(volumetricPath, line.c_str());
			std::getline(file, line);
			firstSlice = atoi(line.c_str());
			std::getline(file, line);
			lastSlice = atoi(line.c_str());
			std::getline(file, line);
			strcpy(volumetricPathExtension, line.c_str());
			std::getline(file, line);
			vrparams.scaleWidth = atof(line.c_str());
			std::getline(file, line);
			vrparams.scaleHeight = atof(line.c_str());
			std::getline(file, line);
			vrparams.scaleDepth = atof(line.c_str());
			std::getline(file, line);
			vrparams.rotationX = atoi(line.c_str());
			std::getline(file, line);
			vrparams.rotationY = atoi(line.c_str());
			std::getline(file, line);
			vrparams.rotationZ = atoi(line.c_str());
			std::cout << "Path: " << volumetricPath << std::endl;
			std::cout << "Number of the first slice: " << firstSlice << std::endl;
			std::cout << "Number of the last slice: " << lastSlice << std::endl;
			std::cout << "Extension: " << volumetricPathExtension << std::endl;
			std::cout << "Scale factor: x (" << vrparams.scaleWidth << "), y(" << vrparams.scaleHeight << "), z(" << vrparams.scaleDepth << ")" << std::endl;
		} else {
			std::cout << "File " << volumeConfigurationFile << " could not be opened" << std::endl;
		}

	}

	//Initialize reconstruction with arguments
	reconstruction->setThreshold(threshold);

}

void reshape(int w, int h)
{
	windowWidth = w;
	windowHeight = h;

	glViewport( 0, 0, windowWidth, windowHeight );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D( 0, windowWidth, 0, windowHeight );
	glMatrixMode( GL_MODELVIEW );

}

void displayDepthData()
{
	glViewport(0, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    

	myGLImageViewer->loadDepthTexture(reconstruction->getCurrentDepthMap(), texVBO, REAL_DEPTH_FROM_DEPTHMAP_BO, 
		reconstruction->getThreshold(), kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, REAL_DEPTH_FROM_DEPTHMAP_BO, windowWidth, windowHeight);

}

void displayRGBData()
{
	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 

	myGLImageViewer->loadRGBTexture(reconstruction->getRGBMap(), texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, REAL_RGB_BO, windowWidth, windowHeight);

}

void displayRaycastedData()
{

	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    

	myGLImageViewer->loadRGBTexture(reconstruction->getRaycastImage(), texVBO, RAYCAST_BO, kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, RAYCAST_BO, windowWidth, windowHeight);

}

void displayCloud(bool globalCoordinates = true)
{
	
	reconstruction->getPointCloud(pointCloud, globalCoordinates);
	reconstruction->getNormalVector(normalVector, globalCoordinates);

	myGLCloudViewer->loadIndices(indices, pointCloud);
	myGLCloudViewer->loadVBOs(meshVBO, indices, pointCloud, normalVector);
	
	glViewport(0, 0, windowWidth/2, windowHeight/2);

	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 
	
	myGLCloudViewer->configureAmbient(reconstruction->getThreshold());
	if(reconstruction->getGlobalTime() > 1)
		myGLCloudViewer->drawMesh(meshVBO, reconstruction->getCurrentTranslation(), reconstruction->getCurrentRotation(), reconstruction->getInitialTranslation(), 
			rotationAngles, shader, globalCoordinates);
	if(workAround == 1) {
		myGLCloudViewer->drawMesh(meshVBO, Eigen::Vector3f::Zero(), Eigen::Matrix3f::Identity(), reconstruction->getInitialTranslation(), rotationAngles, shader, 
			globalCoordinates);
		workAround = 2;
	}
}

void displayARDataFromVolume()
{
	
	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayCloud(true);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
			
	glBindFramebuffer(GL_FRAMEBUFFER, realFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayCloud(false);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//Second Viewport: Virtual Object
	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 

	myGLImageViewer->loadRGBTexture(reconstruction->getRaycastImage(), texVBO, RAYCAST_BO, kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, RAYCAST_BO, windowWidth, windowHeight);

	//Fourth Viewport: Virtual + Real Object
	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 

	myGLImageViewer->loadRGBTexture(reconstruction->getRGBMap(), texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());

	myGLImageViewer->setProgram(shaderProg[1]);
	myGLImageViewer->drawARTextureWithOcclusion(texVBO, REAL_RGB_BO, REAL_DEPTH_FROM_DEPTHBUFFER_BO, VIRTUAL_RGB_BO, VIRTUAL_DEPTH_BO, windowWidth, windowHeight);
	
}

void displayARDataFromOBJFile()
{
	
	if(ARConfiguration)
		displayCloud(true);

	glViewport(0, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 
	myGLCloudViewer->configureAmbient(reconstruction->getThreshold());
	
	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	myGLCloudViewer->drawOBJ(translationVector, rotationAngles, reconstruction->getCurrentTranslation(), reconstruction->getCurrentRotation(), 
		reconstruction->getInitialTranslation());
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	glBindFramebuffer(GL_FRAMEBUFFER, realFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayCloud(false);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//First Viewport: Only the virtual object
	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 
	
	if(ARConfiguration) {

		myGLCloudViewer->configureOBJAmbient(reconstruction->getThreshold());
		myGLCloudViewer->drawOBJ(translationVector, rotationAngles, reconstruction->getCurrentTranslation(), reconstruction->getCurrentRotation(), 
			reconstruction->getInitialTranslation());

	}

	myGLImageViewer->loadRGBTexture(reconstruction->getRGBMap(), texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());
	
	//Third Viewport: Virtual + Real Object
	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();   
	
	myGLImageViewer->setProgram(shaderProg[1]);
	myGLImageViewer->drawARTextureWithOcclusion(texVBO, REAL_RGB_BO, REAL_DEPTH_FROM_DEPTHBUFFER_BO, VIRTUAL_RGB_BO, VIRTUAL_DEPTH_BO, windowWidth, windowHeight, true);
	
}

void displayARDataFromVolumeRendering()
{
	
	glEnable(GL_DEPTH_TEST);

	if(ARConfiguration)
		displayCloud(true);
			
	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayCloud(true);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
			
	glBindFramebuffer(GL_FRAMEBUFFER, realFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayCloud(false);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	glViewport(0, windowHeight/2, windowWidth/2, windowHeight/2);
	myGLCloudViewer->configureQuadAmbient(reconstruction->getThreshold());
	
	myGLCloudViewer->updateModelViewMatrix(translationVector, rotationAngles, reconstruction->getCurrentTranslation(), 
		reconstruction->getCurrentRotation(), reconstruction->getInitialTranslation(), true, medicalVolume->getWidth(), 
		medicalVolume->getHeight(), medicalVolume->getDepth(), vrparams.scaleWidth, vrparams.scaleHeight, vrparams.scaleDepth, 
		vrparams.rotationX, vrparams.rotationY, vrparams.rotationZ);
	glScalef(scale[0], scale[1], scale[2]);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	myGLImageViewer->drawQuads(1.0f/vrparams.scaleWidth, 1.0f/vrparams.scaleHeight, 1.0f/vrparams.scaleDepth);
	myGLImageViewer->loadFrameBufferTexture(texVBO, BACK_FBO, 0, windowHeight/2, windowWidth/2, windowHeight/2);
	glDisable(GL_CULL_FACE);
	
	myGLImageViewer->drawQuads(1.0f/vrparams.scaleWidth, 1.0f/vrparams.scaleHeight, 1.0f/vrparams.scaleDepth);
	myGLImageViewer->loadFrameBufferTexture(texVBO, FRONT_FBO, 0, windowHeight/2, windowWidth/2, windowHeight/2);
	glPopMatrix();
	
	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glEnable(GL_TEXTURE_3D);
	myGLCloudViewer->configureARAmbientWithBlending(reconstruction->getThreshold());
	myGLCloudViewer->setAmbientIntensity(0);
	myGLCloudViewer->setDiffuseIntensity(0.01);
	myGLCloudViewer->setSpecularIntensity(0.01);

	myGLImageViewer->setProgram(shaderProg[VRShaderID]);
	myGLCloudViewer->updateModelViewMatrix(translationVector, rotationAngles, reconstruction->getCurrentTranslation(), 
		reconstruction->getCurrentRotation(), reconstruction->getInitialTranslation(), true, medicalVolume->getWidth(), 
		medicalVolume->getHeight(), medicalVolume->getDepth(), vrparams.scaleWidth, vrparams.scaleHeight, vrparams.scaleDepth, 
		vrparams.rotationX, vrparams.rotationY, vrparams.rotationZ);
	glScalef(scale[0], scale[1], scale[2]);

	myGLImageViewer->draw3DTexture(texVBO, AR_FROM_VOLUME_RENDERING_BO, MIN_MAX_OCTREE_BO, vrparams, FRONT_FBO, BACK_FBO, 
		myGLCloudViewer->getEyePosition(), windowWidth/2, windowHeight/2, TRANSFER_FUNCTION_BO, NOISE_BO);
	glPopMatrix();

	glDisable(GL_BLEND);
	glDisable(GL_ALPHA_TEST);	
	glEnable(GL_DEPTH_TEST);
	
	//Fourth Viewport: Virtual + Real Object
	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();

	myGLImageViewer->loadRGBTexture(reconstruction->getRGBMap(), texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->loadFrameBufferTexture(texVBO, VIRTUAL_RGB_BO, windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);

	myGLImageViewer->setProgram(shaderProg[1]);
	myGLImageViewer->drawARTextureWithOcclusion(texVBO, REAL_RGB_BO, REAL_DEPTH_FROM_DEPTHBUFFER_BO, VIRTUAL_RGB_BO, VIRTUAL_DEPTH_BO, windowWidth, windowHeight);

}

void display()
{
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	if(!AR)
	{
		if(workAround == 1)
			displayCloud();
		if(showDepthMap)
			displayDepthData();
		if(showRGBMap)
			displayRGBData();
		if(showRaycasting && reconstruction->hasImage())
			displayRaycastedData();
		if(showCloud && reconstruction->hasImage())
			if(ARPolygonal)
				displayCloud(!reconstruction->isOnlyTrackingOn());
			else
				displayCloud();
	} else {
		if(showCloud)
			displayCloud(ARConfiguration);
		if(ARPolygonal)
			displayARDataFromOBJFile();
		if(ARVolumetric)
			displayARDataFromVolumeRendering();
		if(ARKinectFusionVolume)			
			displayARDataFromVolume();
	}
	
	glutSwapBuffers();
	glutPostRedisplay();

}

void idle()
{
	
	bool preGlobalTimeGreaterThanZero;
	if(kinect->grabFrame()) {
		if(hasFaceDetection) {
			if(faceDetected) {
				reconstruction->stopTracking(false);
				faceDetector->segmentFace(kinect->getRGBImage(), kinect->getDepthImage());
			} else {
				reconstruction->reset();
				reconstruction->stopTracking(true);
				faceDetected = faceDetector->run(kinect->getRGBImage(), kinect->getDepthImage());
				if(faceDetected)
					faceDetector->segmentFace(kinect->getRGBImage(), kinect->getDepthImage());
			}
			//to check if the reconstruction was reseted
			if(reconstruction->getGlobalTime() > 0) preGlobalTimeGreaterThanZero = true;
			else preGlobalTimeGreaterThanZero = false;
		}
		if(!ARConfiguration)
			reconstruction->run(kinect->getRGBImage(), kinect->getDepthImage()); 
		//if the reconstruction was reseted, the face is no more detected
		if(hasFaceDetection)
			if(reconstruction->getGlobalTime() == 0 && preGlobalTimeGreaterThanZero)
				faceDetected = false;
		if(!ARConfiguration && !AR && integrateColors)
			coloredReconstructionMediator->updateColorVolume(reconstruction);
		if(workAround != 2)
			workAround = 1;
	}
	calculateFPS();

}

void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case 27:
		exit(0);
		break;
	case (int)'i' : case (int)'I':
		std::cout << "Head Pose Tracking Activated..." << std::endl;
		reconstruction->enableOnlyTracking();
		hasFaceDetection = false;
		break;
	case (int)'p' : case (int)'P':
		std::cout << "Pause..." << std::endl;
		reconstruction->stopTracking(true);
		if(isHeadPoseEstimationEnabled)
			headPoseEstimationMediator->stopTracking(true, reconstruction);
		break;
	case (int)'c' : case (int)'C':
		std::cout << "Continue..." << std::endl;
		reconstruction->stopTracking(false);
		if(isHeadPoseEstimationEnabled)
			headPoseEstimationMediator->stopTracking(false, reconstruction);
		break;
	case (int)'s' : case (int)'S':
		if(AR && scaleOn) {
			if(key == 's') {
				setScale(0, true);
				setScale(1, true);
				setScale(2, true);
			} else {
				setScale(0, false);
				setScale(1, false);
				setScale(2, false);
			}
		}
		break;
	case (int)'a' : case (int)'A':
		if(!AR) {
			
			translationVector[0] = -(reconstruction->getInitialTranslation()(0) - reconstruction->getCurrentTranslation()(0));
			translationVector[1] = -(reconstruction->getInitialTranslation()(1) - reconstruction->getCurrentTranslation()(1));
			translationVector[2] = -(reconstruction->getInitialTranslation()(2) - reconstruction->getCurrentTranslation()(2));
			AR = true;
			ARConfiguration = true;
			
			if(ARPolygonal || ARKinectFusionVolume)
				glEnable(GL_DEPTH_TEST);
			
			reconstruction->enableOnlyTracking();
			hasFaceDetection = false;
			std::cout << "Enabling AR" << std::endl;
			std::cout << "AR Enabled: Click the window to position the object (if necessary, use the scale factor (s)" << std::endl;
			myGLCloudViewer->setEyePosition(1, 0, 120);

		} else if(ARConfiguration) {
			ARConfiguration = false;
			std::cout << "AR Enabled: Configuration finished" << std::endl;
			myGLCloudViewer->setEyePosition(1, 0, 170);
		} else if(AR && !ARConfiguration){
			AR = false;
		}
		break;
	case (int)'r' : case (int)'R':
		reconstruction->reset();
		faceDetected = false;
		break;
	case (int)'u':
		shader = !shader;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void specialKeyboard(int key, int x, int y)
{

	switch (key)
	{
	case GLUT_KEY_UP:
		if(translationOn)
			translationVector[1] += vel;
		if(rotationOn)
			rotationAngles[1] += vel;
		if(scaleOn)
			setScale(1, true);
		if(earlyRayTerminationOn)
			vrparams.earlyRayTerminationThreshold += 0.01;
		if(stepSizeModificationOn)
			vrparams.stepSize += 1.0/2048.0;
		if(isoSurfaceThresholdModificationOn)
			vrparams.isoSurfaceThreshold += 0.05;
		if(ksOn)
			vrparams.ks += 0.01;
		if(ktOn)
			vrparams.kt += 1;
		break;
	case GLUT_KEY_DOWN:
		if(translationOn)
			translationVector[1] -= vel;
		if(rotationOn)
			rotationAngles[1] -= vel;
		if(scaleOn)
			setScale(1, false);
		if(earlyRayTerminationOn)
			vrparams.earlyRayTerminationThreshold -= 0.01;
		if(stepSizeModificationOn) {
			vrparams.stepSize -= 1.0/2048.0;
			if(vrparams.stepSize <= 0) vrparams.stepSize = 0;
		}
		if(isoSurfaceThresholdModificationOn)
			vrparams.isoSurfaceThreshold -= 0.05;
		if(ksOn)
			vrparams.ks -= 0.01;
		if(ktOn)
			vrparams.kt -= 1;
		break;
	case GLUT_KEY_LEFT:
		if(translationOn)
			translationVector[0] -= vel;
		if(rotationOn)
			rotationAngles[0] -= vel;
		if(scaleOn)
			setScale(0, false);
		break;
	case GLUT_KEY_RIGHT:
		if(translationOn)
			translationVector[0] += vel;
		if(rotationOn)
			rotationAngles[0] += vel;
		if(scaleOn)
			setScale(0, true);
		break;
	case GLUT_KEY_PAGE_UP:
		if(translationOn)
			translationVector[2] += vel;
		if(rotationOn)
			rotationAngles[2] += vel;
		if(scaleOn)
			setScale(2, true);
		break;
	case GLUT_KEY_PAGE_DOWN:
		if(translationOn)
			translationVector[2] -= vel;
		if(rotationOn)
			rotationAngles[2] -= vel;
		if(scaleOn)
			setScale(2, false);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) 
{
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_UP)
			if(ARConfiguration)
				positionVirtualObject(x, y);

	glutPostRedisplay();
}

void mainMenu(int id)
{
}

void volumeRenderingMenu(int id)
{
	switch(id)
	{
	case 0:
		vrparams.stochasticJithering = !vrparams.stochasticJithering;
		break;
	case 1:
		vrparams.triCubicInterpolation = !vrparams.triCubicInterpolation;
		break;
	case 2:
		vrparams.MIP = !vrparams.MIP;
		break;
	case 3:
		vrparams.NonPolygonalIsoSurface = !vrparams.NonPolygonalIsoSurface;
		VRShaderID = 6;
		break;
	case 4:
		vrparams.gradientByForwardDifferences = !vrparams.gradientByForwardDifferences;
		break;
	case 5:
		VRShaderID = 2;
		break;
	case 6:
		VRShaderID = 4;
		break;
	case 7:
		VRShaderID = 5;
		break;
	case 8:
		VRShaderID = 3;
		break;
	}
}

void thresholdMenu(int id)
{
	switch(id)
	{
	case 0:
		translationOn = false;
		rotationOn = false;
		scaleOn = false;
		earlyRayTerminationOn = true;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = false;
		ksOn = false;
		ktOn = false;
		break;
	case 1:
		translationOn = false;
		rotationOn = false;
		scaleOn = false;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = true;
		isoSurfaceThresholdModificationOn = false; 
		ksOn = false;
		ktOn = false;
		break;
	case 2:
		translationOn = false;
		rotationOn = false;
		scaleOn = false;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = true; 
		ksOn = false;
		ktOn = false;
		break;
	case 3:
		translationOn = false;
		rotationOn = false;
		scaleOn = false;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = false; 
		ksOn = true;
		ktOn = false;
		break;
	case 4:
		translationOn = false;
		rotationOn = false;
		scaleOn = false;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = false; 
		ksOn = false;
		ktOn = true;
		break;
	}
}

void transformationMenu(int id)
{
	switch(id)
	{
	case 0:
		translationOn = true;
		rotationOn = false;
		scaleOn = false;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = false;
		ksOn = false;
		ktOn = false;
		break;
	case 1:
		translationOn = false;
		rotationOn = true;
		scaleOn = false;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = false;
		ksOn = false;
		ktOn = false;
		break;
	case 2:
		translationOn = false;
		rotationOn = false;
		scaleOn = true;
		earlyRayTerminationOn = false;
		stepSizeModificationOn = false;
		isoSurfaceThresholdModificationOn = false;
		ksOn = false;
		ktOn = false;
		break;
	}
}

void otherFunctionsMenu(int id)
{
	switch(id)
	{
	case 0:
		saveModel();
		break;
	}
}

void createMenu()
{

	GLint volumeRenderingMenuID, thresholdMenuID, transformationMenuID, otherFunctionsMenuID;

	volumeRenderingMenuID = glutCreateMenu(volumeRenderingMenu);
		glutAddMenuEntry("Stochastic Jithering [On/Off]", 0);
		glutAddMenuEntry("Tricubic Interpolation [On/Off]", 1);
		glutAddMenuEntry("MIP [On/Off]", 2);
		glutAddMenuEntry("Non Polygonal Iso Surface Rendering", 3);
		glutAddMenuEntry("Gradient by Forward Differences [On/Off]", 4);
		glutAddMenuEntry("Default Volume Rendering", 5);
		glutAddMenuEntry("Only Transfer Function", 6);
		glutAddMenuEntry("Transfer Function + Local Illumination", 7);
		glutAddMenuEntry("Context-Preserving Volume Rendering", 8);
	
	thresholdMenuID = glutCreateMenu(thresholdMenu);
		glutAddMenuEntry("Change Early Ray Termination", 0);
		glutAddMenuEntry("Change Step Size (Raycasting)", 1);
		glutAddMenuEntry("Change Iso Surface", 2);
		glutAddMenuEntry("Change Ks (Context-Preserving VR)", 3);
		glutAddMenuEntry("Change Kt (Context-Preserving VR)", 4);

	transformationMenuID = glutCreateMenu(transformationMenu);
		glutAddMenuEntry("Change Translation", 0);
		glutAddMenuEntry("Change Rotation", 1);
		glutAddMenuEntry("Change Scale", 2);

	otherFunctionsMenuID = glutCreateMenu(otherFunctionsMenu);
		glutAddMenuEntry("Save Model", 0);

	glutCreateMenu(mainMenu);
		glutAddSubMenu("Transformation", transformationMenuID);
		glutAddSubMenu("Volume Rendering", volumeRenderingMenuID);
		glutAddSubMenu("Threshold", thresholdMenuID);
		glutAddSubMenu("Other Functions", otherFunctionsMenuID);

		glutAttachMenu(GLUT_RIGHT_BUTTON);

}

void init()
{
	
	//initialize some conditions
	glClearColor( 0.0f, 0.0f, 0.0f, 0.0 );
	glShadeModel(GL_SMOOTH);
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1);  

	//buffer objects
	if(texVBO[0] == 0)
		glGenTextures(15, texVBO);
	if(meshVBO[0] == 0)
		glGenBuffers(4, meshVBO);
	if(virtualFrameBuffer == 0)
		glGenFramebuffers(1, &virtualFrameBuffer);
	if(realFrameBuffer == 0)
		glGenFramebuffers(1, &realFrameBuffer);

	myGLImageViewer = new MyGLImageViewer();
	myGLCloudViewer = new MyGLCloudViewer();
	myGLCloudViewer->setEyePosition(1, 0, 120);

	if(ARPolygonal) {
	
		//load the polygonal model
		myGLCloudViewer->loadARModel(ARObjectFileName);
	
	} else if(ARVolumetric) {

		medicalVolume = new MedicalVolume();

		//load the medical volume
		if(!strcmp(volumetricPathExtension, "tif"))
			medicalVolume->loadTIFData(volumetricPath, firstSlice, lastSlice);
		else if(!strcmp(volumetricPathExtension, "pgm"))
			medicalVolume->loadPGMData(volumetricPath, firstSlice, lastSlice);
		else
			medicalVolume->loadRAWData(volumetricPath, firstSlice, lastSlice, lastSlice);
	
		//(Optional) improve the accuracy of the tricubic interpolation
		//triCubicInterpolationPreFilter = new TriCubicInterpolationPreFilter();
		//triCubicInterpolationPreFilter->applyPreFilterForAccurateCubicBSplineInterpolation(medicalVolume->getData(), 
			//medicalVolume->getWidth(), medicalVolume->getHeight(), medicalVolume->getDepth());

		//build a min-max octree to allow the use of empty-space skipping and adaptive sampling
		minMaxOctree = new MinMaxOctree(medicalVolume->getWidth(), medicalVolume->getHeight(), medicalVolume->getDepth());
		minMaxOctree->build(medicalVolume->getData(), medicalVolume->getWidth(), medicalVolume->getHeight(), medicalVolume->getDepth());
		
		//load both textures
		myGLImageViewer->load3DTexture(medicalVolume->getData(), texVBO, AR_FROM_VOLUME_RENDERING_BO, medicalVolume->getWidth(), 
			medicalVolume->getHeight(), medicalVolume->getDepth());
		myGLImageViewer->load3DTexture(minMaxOctree->getData(), texVBO, MIN_MAX_OCTREE_BO, minMaxOctree->getWidth(), 
			minMaxOctree->getHeight(), minMaxOctree->getDepth());
		
		//compute a transfer function to map the scalar value to some color
		transferFunction = new TransferFunction();
		transferFunction->load();
		transferFunction->computePreIntegrationTable();
		myGLImageViewer->loadRGBATexture(transferFunction->getPreIntegrationTable(), texVBO, TRANSFER_FUNCTION_BO, 256, 256);
		
		//compute a noise texture to allow the use of stochastic jittering (i.e. random ray start)
		myGLImageViewer->load2DNoiseTexture(texVBO, NOISE_BO, 32, 32);
		
		//initialize some parameters
		vrparams.stepSize = 1.0/50.0;
		vrparams.earlyRayTerminationThreshold = 0.95;
		vrparams.kt = 1;
		vrparams.ks = 0;
		vrparams.stochasticJithering = false;
		vrparams.triCubicInterpolation = false;
		vrparams.MIP = false;
		vrparams.gradientByForwardDifferences = false;
		vrparams.NonPolygonalIsoSurface = false;
		vrparams.isoSurfaceThreshold = 0.1;

		createMenu();
	
	}

	myGLImageViewer->loadDepthComponentTexture(NULL, texVBO, VIRTUAL_DEPTH_BO, windowWidth, windowHeight);
	myGLImageViewer->loadDepthComponentTexture(NULL, texVBO, REAL_DEPTH_FROM_DEPTHBUFFER_BO, windowWidth, windowHeight);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, VIRTUAL_RGB_BO, windowWidth/2, windowHeight/2);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, REAL_RGB_FROM_FBO, windowWidth/2, windowHeight/2);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, FRONT_FBO, windowWidth/2, windowHeight/2);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, BACK_FBO, windowWidth/2, windowHeight/2);

	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texVBO[VIRTUAL_DEPTH_BO], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texVBO[VIRTUAL_RGB_BO], 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
		std::cout << "FBO OK" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, realFrameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texVBO[REAL_DEPTH_FROM_DEPTHBUFFER_BO], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texVBO[REAL_RGB_FROM_FBO], 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
		std::cout << "FBO OK" << std::endl;

	//AR Configuration
	translationVector[0] = 0;
	translationVector[1] = 0;
	translationVector[2] = 0;
	rotationAngles[0] = 0;
	rotationAngles[1] = 0;
	rotationAngles[2] = 0;
	scale[0] = 1;
	scale[1] = 1;
	scale[2] = 1;

}

int main(int argc, char **argv) {

  pcl::gpu::setDevice (0);
  pcl::gpu::printShortCudaDeviceInfo (0);

  //This argument is an exception. It is loaded first because it is necessary to instantiate the Reconstruction object
  Eigen::Vector3i volumeSize(3000, 3000, 3000); //mm
  if(pcl::console::parse_3x_arguments(argc, argv, "--volumesize", volumeSize(0), volumeSize(1), volumeSize(2)) >= 0) {
  }
  
  try
  {
	//Initialize the Reconstruction object
	reconstruction = new Reconstruction(volumeSize);
	kinect = new Kinect();
	loadArguments(argc, argv, reconstruction);
	
	//Initialize the GL window
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_ALPHA);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInit(&argc, argv);
	glutCreateWindow("My KinFu");

	//Initialize glew
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cout << "Error: " << glewGetErrorString(err) << std::endl;
		exit(0);
	}
	init();

	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutSpecialFunc(specialKeyboard);

	initShader("Shaders/Phong", 0);
	initShader("Shaders/Occlusion", 1);
	initShader("Shaders/VRBlendRaycasting", 2);
	initShader("Shaders/VRContextPreservingPreIntegrationRaycasting", 3);
	initShader("Shaders/VRPreIntegrationRaycasting", 4);
	initShader("Shaders/VRLocalIlluminationPreIntegrationRaycasting", 5);
	initShader("Shaders/VRNonPolygonalRaycasting", 6);

	myGLCloudViewer->setProgram(shaderProg[0]);
	myGLImageViewer->setProgram(shaderProg[1]);
	
	glutMainLoop();

  } 
  catch (const std::bad_alloc& /*e*/)
  {
    cout << "Bad alloc" << endl;
  }
  catch (const std::exception& /*e*/)
  {
    cout << "Exception" << endl;
  }

  delete kinect;
  delete reconstruction;
  delete myGLImageViewer;
  delete myGLCloudViewer;
  if(ARVolumetric) {
	  delete medicalVolume;
	  delete minMaxOctree;
	  delete transferFunction;
	  delete triCubicInterpolationPreFilter;
  }
  if(integrateColors)
	  delete coloredReconstructionMediator;
  if(isHeadPoseEstimationEnabled)
	  delete headPoseEstimationMediator;
  if(hasFaceDetection)
	  delete faceDetector;
  return 0;

}
