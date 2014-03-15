
#include <iostream>
#include <pcl/console/parse.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include "pcl/gpu/containers/initialization.hpp"
#include "Reconstruction.h"
#include "Viewers/MyGLImageViewer.h"
#include "Viewers/MyGLCloudViewer.h"
#include "Viewers/shader.h"
#include "Viewers/AROcclusionParams.h"
#include "Viewers/ModelViewParams.h"
#include "Mediators/MeshGenerationMediator.h"
#include "Mediators/ColoredReconstructionMediator.h"
#include "Mediators/HeadPoseEstimationMediator.h"
#include "VolumeRendering/VRParams.h"
#include "VolumeRendering/MedicalVolume.h"
#include "VolumeRendering/MinMaxOctree.h"
#include "VolumeRendering/TransferFunction.h"
#include "VolumeRendering/TriCubicInterpolationPreFilter.h"
#include "Kinect.h"
#include "Image.h"
#include "FaceDetection.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;

//Window's size
int windowWidth = 1280;
int windowHeight = 960;

//Our Objects
Image *imageCollection;
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
AROcclusionParams occlusionParams;
ModelViewParams modelViewParams;

GLuint texVBO[25]; 
GLuint spt[2];
GLuint meshVBO[4];
GLuint quadVBO[4];
GLuint virtualFrameBuffer;
GLuint realFrameBuffer;
GLuint frontQuadFrameBuffer;
GLuint backQuadFrameBuffer;

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
	FRONT_QUAD_RGB_FBO = 12,
	FRONT_QUAD_DEPTH_FBO = 13,
	BACK_QUAD_RGB_FBO = 14,
	BACK_QUAD_DEPTH_FBO = 15,
	CURVATURE_MAP_FBO = 16, 
	CONTOURS_FBO = 17,
	BACKGROUND_SCENE_FBO = 18,
	SUBTRACTION_MASK_FBO = 19,
	FACE_MAP_FBO = 20
};

int indices[640 * 480 * 6];
float pointCloud[640 * 480 * 3];
float normalVector[640 * 480 * 3];
float depthData[640 * 480];
float curvature[640 * 480];
unsigned short curv[640 * 480];
unsigned char backgroundScene[640 * 480 * 3];

float curvatureWeight = 0;
float distanceFalloffWeight = 10.0;
float clippingWeight = 0;
float focusPoint[2] = {0, 0};
float focusRadius = 50;

//AR (General attributes)
int vel = 5;
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
bool curvatureWeightOn = false;
bool distanceFallOffWeightOn = false;
bool clippingWeightOn = false;
bool focusRadiusOn = false;
bool clippingPlaneLeftXOn = false;
bool clippingPlaneRightXOn = false;
bool clippingPlaneUpYOn = false;
bool clippingPlaneDownYOn = false;
bool clippingPlaneFrontZOn = false;
bool clippingPlaneBackZOn = false;
bool AR = false;
bool ARConfiguration = false;

bool alphaBlendingOn = true;
bool curvatureBlendingOn = false;
bool distanceFalloffBlendingOn = false;
bool clippingBlendingOn = false;
bool subtractionMaskOn = false;

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
bool showCurvatureMap = false;
bool showContoursMap = false;

//
// Global handles for the currently active program object, with its two shader objects
//
GLuint ProgramObject = 0;
GLuint VertexShaderObject = 0;
GLuint FragmentShaderObject = 0;

GLuint shaderVS, shaderFS, shaderProg[10];   // handles to objects
GLint  linked;

int w1 = 1, w2 = 0, w3 = 120; 
int workAround = 0;

//  The number of frames
int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;
std::vector<unsigned short> sourceDepthData;

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
	//We compute the translationVector required to move the virtual object
	float virtualCentroid[3];
	if(ARPolygonal)
		myGLCloudViewer->computeARModelCentroid(virtualCentroid);
	if(ARVolumetric) {
		virtualCentroid[0] = 0; virtualCentroid[1] = 0; virtualCentroid[2] = 0;
	}

	//We need to update the centroid
	virtualCentroid[0] += translationVector[0];
	virtualCentroid[1] += translationVector[1];
	virtualCentroid[2] += translationVector[2];

	if(ARPolygonal) {
	
		y = windowHeight/2 - (y - windowHeight/2);
		y = windowHeight/2 - y;
	
		int pixel = y * windowWidth/2 + x;
	
		//intrinsics
		float cx = imageCollection->getIntrinsics().cx;
		float cy = imageCollection->getIntrinsics().cy;
		float fx = imageCollection->getIntrinsics().fx;
		float fy = imageCollection->getIntrinsics().fy;

		if(pixel >= 0 && pixel < (640 * 480)) {
			if(imageCollection->getDepthMap()[pixel] != 0) {
				
				float xp = (float)(x - cx) * imageCollection->getDepthMap()[pixel]/fx;
				float yp = (float)(y - cy) * imageCollection->getDepthMap()[pixel]/fy;
				float zp = imageCollection->getDepthMap()[pixel];

				translationVector[0] += xp - virtualCentroid[0];
				translationVector[1] += yp - virtualCentroid[1];
				translationVector[2] += zp - virtualCentroid[2];

			}
		}

	}

	if(ARVolumetric) {

		//extract the reference point cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr referenceCloud = reconstruction->extractFullPointCloud();
		Eigen::Vector3f scaleFactor = reconstruction->getCurrentPointCloud()->computeScaleFactor(referenceCloud);
		scale[0] = scaleFactor(0) + vel;
		scale[1] = scaleFactor(1) + vel;
		scale[2] = scaleFactor(2) + 2 * vel;

		//convert the reference from global to current coordinates
		Matrix3frm inverseRotation = reconstruction->getCurrentRotation().inverse();
		reconstruction->getCurrentPointCloud()->convertFromGlobalToCurrent(referenceCloud, inverseRotation, reconstruction->getCurrentTranslation());
				
		//position the medical volume based to the reference's centroid
		float centroid[3];
		for(int axis = 0; axis < 3; axis++) centroid[axis] = 0;
		for(int point = 0; point < referenceCloud->points.size(); point++) {
			centroid[0] += referenceCloud->points[point].x;
			centroid[1] += referenceCloud->points[point].y;
			centroid[2] += referenceCloud->points[point].z;
		}
		for(int axis = 0; axis < 3; axis++) centroid[axis] /= referenceCloud->points.size();

		translationVector[0] += centroid[0] - virtualCentroid[0];
		translationVector[1] += centroid[1] - virtualCentroid[1];
		translationVector[2] += centroid[2] - virtualCentroid[2];

		if(isHeadPoseEstimationEnabled) {
			sourceDepthData.resize(kinect->getImageWidth() * kinect->getImageHeight());
			kinect->getDepthImage()->fillDepthImageRaw(kinect->getImageWidth(), kinect->getImageHeight(), &sourceDepthData[0]);
			headPoseEstimationMediator->getHeadPoseEstimator()->run(&sourceDepthData[0]);
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
			std::getline(file, line);
			rotationAngles[0] = atoi(line.c_str());
			std::getline(file, line);
			rotationAngles[1] = atoi(line.c_str());
			std::getline(file, line);
			rotationAngles[2] = atoi(line.c_str());
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

void loadARDepthDataBasedOnDepthMaps() 
{

	glPixelTransferf(GL_DEPTH_SCALE, 1.0/reconstruction->getThreshold());
	for(int p = 0; p < 640 * 480; p++)
		depthData[p] = (float)imageCollection->getDepthMap().data[p];///(float)reconstruction->getThreshold();
	myGLImageViewer->loadDepthComponentTexture(depthData, texVBO, REAL_DEPTH_FROM_DEPTHBUFFER_BO, windowWidth, windowHeight);
	//glBindTexture(GL_TEXTURE_2D, texVBO[REAL_DEPTH_FROM_DEPTHBUFFER_BO]);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth/2, windowHeight/2, GL_DEPTH_COMPONENT, GL_FLOAT, depthData);
	reconstruction->getGlobalPreviousPointCloud()->getHostPointCloud(pointCloud);
	
	Matrix3frm rotInverse = reconstruction->getCurrentRotation().inverse();
	for(int point = 0; point < (640 * 480); point++) {
		if(pointCloud[point * 3 + 2] > 0 && pointCloud[point * 3 + 2] == pointCloud[point * 3 + 2]) {

			pointCloud[point * 3 + 0] -= reconstruction->getCurrentTranslation()[0];
            pointCloud[point * 3 + 1] -= reconstruction->getCurrentTranslation()[1];
            pointCloud[point * 3 + 2] -= reconstruction->getCurrentTranslation()[2];
			pointCloud[point * 3 + 2] = rotInverse(2, 0) * pointCloud[point * 3 + 0] + rotInverse(2, 1) * pointCloud[point * 3 + 1] +
                rotInverse(2, 2) * pointCloud[point * 3 + 2];

        }
    }
	
	for(int p = 0; p < 640 * 480; p++)
		depthData[p] = pointCloud[p * 3 + 2];///(float)reconstruction->getThreshold();
	
	myGLImageViewer->loadDepthComponentTexture(depthData, texVBO, VIRTUAL_DEPTH_BO, windowWidth, windowHeight);
	glPixelTransferf(GL_DEPTH_SCALE, 1);
	
	if(subtractionMaskOn) {
	
		IplImage *faceImage = cvCreateImage(cvSize(kinect->getImageWidth(), kinect->getImageHeight()), IPL_DEPTH_8U, 3);
		cvSetZero(faceImage);
		
		for(int pixel = 0; pixel < faceImage->width * faceImage->height; pixel++)
			if(depthData[pixel] > 0 && depthData[pixel] == depthData[pixel]) {
				faceImage->imageData[pixel * 3 + 0] = 255;
				faceImage->imageData[pixel * 3 + 1] = 255;
				faceImage->imageData[pixel * 3 + 2] = 255;
			}
	
		cvDilate(faceImage, faceImage, 0, 2);
		myGLImageViewer->loadRGBTexture((unsigned char*)faceImage->imageData, texVBO, FACE_MAP_FBO, faceImage->width, faceImage->height);
		cvReleaseImage(&faceImage);

	}

}

void computeVolumeContours()
{

	glViewport(0, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    
	
	myGLImageViewer->setProgram(shaderProg[7]);
	myGLImageViewer->drawRGBTextureOnShader(texVBO, VIRTUAL_RGB_BO, windowWidth, windowHeight);
	myGLImageViewer->loadFrameBufferTexture(0, windowHeight/2, windowWidth/2, windowHeight/2);
	
	glScissor(0, windowHeight/2, windowWidth/2, windowHeight/2);
	glEnable(GL_SCISSOR_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_SCISSOR_TEST);
	
	unsigned char *volumeImage = myGLImageViewer->getFrameBuffer();
	
	IplImage *image = cvCreateImage(cvSize(windowWidth/2, windowHeight/2), IPL_DEPTH_8U, 3);
	IplImage *grayImage = cvCreateImage(cvSize(windowWidth/2, windowHeight/2), IPL_DEPTH_8U, 1);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;
	
	image->imageData = (char*)volumeImage;
	cvCvtColor(image, grayImage, CV_BGR2GRAY);
	cvThreshold(grayImage, grayImage, 0, 255, CV_THRESH_OTSU);
	cvFindContours(grayImage, storage, &contours);
	cvDrawContours(grayImage, contours, cvScalarAll(255), cvScalarAll(0), 100);

	double K[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };  
  
	float t=0;  
	for(int i = 0; i < (3 * 3); ++i)  
		 t = t + K[i];  
	for(int i = 0; i < (3 * 3); ++i)  
		 K[i] = K[i] / t;  
  
	CvMat kernel = cvMat(3, 3, CV_64FC1, K); 
	for(int x = 0; x < 5; x++)
		cvFilter2D(grayImage, grayImage, &kernel);

	for(int pixel = 0; pixel < (640 * 480); pixel++)
		for(int ch = 0; ch < 3; ch++)
			image->imageData[pixel * 3 + ch] = grayImage->imageData[pixel];

	myGLImageViewer->loadRGBTexture((unsigned char*)image->imageData, texVBO, CONTOURS_FBO, image->width, image->height);

	cvReleaseImage(&image);
	cvReleaseImage(&grayImage);
	cvClearMemStorage(storage);

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

void displayContoursData()
{

	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    
	
	myGLImageViewer->setProgram(shaderProg[7]);
	myGLImageViewer->drawRGBTextureOnShader(texVBO, CONTOURS_FBO, windowWidth, windowHeight);

}

void displayCurvatureData()
{
	glViewport(0, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    

	reconstruction->getGlobalPreviousPointCloud()->getHostCurvature(curvature);
	for(int i = 0; i < kinect->getImageWidth() * kinect->getImageHeight(); i++)
		curv[i] = curvature[i] * 1300;
	myGLImageViewer->loadDepthTexture(curv, texVBO, CURVATURE_MAP_FBO, reconstruction->getThreshold(), kinect->getImageWidth(), 
		kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, CURVATURE_MAP_FBO, windowWidth, windowHeight);
	
}

void displayDepthData()
{
	
	glViewport(0, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    
	
	myGLImageViewer->loadDepthTexture((unsigned short*)imageCollection->getDepthMap().data, texVBO, REAL_DEPTH_FROM_DEPTHMAP_BO, 
		reconstruction->getThreshold(), kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, REAL_DEPTH_FROM_DEPTHMAP_BO, windowWidth, windowHeight);

}

void displayRGBData()
{
	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 

	myGLImageViewer->loadRGBTexture((const unsigned char*)imageCollection->getRGBMap().data, texVBO, REAL_RGB_BO, kinect->getImageWidth(), 
		kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, REAL_RGB_BO, windowWidth, windowHeight);

}

void displayRaycastedData()
{

	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    

	myGLImageViewer->loadRGBTexture(imageCollection->getRaycastImage(reconstruction->getVolumeSize(), 
		reconstruction->getGlobalPreviousPointCloud()), texVBO, RAYCAST_BO, kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, RAYCAST_BO, windowWidth, windowHeight);

}

void displayCloud(bool globalCoordinates = true)
{
	
	if(globalCoordinates) {
		reconstruction->getGlobalPreviousPointCloud()->getHostPointCloud(pointCloud);
		reconstruction->getGlobalPreviousPointCloud()->getHostNormalVector(normalVector, 1);
	} else {
		reconstruction->getCurrentPointCloud()->getHostPointCloud(pointCloud);
		reconstruction->getCurrentPointCloud()->getHostNormalVector(normalVector, -1);
	}
	
	myGLCloudViewer->loadIndices(indices, pointCloud);
	myGLCloudViewer->loadVBOs(meshVBO, indices, pointCloud, normalVector);
	
	//glViewport(0, 0, windowWidth/2, windowHeight/2);
	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	
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

void displayQuadForVolumeRendering(bool front)
{

	glEnable(GL_DEPTH_TEST);
	
    glViewport(0, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
    glLoadIdentity(); 
	myGLCloudViewer->configureQuadAmbient(reconstruction->getThreshold());

	for(int axis = 0; axis < 3; axis++) {
		modelViewParams.translationVector[axis] = translationVector[axis];
		modelViewParams.rotationAngles[axis] = rotationAngles[axis];
	}
	
	modelViewParams.gTrans = reconstruction->getCurrentTranslation();
	modelViewParams.gRot = reconstruction->getCurrentRotation();
	modelViewParams.initialTranslation = reconstruction->getInitialTranslation();
	modelViewParams.rotationIndices[0] = vrparams.rotationX;
	modelViewParams.rotationIndices[1] = vrparams.rotationY;
	modelViewParams.rotationIndices[2] = vrparams.rotationZ;
	modelViewParams.useTextureRotation = true;
	modelViewParams.useHeadPoseRotation = false;
	
	if(isHeadPoseEstimationEnabled) {
		Eigen::Matrix3f headRotationMatrix;
		float mult = 0.0174532925f;
		modelViewParams.useHeadPoseRotation = headPoseEstimationMediator->getHeadPoseEstimator()->hadSuccess();
		if(headPoseEstimationMediator->getHeadPoseEstimator()->hadSuccess()) {
			headPoseEstimationMediator->getHeadPoseEstimator()->eulerToRotationMatrix(headRotationMatrix, 
				mult * -headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles()(0), 
				mult * headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles()(2), 
				mult * headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles()(1));
			Eigen::Vector3f headCenterRotated = headRotationMatrix * headPoseEstimationMediator->getHeadPoseEstimator()->getHeadCenter(); 
			modelViewParams.headCenter = headPoseEstimationMediator->getHeadPoseEstimator()->getHeadCenter();
			modelViewParams.headEulerAngles = headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles();
			modelViewParams.headCenterRotated = headCenterRotated;
		}
	}
	else modelViewParams.useHeadPoseRotation = false;
	
	myGLCloudViewer->updateModelViewMatrix(modelViewParams);
	glScalef(scale[0], scale[1], scale[2]);
	
	if(!front) {
		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
	} else {
		glDisable(GL_CULL_FACE);
	}
	myGLCloudViewer->drawQuad(quadVBO);
	glPopMatrix();

}

void displayMedicalVolume()
{

	glViewport(0, 0, windowWidth/2, windowHeight/2);
	
	glEnable(GL_TEXTURE_3D);
	myGLCloudViewer->configureARAmbientWithBlending(reconstruction->getThreshold());
	myGLCloudViewer->setAmbientIntensity(0.1);
	myGLCloudViewer->setDiffuseIntensity(0.2);
	myGLCloudViewer->setSpecularIntensity(0.5);
	myGLCloudViewer->configureLight();

	myGLImageViewer->setProgram(shaderProg[VRShaderID]);
	for(int axis = 0; axis < 3; axis++) {
		modelViewParams.translationVector[axis] = translationVector[axis];
		modelViewParams.rotationAngles[axis] = rotationAngles[axis];
	}
	
	modelViewParams.gTrans = reconstruction->getCurrentTranslation();
	modelViewParams.gRot = reconstruction->getCurrentRotation();
	modelViewParams.initialTranslation = reconstruction->getInitialTranslation();
	modelViewParams.rotationIndices[0] = vrparams.rotationX;
	modelViewParams.rotationIndices[1] = vrparams.rotationY;
	modelViewParams.rotationIndices[2] = vrparams.rotationZ;
	modelViewParams.useTextureRotation = true;
	modelViewParams.useHeadPoseRotation = false;

	if(isHeadPoseEstimationEnabled) {
		Eigen::Matrix3f headRotationMatrix;
		float mult = 0.0174532925f;
		modelViewParams.useHeadPoseRotation = headPoseEstimationMediator->getHeadPoseEstimator()->hadSuccess();
		if(headPoseEstimationMediator->getHeadPoseEstimator()->hadSuccess()) {
			headPoseEstimationMediator->getHeadPoseEstimator()->eulerToRotationMatrix(headRotationMatrix, 
				mult * -headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles()(0), 
				mult * headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles()(2), 
				mult * headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles()(1));
			Eigen::Vector3f headCenterRotated = headRotationMatrix * headPoseEstimationMediator->getHeadPoseEstimator()->getHeadCenter(); 
			modelViewParams.headCenter = headPoseEstimationMediator->getHeadPoseEstimator()->getHeadCenter();
			modelViewParams.headEulerAngles = headPoseEstimationMediator->getHeadPoseEstimator()->getEulerAngles();
			modelViewParams.headCenterRotated = headCenterRotated;
		}
	}
	else modelViewParams.useHeadPoseRotation = false;
	
	myGLCloudViewer->updateModelViewMatrix(modelViewParams);
	glScalef(scale[0], scale[1], scale[2]);
	
	myGLImageViewer->draw3DTexture(texVBO, AR_FROM_VOLUME_RENDERING_BO, MIN_MAX_OCTREE_BO, vrparams, FRONT_QUAD_RGB_FBO, BACK_QUAD_RGB_FBO, 
		myGLCloudViewer->getEyePosition(), windowWidth/2, windowHeight/2, myGLCloudViewer, quadVBO, TRANSFER_FUNCTION_BO, NOISE_BO);
	glPopMatrix();

	glDisable(GL_BLEND);
	glDisable(GL_ALPHA_TEST);	
	glEnable(GL_DEPTH_TEST);

}

void computeVolumeSubtractionMask()
{

	int old = VRShaderID;
	VRShaderID = 2;

	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayMedicalVolume();	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	VRShaderID = old;

	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();    
	
	myGLImageViewer->setProgram(shaderProg[7]);
	myGLImageViewer->drawRGBTextureOnShader(texVBO, VIRTUAL_RGB_BO, windowWidth, windowHeight);
	myGLImageViewer->loadFrameBufferTexture(windowWidth/2, 0, windowWidth/2, windowHeight/2);

	glScissor(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glEnable(GL_SCISSOR_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_SCISSOR_TEST);

	unsigned char *volumeImage = myGLImageViewer->getFrameBuffer();

	IplImage *image = cvCreateImage(cvSize(windowWidth/2, windowHeight/2), IPL_DEPTH_8U, 3);
	IplImage *grayImage = cvCreateImage(cvSize(windowWidth/2, windowHeight/2), IPL_DEPTH_8U, 1);

	image->imageData = (char*)volumeImage;
	cvCvtColor(image, grayImage, CV_BGR2GRAY);
	cvThreshold(grayImage, grayImage, 0, 255, CV_THRESH_OTSU);

	for(int pixel = 0; pixel < (640 * 480); pixel++)
		for(int ch = 0; ch < 3; ch++)
			image->imageData[pixel * 3 + ch] = grayImage->imageData[pixel];

	myGLImageViewer->loadRGBTexture((unsigned char*)image->imageData, texVBO, SUBTRACTION_MASK_FBO, image->width, image->height);

	cvReleaseImage(&image);
	cvReleaseImage(&grayImage);
	
}

void displayARDataFromVolume()
{
	
	loadARDepthDataBasedOnDepthMaps();

	//Second Viewport: Virtual Object
	glViewport(windowWidth/2, windowHeight/2, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 

	myGLImageViewer->loadRGBTexture(imageCollection->getRaycastImage(reconstruction->getVolumeSize(), reconstruction->getGlobalPreviousPointCloud()), texVBO, RAYCAST_BO, 
		kinect->getImageWidth(), kinect->getImageHeight());
	myGLImageViewer->drawRGBTexture(texVBO, RAYCAST_BO, windowWidth, windowHeight);

	//Fourth Viewport: Virtual + Real Object
	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 

	myGLImageViewer->loadRGBTexture((const unsigned char*)imageCollection->getRGBMap().data, texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());

	occlusionParams.texVBO = texVBO;
	occlusionParams.realRGBIndex = REAL_RGB_BO;
	occlusionParams.realDepthIndex = REAL_DEPTH_FROM_DEPTHBUFFER_BO;
	occlusionParams.virtualRGBIndex = VIRTUAL_RGB_BO;
	occlusionParams.virtualDepthIndex = VIRTUAL_DEPTH_BO;
	occlusionParams.windowWidth = windowWidth;
	occlusionParams.windowHeight = windowHeight;
	occlusionParams.ARPolygonal = false;
	occlusionParams.ARFromKinectFusionVolume = true;
	occlusionParams.ARFromVolumeRendering = false;
	occlusionParams.alphaBlending = true;
	occlusionParams.ghostViewBasedOnCurvatureMap = false;
	occlusionParams.ghostViewBasedOnDistanceFalloff = false;
	occlusionParams.ghostViewBasedOnClipping = false;
	
	myGLImageViewer->setProgram(shaderProg[1]);
	myGLImageViewer->drawARTextureWithOcclusion(occlusionParams);
	
}

void displayARDataFromOBJFile()
{
	
	glViewport(0, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity(); 
	myGLCloudViewer->configureAmbient(reconstruction->getThreshold());
	
	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	for(int axis = 0; axis < 3; axis++) {
		modelViewParams.translationVector[axis] = translationVector[axis];
		modelViewParams.rotationAngles[axis] = rotationAngles[axis];
	}
	
	modelViewParams.gTrans = reconstruction->getCurrentTranslation();
	modelViewParams.gRot = reconstruction->getCurrentRotation();
	modelViewParams.initialTranslation = reconstruction->getInitialTranslation();
	modelViewParams.useTextureRotation = false;
	modelViewParams.useHeadPoseRotation = false;
	myGLCloudViewer->drawOBJ(modelViewParams);
	
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
		myGLCloudViewer->drawOBJ(modelViewParams);

	}

	myGLImageViewer->loadRGBTexture((const unsigned char*)imageCollection->getRGBMap().data, texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());
	
	//Third Viewport: Virtual + Real Object
	glViewport(0, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();   
	
	occlusionParams.texVBO = texVBO;
	occlusionParams.realRGBIndex = REAL_RGB_BO;
	occlusionParams.realDepthIndex = REAL_DEPTH_FROM_DEPTHBUFFER_BO;
	occlusionParams.virtualRGBIndex = VIRTUAL_RGB_BO;
	occlusionParams.virtualDepthIndex = VIRTUAL_DEPTH_BO;
	occlusionParams.windowWidth = windowWidth;
	occlusionParams.windowHeight = windowHeight;
	occlusionParams.ARPolygonal = true;
	occlusionParams.ARFromKinectFusionVolume = false;
	occlusionParams.ARFromVolumeRendering = false;
	occlusionParams.alphaBlending = true;
	occlusionParams.ghostViewBasedOnCurvatureMap = false;
	occlusionParams.ghostViewBasedOnDistanceFalloff = false;
	occlusionParams.ghostViewBasedOnClipping = false;
	
	myGLImageViewer->setProgram(shaderProg[1]);
	myGLImageViewer->drawARTextureWithOcclusion(occlusionParams);
	
}

void displayARDataFromVolumeRendering()
{
	
	glEnable(GL_DEPTH_TEST);
	
	if(curvatureBlendingOn) {
		reconstruction->getGlobalPreviousPointCloud()->getHostCurvature(curvature);
		myGLImageViewer->loadDepthComponentTexture(curvature, texVBO, CURVATURE_MAP_FBO, windowWidth, windowHeight);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, backQuadFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayQuadForVolumeRendering(false);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, frontQuadFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayQuadForVolumeRendering(true);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	if(subtractionMaskOn)
		computeVolumeSubtractionMask();

	glBindFramebuffer(GL_FRAMEBUFFER, virtualFrameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	displayMedicalVolume();	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	loadARDepthDataBasedOnDepthMaps();	

	if(clippingBlendingOn)
		computeVolumeContours();

	//Fourth Viewport: Virtual + Real Object
	glViewport(0, 0, windowWidth/2, windowHeight/2);
	glMatrixMode(GL_PROJECTION);          
	glLoadIdentity();

	myGLImageViewer->loadRGBTexture((const unsigned char*)imageCollection->getRGBMap().data, texVBO, REAL_RGB_BO, kinect->getImageWidth(), kinect->getImageHeight());

	occlusionParams.texVBO = texVBO;
	occlusionParams.realRGBIndex = REAL_RGB_BO;
	occlusionParams.realDepthIndex = REAL_DEPTH_FROM_DEPTHBUFFER_BO;
	occlusionParams.virtualRGBIndex = VIRTUAL_RGB_BO;
	occlusionParams.virtualDepthIndex = VIRTUAL_DEPTH_BO;
	occlusionParams.curvatureMapIndex = CURVATURE_MAP_FBO;
	occlusionParams.contoursMapIndex = CONTOURS_FBO;
	occlusionParams.backgroundMapIndex = BACKGROUND_SCENE_FBO;
	occlusionParams.subtractionMapIndex = SUBTRACTION_MASK_FBO;
	occlusionParams.faceMapIndex = FACE_MAP_FBO;
	occlusionParams.windowWidth = windowWidth;
	occlusionParams.windowHeight = windowHeight;
	occlusionParams.ARPolygonal = false;
	occlusionParams.ARFromKinectFusionVolume = false;
	occlusionParams.ARFromVolumeRendering = true;
	occlusionParams.alphaBlending = alphaBlendingOn;
	occlusionParams.ghostViewBasedOnCurvatureMap = curvatureBlendingOn;
	occlusionParams.ghostViewBasedOnDistanceFalloff = distanceFalloffBlendingOn;
	occlusionParams.ghostViewBasedOnClipping = clippingBlendingOn;
	occlusionParams.ghostViewBasedOnSubtractionMask = subtractionMaskOn;
	occlusionParams.curvatureWeight = curvatureWeight;
	occlusionParams.distanceFalloffWeight = distanceFalloffWeight;
	occlusionParams.clippingWeight = clippingWeight;
	occlusionParams.focusPoint[0] = focusPoint[0];
	occlusionParams.focusPoint[1] = focusPoint[1];
	occlusionParams.focusRadius = focusRadius;

	myGLImageViewer->setProgram(shaderProg[1]);
	myGLImageViewer->drawARTextureWithOcclusion(occlusionParams);
	
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
		if(showCurvatureMap)
			displayCurvatureData();
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
		if(showContoursMap)
			displayContoursData();
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

		if(!ARConfiguration)
			imageCollection->load(kinect->getRGBImage(), kinect->getDepthImage());

		if(hasFaceDetection) {
			if(faceDetected) {
				reconstruction->stopTracking(false);
				faceDetector->segmentFace(imageCollection);
			} else {
				reconstruction->reset();
				reconstruction->stopTracking(true);
				faceDetected = faceDetector->run(imageCollection);
				if(faceDetected)
					faceDetector->segmentFace(imageCollection);
			}
			//to check if the reconstruction was reseted
			if(reconstruction->getGlobalTime() > 0) preGlobalTimeGreaterThanZero = true;
			else preGlobalTimeGreaterThanZero = false;
		}

		if(!ARConfiguration) {
			imageCollection->applyBilateralFilter();
			imageCollection->applyDepthTruncation(reconstruction->getThreshold());
			imageCollection->applyPyrDown();
			imageCollection->convertToPointCloud(reconstruction->getCurrentPointCloud());
			imageCollection->applyDepthTruncation(imageCollection->getDepthDevice(), reconstruction->getThreshold());
			pcl::device::sync ();
			reconstruction->run(imageCollection); 
		}

		//if the reconstruction was reseted, the face is no more detected
		if(hasFaceDetection)
			if(reconstruction->getGlobalTime() == 0 && preGlobalTimeGreaterThanZero)
				faceDetected = false;
		if(!ARConfiguration && !AR && integrateColors)
			coloredReconstructionMediator->updateColorVolume(imageCollection->getRgbDevice(), reconstruction);
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
			headPoseEstimationMediator->stopTracking(true, (unsigned short*)imageCollection->getDepthMap().data, reconstruction);
		break;
	case (int)'c' : case (int)'C':
		std::cout << "Continue..." << std::endl;
		reconstruction->stopTracking(false);
		if(isHeadPoseEstimationEnabled)
			headPoseEstimationMediator->stopTracking(false, (unsigned short*)imageCollection->getDepthMap().data, reconstruction);
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
			myGLCloudViewer->setEyePosition(1, 0, 120);
			focusRadius = 500;

			if(ARVolumetric)
				positionVirtualObject(320, 320);
			
			std::cout << "Enabling AR" << std::endl;
			std::cout << "AR Enabled: Click the window to position the object (if necessary, use the scale factor (s)" << std::endl;
			
			
		} else if(ARConfiguration) {

			ARConfiguration = false;
			myGLCloudViewer->setEyePosition(1, 0, 170);
			w3 = 170;
			std::cout << "AR Enabled: Configuration finished" << std::endl;
			
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
	case (int)'d':
		focusRadius = 50;
		distanceFalloffWeight = 2.0;
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
		if(curvatureWeightOn)
			curvatureWeight += 0.5;
		if(focusRadiusOn)
			focusRadius += 5;
		if(distanceFallOffWeightOn)
			distanceFalloffWeight += 0.1;
		if(clippingWeightOn)
			clippingWeight += 0.5;
		if(clippingPlaneUpYOn) {
			vrparams.clippingPlaneUpY += 0.05;
			if(vrparams.clippingPlaneUpY > 1) vrparams.clippingPlaneUpY = 1;
		}
		if(clippingPlaneDownYOn) {
			vrparams.clippingPlaneDownY += 0.05;
			if(vrparams.clippingPlaneDownY > 1) vrparams.clippingPlaneDownY = 1;
		}
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
		if(curvatureWeightOn) {
			curvatureWeight -= 0.5;
			if(curvatureWeight < 0) curvatureWeight = 0;
		}
		if(focusRadiusOn) {
			focusRadius -= 5;
			if(focusRadius < 0) focusRadius = 0;
		}
		if(distanceFallOffWeightOn) {
			distanceFalloffWeight -= 0.1;
			if(distanceFalloffWeight < 0) distanceFalloffWeight = 0;
		}
		if(clippingWeightOn) {
			clippingWeight -= 0.5;
			if(clippingWeight < 0) clippingWeight = 0;
		}
		if(clippingPlaneUpYOn) {
			vrparams.clippingPlaneUpY -= 0.05;
			if(vrparams.clippingPlaneUpY < 0) vrparams.clippingPlaneUpY = 0;
		}
		if(clippingPlaneDownYOn) {
			vrparams.clippingPlaneDownY -= 0.05;
			if(vrparams.clippingPlaneDownY < 0) vrparams.clippingPlaneDownY = 0;
		}
		break;
	case GLUT_KEY_LEFT:
		if(translationOn)
			translationVector[0] -= vel;
		if(rotationOn)
			rotationAngles[0] -= vel;
		if(scaleOn)
			setScale(0, false);
		if(clippingPlaneLeftXOn) {
			vrparams.clippingPlaneLeftX += 0.05f;
			if(vrparams.clippingPlaneLeftX > 1) vrparams.clippingPlaneLeftX = 1;
		}
		if(clippingPlaneRightXOn) {
			vrparams.clippingPlaneRightX += 0.05f;
			if(vrparams.clippingPlaneRightX > 1) vrparams.clippingPlaneRightX = 1;
		}
		break;
	case GLUT_KEY_RIGHT:
		if(translationOn)
			translationVector[0] += vel;
		if(rotationOn)
			rotationAngles[0] += vel;
		if(scaleOn)
			setScale(0, true);
		if(clippingPlaneLeftXOn) {
			vrparams.clippingPlaneLeftX -= 0.05f;
			if(vrparams.clippingPlaneLeftX < 0) vrparams.clippingPlaneLeftX = 0;
		} if(clippingPlaneRightXOn) {
			vrparams.clippingPlaneRightX -= 0.05f;
			if(vrparams.clippingPlaneRightX < 0) vrparams.clippingPlaneRightX = 0;
		}
		break;
	case GLUT_KEY_PAGE_UP:
		if(translationOn)
			translationVector[2] += vel;
		if(rotationOn)
			rotationAngles[2] += vel;
		if(scaleOn)
			setScale(2, true);
		if(clippingPlaneFrontZOn) {
			vrparams.clippingPlaneFrontZ += 0.05f;
			if(vrparams.clippingPlaneFrontZ > 1) vrparams.clippingPlaneFrontZ = 1;
		}
		if(clippingPlaneBackZOn) {
			vrparams.clippingPlaneBackZ += 0.05f;
			if(vrparams.clippingPlaneBackZ > 1) vrparams.clippingPlaneBackZ = 1;
		}
		break;
	case GLUT_KEY_PAGE_DOWN:
		if(translationOn)
			translationVector[2] -= vel;
		if(rotationOn)
			rotationAngles[2] -= vel;
		if(scaleOn)
			setScale(2, false);
		if(clippingPlaneFrontZOn) {
			vrparams.clippingPlaneFrontZ -= 0.05f;
			if(vrparams.clippingPlaneFrontZ < 0) vrparams.clippingPlaneFrontZ = 0;
		}
		if(clippingPlaneBackZOn) {
			vrparams.clippingPlaneBackZ -= 0.05f;
			if(vrparams.clippingPlaneBackZ < 0) vrparams.clippingPlaneBackZ = 0;
		}
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) 
{
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_UP) {
			focusPoint[0] = x;
			focusPoint[1] = windowHeight/2 - (y - windowHeight/2);
			if(ARConfiguration)
				positionVirtualObject(x, y);
		}

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
	case 9:
		vrparams.clippingOcclusion = !vrparams.clippingOcclusion;
		break;
	}
}

void thresholdMenu(int id)
{

	translationOn = false;
	rotationOn = false;
	scaleOn = false;
	earlyRayTerminationOn = false;
	stepSizeModificationOn = false;
	isoSurfaceThresholdModificationOn = false;
	ksOn = false;
	ktOn = false;
	curvatureWeightOn = false;
	focusRadiusOn = false;
	distanceFallOffWeightOn = false;
	clippingPlaneLeftXOn = false;
	clippingPlaneRightXOn = false;
	clippingPlaneUpYOn = false;
	clippingPlaneDownYOn = false;
	clippingPlaneFrontZOn = false;
	clippingPlaneBackZOn = false;
	clippingWeightOn = false;
	
	switch(id)
	{
	case 0:
		earlyRayTerminationOn = true;
		break;
	case 1:
		stepSizeModificationOn = true;
		break;
	case 2:
		isoSurfaceThresholdModificationOn = true; 
		break;
	case 3:
		ksOn = true;
		break;
	case 4:
		ktOn = true;
		break;
	case 5:
		curvatureWeightOn = true;
		break;
	case 6:
		distanceFallOffWeightOn = true;
		break;
	case 7:
		focusRadiusOn = true;
		break;
	case 8:
		clippingPlaneLeftXOn = true;
		break;
	case 9:
		clippingPlaneRightXOn = true;
		break;
	case 10:
		clippingPlaneUpYOn = true;
		break;
	case 11:
		clippingPlaneDownYOn = true;
		break;
	case 12:
		clippingPlaneFrontZOn = true;
		break;
	case 13:
		clippingPlaneBackZOn = true;
		break;
	case 14:
		clippingWeightOn = true;
		break;
	}
}

void transformationMenu(int id)
{

	translationOn = false;
	rotationOn = false;
	scaleOn = false;
	earlyRayTerminationOn = false;
	stepSizeModificationOn = false;
	isoSurfaceThresholdModificationOn = false;
	ksOn = false;
	ktOn = false;
	curvatureWeightOn = false;
	focusRadiusOn = false;
	distanceFallOffWeightOn = false;
	clippingPlaneLeftXOn = false;
	clippingPlaneRightXOn = false;
	clippingPlaneUpYOn = false;
	clippingPlaneDownYOn = false;
	clippingPlaneFrontZOn = false;
	clippingPlaneBackZOn = false;
	clippingWeightOn = false;

	switch(id)
	{
	case 0:
		translationOn = true;
		break;
	case 1:
		rotationOn = true;
		break;
	case 2:
		scaleOn = true;
		break;
	}
}

void blendingMenu(int id) 
{
	
	switch(id)
	{
	case 0:
		alphaBlendingOn = true;
		curvatureBlendingOn = false;
		distanceFalloffBlendingOn = false;
		clippingBlendingOn = false;
		subtractionMaskOn = false;
		break;
	case 1:
		alphaBlendingOn = false;
		curvatureBlendingOn = !curvatureBlendingOn;
		break;
	case 2:
		alphaBlendingOn = false;
		distanceFalloffBlendingOn = !distanceFalloffBlendingOn;
		break;
	case 3:
		alphaBlendingOn = false;
		clippingBlendingOn = !clippingBlendingOn;
		break;
	case 4:
		alphaBlendingOn = false;
		subtractionMaskOn = !subtractionMaskOn;
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
	case 1:
		showCurvatureMap = !showCurvatureMap;
		if(showCurvatureMap) showDepthMap = false;
		else showDepthMap = true;
		break;
	case 2:
		showContoursMap = !showContoursMap;
		break;
	case 3:
		kinect->getRGBImage()->fillRGB(kinect->getImageWidth(), kinect->getImageHeight(), backgroundScene);
		myGLImageViewer->loadRGBTexture(backgroundScene, texVBO, BACKGROUND_SCENE_FBO, kinect->getImageWidth(), kinect->getImageHeight());
		break;
	}
}

void createMenu()
{

	GLint volumeRenderingMenuID, thresholdMenuID, transformationMenuID, blendingMenuID, otherFunctionsMenuID;

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
		glutAddMenuEntry("Occlusion based on clipping", 9);
	
	thresholdMenuID = glutCreateMenu(thresholdMenu);
		glutAddMenuEntry("Change Early Ray Termination", 0);
		glutAddMenuEntry("Change Step Size (Raycasting)", 1);
		glutAddMenuEntry("Change Iso Surface", 2);
		glutAddMenuEntry("Change Ks (Context-Preserving VR)", 3);
		glutAddMenuEntry("Change Kt (Context-Preserving VR)", 4);
		glutAddMenuEntry("Change Curvature Weight (Ghosted Views)", 5);
		glutAddMenuEntry("Change Distance Fall Off Weight (Ghosted Views)", 6);
		glutAddMenuEntry("Change Focus Radius (Ghosted Views)", 7);
		glutAddMenuEntry("Change Clipping Plane Right X (Ghosted Views)", 8); //inverted (also in specialKeyboard)
		glutAddMenuEntry("Change Clipping Plane Left X (Ghosted Views)", 9); //inverted (also in specialKeyboard)
		glutAddMenuEntry("Change Clipping Plane Up Y (Ghosted Views)", 10);
		glutAddMenuEntry("Change Clipping Plane Down Y (Ghosted Views)", 11);
		glutAddMenuEntry("Change Clipping Plane Front Z (Ghosted Views)", 12);
		glutAddMenuEntry("Change Clipping Plane Back Z (Ghosted Views)", 13);
		glutAddMenuEntry("Change Clipping Distance Fall Off Weight (Ghosted Views)", 14);

	transformationMenuID = glutCreateMenu(transformationMenu);
		glutAddMenuEntry("Change Translation", 0);
		glutAddMenuEntry("Change Rotation", 1);
		glutAddMenuEntry("Change Scale", 2);

	blendingMenuID = glutCreateMenu(blendingMenu);
		glutAddMenuEntry("Simple Blending [On/Off]", 0);
		glutAddMenuEntry("Curvature Blending [On/Off]", 1);
		glutAddMenuEntry("Distance Falloff Blending [On/Off]", 2);
		glutAddMenuEntry("Clipping-based Blending [On/Off]", 3);
		glutAddMenuEntry("Subtraction Mask Blending [On/Off]", 4);

	otherFunctionsMenuID = glutCreateMenu(otherFunctionsMenu);
		glutAddMenuEntry("Save Model", 0);
		glutAddMenuEntry("Show/Hide Curvature Map", 1);
		glutAddMenuEntry("Show/Hide Contours Map", 2);
		glutAddMenuEntry("Save the Background Scene", 3);

	glutCreateMenu(mainMenu);
		glutAddSubMenu("Transformation", transformationMenuID);
		glutAddSubMenu("Volume Rendering", volumeRenderingMenuID);
		glutAddSubMenu("Threshold", thresholdMenuID);
		glutAddSubMenu("Blending Mode", blendingMenuID);
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
		glGenTextures(25, texVBO);
	if(meshVBO[0] == 0)
		glGenBuffers(4, meshVBO);
	if(quadVBO[0] == 0)
		glGenBuffers(4, quadVBO);
	if(virtualFrameBuffer == 0)
		glGenFramebuffers(1, &virtualFrameBuffer);
	if(realFrameBuffer == 0)
		glGenFramebuffers(1, &realFrameBuffer);
	if(frontQuadFrameBuffer == 0)
		glGenFramebuffers(1, &frontQuadFrameBuffer);
	if(backQuadFrameBuffer == 0)
		glGenFramebuffers(1, &backQuadFrameBuffer);

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

		//Organize the data (only used by Marching Cubes)
		medicalVolume->organizeData();

		//build a min-max octree to allow the use of empty-space skipping and adaptive sampling
		minMaxOctree = new MinMaxOctree(medicalVolume->getWidth(), medicalVolume->getHeight(), medicalVolume->getDepth());
		minMaxOctree->build(medicalVolume->getData(), medicalVolume->getWidth(), medicalVolume->getHeight(), medicalVolume->getDepth());
		
		//load both textures
		myGLImageViewer->load3DTexture(medicalVolume->getData(), texVBO, AR_FROM_VOLUME_RENDERING_BO, medicalVolume->getWidth(), medicalVolume->getHeight(), 
			medicalVolume->getDepth());
		myGLImageViewer->load3DTexture(minMaxOctree->getData(), texVBO, MIN_MAX_OCTREE_BO, minMaxOctree->getWidth(), minMaxOctree->getHeight(), minMaxOctree->getDepth());
		
		//compute a transfer function to map the scalar value to some color
		transferFunction = new TransferFunction();
		transferFunction->load();
		transferFunction->computePreIntegrationTable();
		myGLImageViewer->loadRGBATexture(transferFunction->getPreIntegrationTable(), texVBO, TRANSFER_FUNCTION_BO, 256, 256);
		
		//compute a noise texture to allow the use of stochastic jittering (i.e. random ray start)
		myGLImageViewer->load2DNoiseTexture(texVBO, NOISE_BO, 32, 32);
		
		myGLCloudViewer->loadVBOQuad(quadVBO, 1.0f/vrparams.scaleWidth, 1.0f/vrparams.scaleHeight, 1.0f/vrparams.scaleDepth);

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
		//Inverted because of the view
		vrparams.clippingPlane = true;
		vrparams.clippingOcclusion = false;
		vrparams.clippingPlaneLeftX = 0.0;
		vrparams.clippingPlaneRightX = 1.0;
		vrparams.clippingPlaneUpY = 1.0;
		vrparams.clippingPlaneDownY = 0.0;
		vrparams.clippingPlaneFrontZ = 0.0;
		vrparams.clippingPlaneBackZ = 1.0;
		
		createMenu();
	
	}

	myGLImageViewer->loadDepthComponentTexture(NULL, texVBO, VIRTUAL_DEPTH_BO, windowWidth, windowHeight);
	myGLImageViewer->loadDepthComponentTexture(NULL, texVBO, REAL_DEPTH_FROM_DEPTHBUFFER_BO, windowWidth, windowHeight);
	myGLImageViewer->loadDepthComponentTexture(NULL, texVBO, FRONT_QUAD_DEPTH_FBO, windowWidth, windowHeight);
	myGLImageViewer->loadDepthComponentTexture(NULL, texVBO, BACK_QUAD_DEPTH_FBO, windowWidth, windowHeight);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, VIRTUAL_RGB_BO, windowWidth/2, windowHeight/2);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, REAL_RGB_FROM_FBO, windowWidth/2, windowHeight/2);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, FRONT_QUAD_RGB_FBO, windowWidth/2, windowHeight/2);
	myGLImageViewer->loadRGBTexture(NULL, texVBO, BACK_QUAD_RGB_FBO, windowWidth/2, windowHeight/2);

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

	glBindFramebuffer(GL_FRAMEBUFFER, frontQuadFrameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texVBO[FRONT_QUAD_DEPTH_FBO], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texVBO[FRONT_QUAD_RGB_FBO], 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
		std::cout << "FBO OK" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, backQuadFrameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texVBO[BACK_QUAD_DEPTH_FBO], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texVBO[BACK_QUAD_RGB_FBO], 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
		std::cout << "FBO OK" << std::endl;

}

void releaseObjects() {

  delete kinect;
  delete imageCollection;
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
	//Initialize some objects
	reconstruction = new Reconstruction(volumeSize);
	kinect = new Kinect();
	imageCollection = new Image(640, 480);

	imageCollection->setDepthIntrinsics(kinect->getFocalLength(), kinect->getFocalLength());
	imageCollection->setTrancationDistance(volumeSize);

	reconstruction->setIntrinsics(imageCollection->getIntrinsics());
	reconstruction->setTrancationDistance(imageCollection->getTrancationDistance());

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
	initShader("Shaders/VRImage", 7);
	
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

  releaseObjects();
  return 0;

}
