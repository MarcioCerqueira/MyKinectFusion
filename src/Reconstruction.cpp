#include "Reconstruction.h"

#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

typedef pcl::ScopeTime ScopeTimeT;

Reconstruction::Reconstruction(Eigen::Vector3i& volumeSize) {

	hasImage_ = false;
	hasIncrement_ = true;
	hasErrorVisualization_ = false;
	hasTsdfVolumeVisualization_ = false;

	isOnlyTrackingOn_ = false;
	stopTracking_ = false;

	headPoseEstimationOk = false;

	image_ = new Image(640, 480);
	tsdfVolume_ = new TsdfVolume(volumeSize);

	currentPointCloud_ = new MyPointCloud(640, 480);
	globalPreviousPointCloud_ = new MyPointCloud(640, 480);
	auxPointCloud_ = new MyPointCloud(640, 480);

	float f = 525.f;
	
	image_->setDepthIntrinsics(f, f);
	image_->setTrancationDistance(tsdfVolume_->getVolumeSize());

	init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
	//init_tcam_ = Eigen::Vector3f::Zero ();
	init_tcam_ = tsdfVolume_->getVolumeSize() * 0.5f - Vector3f (0, 0, tsdfVolume_->getVolumeSize() (2) / 2 * 1.2f);
	
	rmats_.reserve (30000);
	tvecs_.reserve (30000);

	reset();

	previousDepthData = new unsigned short[640 * 480];

}

void Reconstruction::savePointCloud() {
	
	DeviceArray<pcl::PointXYZ> extractedCloudDevice;
	DeviceArray<PointXYZ> extracted =  tsdfVolume_->fetchCloud(extractedCloudDevice);

	pcl::PointCloud<pcl::PointXYZ>::Ptr hostCloud = pcl::PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
	extracted.download(hostCloud->points);
	hostCloud->width = (int)hostCloud->points.size ();
	hostCloud->height = 1;
	
	char fileToSave[1000];
	std::cout << "Write filename..." << std::endl;
	std::cin >> fileToSave;
	std::cout << fileToSave << std::endl;
	pcl::io::savePCDFile(fileToSave, *hostCloud); 
	std::cout << "Model saved..." << std::endl;

}

void Reconstruction::reset() {

	if(!isOnlyTrackingOn_) {
	
		rmats_.clear ();
		tvecs_.clear ();

		rmats_.push_back (init_Rcam_);
		tvecs_.push_back (init_tcam_);
		tsdfVolume_->reset();

		globalTime = 0;

		hasIncrement_ = false;

		std::cout << "Reset" << std::endl;
	
	}

}

void Reconstruction::run(boost::shared_ptr<openni_wrapper::Image>& rgbImage, boost::shared_ptr<openni_wrapper::DepthImage>& depthImage) {

	depthMap = image_->getDepthMap();
	depthDevice = image_->getDepthDevice();
	rgbDevice = image_->getRgbDevice();

	depthMap.cols = depthImage->getWidth();
	depthMap.rows = depthImage->getHeight();
	depthMap.step = depthMap.cols * depthMap.elemSize();
	
	sourceDepthData.resize(depthMap.cols * depthMap.rows);
	depthImage->fillDepthImageRaw(depthMap.cols, depthMap.rows, &sourceDepthData[0]);
	depthMap.data = &sourceDepthData[0];

	rgbMap.cols = rgbImage->getWidth();
	rgbMap.rows = rgbImage->getHeight();
	rgbMap.step = rgbMap.cols * rgbMap.elemSize();

	sourceRgbData.resize(rgbMap.cols * rgbMap.rows);
	rgbImage->fillRGB(rgbMap.cols, rgbMap.rows, (unsigned char*)&sourceRgbData[0]);
	rgbMap.data = &sourceRgbData[0];
	
	rgbDevice.upload(rgbMap.data, rgbMap.step, rgbMap.rows, rgbMap.cols);
	depthDevice.upload(depthMap.data, depthMap.step, depthMap.rows, depthMap.cols);
	
	//ScopeTimeT time ("total-frame");
	//{
		image_->setDepthDevice(depthDevice);
		image_->setRgbDevice(rgbDevice);
		image_->applyBilateralFilter();
		image_->applyDepthTruncation(threshold_);
		image_->applyPyrDown();
		image_->convertToPointCloud(currentPointCloud_);
		image_->applyDepthTruncation(depthDevice, threshold_);
		pcl::device::sync ();

		if(globalTime == 0) {
		
			tsdfVolume_->integrateVolume(rmats_, tvecs_, depthDevice, image_->getIntrinsics(), image_->getTrancationDistance(), image_->getDepthRawScaled(), globalTime);
			currentPointCloud_->transformPointCloud(rmats_[0], tvecs_[0], globalPreviousPointCloud_->getVertexMaps(), globalPreviousPointCloud_->getNormalMaps());
				
		} else {
			
			if(!stopTracking_) {

				hasImage_ = currentPointCloud_->alignPointClouds(rmats_, tvecs_, globalPreviousPointCloud_, image_->getIntrinsics(), globalTime);

#if (USE_HEAD_POSE_ESTIMATION)
				if(!hasImage_ && !isOnlyTrackingOn_ && headPoseEstimationOk) 
					((HeadPoseEstimationMediator*)headPoseEstimationMediator)->runHeadPoseEstimationPlusICP(this);
#endif

				if(!hasImage_)
					reset();
				else {
				
					if(!isOnlyTrackingOn_)
						tsdfVolume_->integrateVolume(rmats_, tvecs_, depthDevice, image_->getIntrinsics(), image_->getTrancationDistance(), image_->getDepthRawScaled(), globalTime);
						
					tsdfVolume_->raycast(rmats_, tvecs_, image_->getIntrinsics(), image_->getTrancationDistance(), globalPreviousPointCloud_, globalTime);
					pcl::device::sync ();
						
				}

			} else
				hasIncrement_ = false;

		}

	//}
		
	if(hasIncrement_) {

		globalTime++;
		for(int pixel = 0; pixel < 640 * 480; pixel++)
			previousDepthData[pixel] = depthMap.data[pixel];
		
	} else
		hasIncrement_ = true;

}

void Reconstruction::enableOnlyTracking(bool stopFaceDetection) 
{
	isOnlyTrackingOn_ = true;	
}

bool Reconstruction::reRunICP() 
{

	hasImage_ = currentPointCloud_->alignPointClouds(rmats_, tvecs_, globalPreviousPointCloud_, image_->getIntrinsics(), globalTime);
	if(hasImage_)
		std::cout << "Error: " << currentPointCloud_->computeFinalError() << std::endl;
	else
		std::cout << "ICP Failed" << std::endl;
	return hasImage_;

}

void Reconstruction::reRunRaycasting() 
{	
	tsdfVolume_->raycast(rmats_, tvecs_, image_->getIntrinsics(), image_->getTrancationDistance(), globalPreviousPointCloud_, globalTime);
}

void Reconstruction::transformGlobalPreviousPointCloud(Eigen::Matrix3f& Rinc, Eigen::Vector3f& tvec, Eigen::Vector3f& centerOfMass)
{

	globalPreviousPointCloud_->transformPointCloud(Rinc, tvec, globalPreviousPointCloud_->getVertexMaps(), globalPreviousPointCloud_->getNormalMaps(), init_tcam_, 
		centerOfMass);

}
 
unsigned char* Reconstruction::getRaycastImage() {
	
	int cols;
	Eigen::Vector3f cpuVolumeSize = tsdfVolume_->getVolumeSize();
	image_->getRaycastImage(viewDevice_, cpuVolumeSize, globalPreviousPointCloud_);
	viewDevice_.download (view_host_, cols);

	return (unsigned char*)view_host_.data();

}

void Reconstruction::getPointCloud(float *pointCloud, bool globalCoordinates) {
	
	int c;
	
	if(globalCoordinates)
		globalPreviousPointCloud_->getLastFrameCloud(cloudDevice_);
	else
		currentPointCloud_->getLastFrameCloud(cloudDevice_);

	cloudDevice_.download (cloudHost_, c);
	
	for(int point = 0; point < (640 * 480); point++)
	{
		pointCloud[point * 3 + 0] = cloudHost_[point].x;
		pointCloud[point * 3 + 1] = cloudHost_[point].y;
		pointCloud[point * 3 + 2] = cloudHost_[point].z;
	}
				
}

void Reconstruction::getNormalVector(float *normalVector, bool globalCoordinates) {
	
	int c;
	int inverse;

	if(globalCoordinates) {
		globalPreviousPointCloud_->getLastFrameNormals(normalsDevice_);
		inverse = 1;
	} else {
		currentPointCloud_->getLastFrameNormals(normalsDevice_);
		inverse = -1;
	}

	normalsDevice_.download (normalsHost_, c);
	
	for(int point = 0; point < (640 * 480); point++)
	{
		normalVector[point * 3 + 0] = inverse * normalsHost_[point].x;
		normalVector[point * 3 + 1] = inverse * normalsHost_[point].y;
		if((inverse * normalsHost_[point].z) > 0)
			normalVector[point * 3 + 2] = normalsHost_[point].z;
		else
			normalVector[point * 3 + 2] = inverse * normalsHost_[point].z;
	}
	
}

Reconstruction::~Reconstruction() {

	delete image_;
	delete tsdfVolume_;
	delete currentPointCloud_;
	delete globalPreviousPointCloud_;

	delete [] previousDepthData;

}

