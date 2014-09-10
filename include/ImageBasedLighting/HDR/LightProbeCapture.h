#ifndef LIGHTPROBECAPTURE_H
#define LIGHTPROBECAPTURE_H

#include <opencv2\opencv.hpp>

static int lightProbeCenter[2];

class LightProbeCapture
{
public:
	LightProbeCapture(int cameraID);
	~LightProbeCapture();
	void captureSphericalMap();
	cv::Mat getImage() { return croppedImage; }
	cv::Mat getOriginalImage() { return image; }
private:
	cv::VideoCapture *capture;
	int lightProbeSize;
	cv::Mat image;
	cv::Mat croppedImage;
};

#endif