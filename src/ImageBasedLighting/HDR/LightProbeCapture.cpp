#include "ImageBasedLighting/HDR/LightProbeCapture.h"

LightProbeCapture::LightProbeCapture(int cameraID, int lightProbeSize) {

	capture = new cv::VideoCapture(cameraID);
	if(!capture->isOpened()) std::cout << "Capture error" << std::endl;
	lightProbeCenter[0] = 320;
	lightProbeCenter[1] = 240;
	this->lightProbeSize = lightProbeSize;
	croppedImage = cv::Mat(lightProbeSize, lightProbeSize, CV_8UC3);

}

LightProbeCapture::~LightProbeCapture() {
	
	delete capture;

}

void LightProbeCapture::captureSphericalMap() {

	*capture >> image;
	cv::resize(image, image, cv::Size(640, 480));
	
	//Segment light probe
	unsigned char *pointer = image.ptr<unsigned char>();
	for(int y = 0; y < 480; y++) {
		for(int x = 0; x < 640; x++) {
			int pixel = y * 640 + x;
			if(y < lightProbeCenter[1] - lightProbeSize/2 || y > lightProbeCenter[1] + lightProbeSize/2 || 
			   x < lightProbeCenter[0] - lightProbeSize/2 || x > lightProbeCenter[0] + lightProbeSize/2) {
				for(int ch = 0; ch < 3; ch++)
					pointer[pixel * 3 + ch] = 0;
			}
		}
	}
	
	unsigned char *croppedPointer = croppedImage.ptr<unsigned char>();

	int x0 = lightProbeCenter[0] - lightProbeSize/2;
	int x1 = lightProbeCenter[0] + lightProbeSize/2;
	int y0 = lightProbeCenter[1] - lightProbeSize/2;
	int y1 = lightProbeCenter[1] + lightProbeSize/2;

	for(int y = y0; y < y1; y++) {
		for(int x = x0; x < x1; x++) {
			int newPixel = (y - y0) * lightProbeSize + (x - x0);
			int oldPixel = y * 640 + x;
			for(int ch = 0; ch < 3; ch++)
				croppedPointer[newPixel * 3 + ch] = pointer[oldPixel * 3 + ch];
		}
	}

}