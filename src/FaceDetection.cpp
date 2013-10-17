#include "FaceDetection.h"
#include <iostream>

FaceDetection::FaceDetection(char *cascadeFileName)
{
	std::ifstream in(cascadeFileName);
	std::string info;

	cascade = 0;

	if(in.is_open())
	{

		in >> info;
		in >> cascade_name;
    
	} else {
		std::cout << "Cascade Config File not found" << std::endl;
	}
	in.close();

	rgbImage = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
	depthData = (unsigned short*)malloc(640 * 480 * sizeof(unsigned short));

}

FaceDetection::~FaceDetection()
{
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseImage(&rgbImage);
	delete [] depthData;
}

void FaceDetection::segmentFace(IplImage *img, unsigned short *depthData)
{
	// Pintando tudo fora do retângulo de preto
	int height = img->height;
	int width  = img->width;
	int x, y;

	for(int pixel = 0; pixel < width * height; pixel++)
	{
		x = pixel % width;
		y = pixel / width;
		if(x < pt1.x || y < pt1.y || x > pt2.x || y > pt2.y)
		{
			img->imageData[pixel * 3 + 0] = 0;
			img->imageData[pixel * 3 + 1] = 0;
			img->imageData[pixel * 3 + 2] = 0;
		}
	}

	// Draw the rectangle in the input image
    cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
		
	//We update the depth map based on the region of the face
	for(int pixel = 0; pixel < img->width * img->height; pixel++)
		if(img->imageData[pixel * 3 + 0] == 0 && img->imageData[pixel * 3 + 1] == 0 && img->imageData[pixel * 3 + 2] == 0)
			depthData[pixel] = 0;

}

bool FaceDetection::detectFace(IplImage* img, unsigned short *depthData, CvMemStorage* storage, CvHaarClassifierCascade* cascade)
{
	
	// There can be more than one face in an image. So create a growable sequence of faces.
    // Detect the objects and store them in the sequence
    CvSeq* faces = cvHaarDetectObjects( img, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(40, 40) );
	
	int scale = 1;
	int i;
    // Loop the number of faces found.
	for(i = 0; i < faces->total; i++)
	{
		//i = 0;
		// Create a new rectangle for drawing the face
        CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
			
        // Find the dimensions of the face, and scale it if necessary
        pt1.x = r->x*scale - 10;
        pt2.x = (r->x+r->width)*scale + 10;
        pt1.y = r->y*scale - 10;
        pt2.y = (r->y+r->height)*scale + 10;

 		return true;
		
	}
	return false;
}
// Function to detect and draw any faces that is present in an image
bool FaceDetection::run(boost::shared_ptr<openni_wrapper::Image>& rgbImage, boost::shared_ptr<openni_wrapper::DepthImage>& depthImage)
{

	if(!cascade)
	{
		cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
	}

	// Check whether the cascade has loaded successfully. Else report and error and quit
    if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        return false;
    }

	static CvMemStorage* storage = 0;
	// Allocate the memory storage
	if(!storage)
		storage = cvCreateMemStorage(0);
	
	// Convert the data
	rgbImage->fillRGB(this->rgbImage->width, this->rgbImage->height, (unsigned char*)this->rgbImage->imageData);
	depthImage->fillDepthImageRaw(this->rgbImage->width, this->rgbImage->height, depthData);

	// Find whether the cascade is loaded, to find the faces. If yes, then:
    if( cascade )
    {
		if(detectFace(this->rgbImage, depthData, storage, cascade))
		{
			segmentFace(this->rgbImage, depthData);

			rgbImage = convertIplImageToOpenNIWrapper();
			depthImage = convertUnsignedShortToOpenNIWrapper();
			return true;
		}
	}
	
	// Clear the memory storage which was used before
    cvClearMemStorage( storage );
	return false;

}

void FaceDetection::segmentFace(boost::shared_ptr<openni_wrapper::Image>& rgbImage, boost::shared_ptr<openni_wrapper::DepthImage>& depthImage)
{
	// Convert the data
	rgbImage->fillRGB(this->rgbImage->width, this->rgbImage->height, (unsigned char*)this->rgbImage->imageData);
	depthImage->fillDepthImageRaw(this->rgbImage->width, this->rgbImage->height, depthData);
	
	segmentFace(this->rgbImage, depthData);
	
	rgbImage = convertIplImageToOpenNIWrapper();
	depthImage = convertUnsignedShortToOpenNIWrapper();

}

boost::shared_ptr<openni_wrapper::Image> FaceDetection::convertIplImageToOpenNIWrapper() {
	
	boost::shared_ptr<xn::ImageMetaData> imageMetaData (new xn::ImageMetaData);
	imageMetaData->AllocateData(this->rgbImage->width, this->rgbImage->height, XN_PIXEL_FORMAT_RGB24);
	XnRGB24Pixel* imageMap = imageMetaData->WritableRGB24Data();
	for(int pixel = 0; pixel < this->rgbImage->width * this->rgbImage->height; pixel++) {
		imageMap[pixel].nRed = static_cast<XnUInt8>(this->rgbImage->imageData[pixel * 3 + 0]);
		imageMap[pixel].nGreen = static_cast<XnUInt8>(this->rgbImage->imageData[pixel * 3 + 1]);
		imageMap[pixel].nBlue = static_cast<XnUInt8>(this->rgbImage->imageData[pixel * 3 + 2]);
	}
	boost::shared_ptr<openni_wrapper::Image> image (new openni_wrapper::ImageRGB24(imageMetaData));
	return image;

}

boost::shared_ptr<openni_wrapper::DepthImage> FaceDetection::convertUnsignedShortToOpenNIWrapper() {
	
	boost::shared_ptr<xn::DepthMetaData> depthMetaData (new xn::DepthMetaData);
	depthMetaData->AllocateData(this->rgbImage->width, this->rgbImage->height);
	XnDepthPixel* depthMap = depthMetaData->WritableData();
	for(int pixel = 0; pixel < this->rgbImage->width * this->rgbImage->height; pixel++) {
		depthMap[pixel] = static_cast<XnDepthPixel>(depthData[pixel]);
	}
	boost::shared_ptr<openni_wrapper::DepthImage> depthImage (new openni_wrapper::DepthImage(depthMetaData, 0.075f, 525.f, 0, 0));
	return depthImage;
}
	