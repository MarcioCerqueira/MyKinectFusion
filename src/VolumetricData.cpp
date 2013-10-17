#include "VolumetricData.h"

VolumetricData::~VolumetricData()
{
	delete [] data;
}

void VolumetricData::loadTIFData(char *path, int firstSlice, int lastSlice) 
{
	IplImage *img = 0;
	char currentPath[1000];
	int imageStep, grayValue;

	for(int image = firstSlice; image <= lastSlice; image++)
	{
		sprintf(currentPath, "%s%d.tif", path, image);
		img = cvLoadImage(currentPath);
		if(image == firstSlice) 
		{
			width = img->width;
			height = img->height;
			depth = (lastSlice - firstSlice + 1);
			data = new unsigned char[width * height * depth * 4];
			imageStep = width * height * 4;
		}

		for(int pixel = 0; pixel < img->width * img->height; pixel++)
		{
			grayValue = img->imageData[pixel * 3 + 0];
			data[(image - firstSlice) * imageStep + pixel * 4 + 0] = grayValue;
			data[(image - firstSlice) * imageStep + pixel * 4 + 1] = grayValue;
			data[(image - firstSlice) * imageStep + pixel * 4 + 2] = grayValue;
			data[(image - firstSlice) * imageStep + pixel * 4 + 3] = grayValue;
		}
	}
}