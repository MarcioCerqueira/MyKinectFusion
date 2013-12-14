#include "VolumeRendering/MedicalVolume.h"

MedicalVolume::~MedicalVolume()
{
	delete [] data;
}

void MedicalVolume::loadTIFData(char *path, int firstSlice, int lastSlice) 
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
			float opacity = (float)(grayValue/255.f);
			data[(image - firstSlice) * imageStep + pixel * 4 + 0] = grayValue * opacity;
			data[(image - firstSlice) * imageStep + pixel * 4 + 1] = grayValue * opacity;
			data[(image - firstSlice) * imageStep + pixel * 4 + 2] = grayValue * opacity;
			data[(image - firstSlice) * imageStep + pixel * 4 + 3] = grayValue;

		}
	}

}

void MedicalVolume::loadPGMData(char *path, int firstSlice, int lastSlice) 
{

	int maxSize = lastSlice;

	IplImage *img = 0;
	IplImage *smallerImg = cvCreateImage(cvSize(maxSize, maxSize), IPL_DEPTH_8U, 3);

	char currentPath[1000];
	char imageNumber[10];
	int imageStep, grayValue, indexGlobalImage;
	
	for(int image = firstSlice; image <= lastSlice; image++)
	{

		if(image < 10)
			sprintf(imageNumber, "-%s%d", "000", image);
		else if(image < 100)
			sprintf(imageNumber, "-%s%d", "00", image);
		else if(image < 1000)
			sprintf(imageNumber, "-%s%d", "0", image);
		sprintf(currentPath, "%s%s.pgm", path, imageNumber);
		
		img = cvLoadImage(currentPath);
		//cvResize(img, smallerImg);
		//img = smallerImg;

		if(image == firstSlice) 
		{
			width = img->width;
			height = img->height;
			depth = (lastSlice - firstSlice + 1);
			data = new unsigned char[width * height * depth * 4];
			imageStep = width * height * 4;
			/*
			volumeData.resize(width);
			for(int x = 0; x < width; x++) {
				volumeData[x].resize(height);
				for(int y = 0; y < height; y++)
					volumeData[x][y].resize(depth);
			}
			*/
		}

		int x, y;
		for(int pixel = 0; pixel < img->width * img->height; pixel++)
		{
			x = pixel % width;
			y = pixel / width;
			grayValue = img->imageData[pixel * 3 + 0];
			float opacity = (float)(grayValue/255.f);
			//volumeData[x][y][image - firstSlice] = grayValue;
			data[(image - firstSlice) * imageStep + pixel * 4 + 0] = grayValue * opacity;
			data[(image - firstSlice) * imageStep + pixel * 4 + 1] = grayValue * opacity;
			data[(image - firstSlice) * imageStep + pixel * 4 + 2] = grayValue * opacity;
			data[(image - firstSlice) * imageStep + pixel * 4 + 3] = grayValue;
		}

	}

}

void MedicalVolume::loadRAWData(char *path, int width, int height, int depth) 
{
	FILE *fp;
	unsigned char *content = NULL;
	int voxelSize;
	int count = 0;

	if (path != NULL) {
		fp = fopen(path, "r");

		if (fp != NULL) {

			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			rewind(fp);

			if (count > 0) {
				voxelSize = count;
				content = (unsigned char *) malloc(sizeof(unsigned char) * (count + 1));
				data = new unsigned char[count * 4];
				count = fread(content, sizeof(char), count, fp);
				content[count] = '\0';
			//printf("\nSize of file %d bytes\n", count);
			}//end if
			fclose(fp);
		}//end if
	} else{ 
		printf("\nError reading file\n");
	}

	this->width = width;
	this->height = height;
	this->depth = depth;
	for(int voxel = 0; voxel < voxelSize; voxel++) {
		float opacity = (float)(content[voxel]/255.f);
		data[voxel * 4 + 0] = content[voxel] * opacity;
		data[voxel * 4 + 1] = content[voxel] * opacity;
		data[voxel * 4 + 2] = content[voxel] * opacity;
		data[voxel * 4 + 3] = content[voxel];
	}

	delete [] content;
}