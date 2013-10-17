#ifndef VOLUMETRICDATA_H
#define VOLUMETRICDATA_H

#include <opencv2\opencv.hpp>

class VolumetricData
{
public:
	~VolumetricData();
	void loadTIFData(char *path, int firstSlice, int lastSlice);

	unsigned char* getData() { return data; }
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getDepth() { return depth; }
private:
	unsigned char* data;
	int width;
	int height;
	int depth;
};

#endif