#ifndef MEDICAL_VOLUME_H
#define MEDICAL_VOLUME_H

#include <opencv2\opencv.hpp>

class MedicalVolume
{
public:
	~MedicalVolume();
	void loadTIFData(char *path, int firstSlice, int lastSlice);
	void loadPGMData(char *path, int firstSlice, int lastSlice);
	void loadRAWData(char *path, int width, int height, int depth);
	unsigned char* getData() { return data; }
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getDepth() { return depth; }
private:
	//4-channel data
	unsigned char* data;
	int width;
	int height;
	int depth;
};

#endif