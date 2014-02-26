#ifndef MEDICAL_VOLUME_H
#define MEDICAL_VOLUME_H

#include <opencv2\opencv.hpp>
#include <vector>

class MedicalVolume
{
public:
	~MedicalVolume();
	void loadTIFData(char *path, int firstSlice, int lastSlice);
	void loadPGMData(char *path, int firstSlice, int lastSlice);
	void loadRAWData(char *path, int width, int height, int depth);
	void organizeData();
	unsigned char* getData() { return data; }
	std::vector< std::vector< std::vector<float> > > getOrganizedData() { return grid; }
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getDepth() { return depth; }
private:
	//4-channel data
	unsigned char* data;
	std::vector< std::vector< std::vector<float> > > grid;
	int width;
	int height;
	int depth;
};

#endif