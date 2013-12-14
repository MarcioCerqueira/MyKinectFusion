#include "VolumeRendering/MinMaxOctree.h"

MinMaxOctree::MinMaxOctree(int width, int height, int depth) {

	this->width = width/8;
	this->height = height/8;
	this->depth = depth/8;
	data = (unsigned char*)malloc(this->width * this->height * this->depth * 4); 

	for(int voxel = 0; voxel < this->width * this->height * this->depth; voxel++) {
		data[voxel * 4 + 0] = 255;
		data[voxel * 4 + 1] = 0;
		data[voxel * 4 + 2] = 0;
		data[voxel * 4 + 3] = 0;
	}

}

MinMaxOctree::~MinMaxOctree() {
	delete [] data;
}

void MinMaxOctree::build(unsigned char* volumeData, int volumeWidth, int volumeHeight, int volumeDepth) {
	
	int octreeIndex, volumeIndex;
	for(int w = 0; w < this->width; w++) {
		for(int h = 0; h < this->height; h++) {
			for(int d = 0; d < this->depth; d++) {
				
				for(int vw = 0; vw < 8; vw++) {
					for(int vh = 0; vh < 8; vh++) {
						for(int vd = 0; vd < 8; vd++) {
							volumeIndex = (d * 8 + vd) * volumeHeight * volumeWidth + (h * 8 + vh) * volumeWidth + (w * 8 + vw);
							octreeIndex = d * height * width + h * width + w;
							if(volumeData[volumeIndex * 4] < data[octreeIndex * 4 + 0])
								data[octreeIndex * 4 + 0] = volumeData[volumeIndex * 4];
							if(volumeData[volumeIndex * 4] > data[octreeIndex * 4 + 1])
								data[octreeIndex * 4 + 1] = volumeData[volumeIndex * 4];
						}
					}
				}

			}
		}
	}

}