#ifndef MINMAXOCTREE_H
#define MINMAXOCTREE_H

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

class MinMaxOctree
{
public:
	MinMaxOctree(int width, int height, int depth);
	~MinMaxOctree();
	void build(unsigned char *volumeData, int volumeWidth, int volumeHeight, int volumeDepth);
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getDepth() { return depth; }
	unsigned char* getData() { return data; }
private:
	unsigned char *data;
	int width;
	int height;
	int depth;
};
#endif