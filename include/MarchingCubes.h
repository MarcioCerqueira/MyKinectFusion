#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include <pcl/point_types.h>
#include "internal.h"
#include "TsdfVolume.h"

class MarchingCubes
{
public:

	MarchingCubes();
	DeviceArray<pcl::PointXYZ> run(TsdfVolume *tsdfVolume, DeviceArray<pcl::PointXYZ>& trianglesBuffer);
	enum
    { 
		POINTS_PER_TRIANGLE = 3,
        DEFAULT_TRIANGLES_BUFFER_SIZE = 2 * 1000 * 1000 * POINTS_PER_TRIANGLE      
    };
private:
	/** \brief Edge table for marching cubes  */
    DeviceArray<int> edgeTable_;
    /** \brief Number of vertextes table for marching cubes  */
    DeviceArray<int> numVertsTable_;
    /** \brief Triangles table for marching cubes  */
    DeviceArray<int> triTable_;     
    /** \brief Temporary buffer used by marching cubes (first row stores occuped voxes id, second number of vetexes, third poits offsets */
    DeviceArray2D<int> occupiedVoxelsBuffer_;
};
#endif
