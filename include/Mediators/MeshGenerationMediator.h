#ifndef MESHGENERATIONMEDIATOR_H
#define MESHGENERATIONMEDIATOR_H

#include <pcl/io/ply_io.h>
#include "MarchingCubes.h"
#include "Mesh.h"
#include "TsdfVolume.h"
#include "VolumeRendering/MedicalVolume.h"

class MeshGenerationMediator
{
public:
	MeshGenerationMediator();
	~MeshGenerationMediator();
	void saveMesh(TsdfVolume *tsdfVolume);
	pcl::PointCloud<pcl::PointXYZ> computeMesh(MedicalVolume *medicalVolume, float minValue, float samplingFactor);
private:
	MarchingCubes *marchingCubes;
};

#endif