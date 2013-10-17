#ifndef MYPOINTCLOUD_H
#define MYPOINTCLOUD_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <vector>
#include "pcl/gpu/containers/kernel_containers.hpp"
#include "internal.h"
#include "openni_capture.h"

#define LEVELS 3

using namespace std;
using namespace Eigen;
using namespace pcl;
using namespace pcl::gpu;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
struct float8 { float x, y, z, w, f1, f2, f3, f4; };
class MyPointCloud
{
public:
	MyPointCloud(int cols, int rows);
	
	std::vector<device::MapArr>& getVertexMaps() { return vmaps_; };
	std::vector<device::MapArr>& getNormalMaps() { return nmaps_; };
	void getHostErrorInRGB(DeviceArray2D<pcl::PointXYZI>& errorInRGB);

	void setVertexMaps(std::vector<device::MapArr>& vmaps) { vmaps_ = vmaps; };
	void setNormalMaps(std::vector<device::MapArr>& nmaps) { nmaps_ = nmaps; };

	void transformPointCloud(Matrix3frm Rcam, Vector3f tcam, std::vector<device::MapArr> &vmapDst, std::vector<device::MapArr> &nmapDst, bool inverse = false);
	void transformPointCloud(Matrix3frm Rcam, Vector3f tcam, std::vector<device::MapArr> &vmapDst, std::vector<device::MapArr> &nmapDst, 
		Eigen::Vector3f& newOrigin, Eigen::Vector3f& objectCentroid);
	bool alignPointClouds(std::vector<Matrix3frm>& Rcam, std::vector<Vector3f>& tcam, MyPointCloud *globalPreviousPointCloud, device::Intr& intrinsics, int globalTime);
	float computeFinalError();
	

	void getLastFrameCloud(DeviceArray2D<pcl::PointXYZ>& cloud);
	void getLastFrameNormals(DeviceArray2D<pcl::PointXYZ>& normals);
	Eigen::Vector3f& getCentroid();
	void getDepthMap(unsigned short *depthMap);

private:
	std::vector<device::MapArr> vmaps_;
    std::vector<device::MapArr> nmaps_;
	std::vector<float> hostError_;
	int rows_, cols_;
	int icpIterations_[LEVELS];
	float  distThres_, angleThres_;
	DeviceArray2D<float> gbuf_;
    DeviceArray<float> sumbuf_;
	DeviceArray2D<float> error_;
};
#endif
