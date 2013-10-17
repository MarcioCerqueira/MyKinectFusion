#include "MyPointCloud.h"

template<class D, class Matx> D&
device_cast (Matx& matx)
{
  return (*reinterpret_cast<D*>(matx.data ()));
}

MyPointCloud::MyPointCloud(int cols, int rows) {

	vmaps_.resize(LEVELS);
	nmaps_.resize(LEVELS);

	for(int i = 0; i < LEVELS; ++i) {
	
		int pyr_rows = rows >> i;
		int pyr_cols = cols >> i;

		vmaps_[i].create(pyr_rows * 3, pyr_cols);
		nmaps_[i].create(pyr_rows * 3, pyr_cols);

	}

	rows_ = rows;
	cols_ = cols;

	//10, 5, 4
	const int iters[] = {10, 5, 4};
	std::copy (iters, iters + LEVELS, icpIterations_);

	distThres_ = 100;
	angleThres_ = sin (20.f * 3.14159254f / 180.f);

	gbuf_.create (27, 20*60);
	sumbuf_.create (27);

}

void MyPointCloud::transformPointCloud(Matrix3frm Rcam, Vector3f tcam, std::vector<device::MapArr> &vmapDst, std::vector<device::MapArr> &nmapDst, bool inverse) {

	if(inverse)
		Rcam = Rcam.inverse();

    device::Mat33&  device_Rcam = device_cast<device::Mat33> (Rcam);
    float3& device_tcam = device_cast<float3>(tcam);

	for (int i = 0; i < LEVELS; ++i)
		device::tranformMaps(vmaps_[i], nmaps_[i], device_Rcam, device_tcam, vmapDst[i], nmapDst[i], inverse);

}

void MyPointCloud::transformPointCloud(Matrix3frm Rcam, Vector3f tcam, std::vector<device::MapArr> &vmapDst, std::vector<device::MapArr> &nmapDst,
	Eigen::Vector3f& newOrigin, Eigen::Vector3f& objectCentroid) {
	
    device::Mat33&  device_Rcam = device_cast<device::Mat33> (Rcam);
    float3& device_tcam = device_cast<float3>(tcam);
	float3& device_newOrigin = device_cast<float3>(newOrigin);
	float3& device_objectCentroid = device_cast<float3>(objectCentroid);

	for (int i = 0; i < LEVELS; ++i)
		device::tranformMaps(vmaps_[i], nmaps_[i], device_Rcam, device_tcam, vmapDst[i], nmapDst[i], device_newOrigin, device_objectCentroid);
}

bool MyPointCloud::alignPointClouds(std::vector<Matrix3frm>& Rcam, std::vector<Vector3f>& tcam, MyPointCloud *globalPreviousPointCloud, device::Intr& intrinsics, int globalTime) {

	Matrix3frm Rprev = Rcam[globalTime - 1]; //  [Ri|ti] - pos of camera, i.e.
	Vector3f tprev = tcam[globalTime - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
	Matrix3frm Rprev_inv = Rprev.inverse(); //Rprev.t();
	
	device::Mat33& device_Rprev_inv = device_cast<device::Mat33> (Rprev_inv);
	float3& device_tprev = device_cast<float3> (tprev);

	Matrix3frm Rcurr = Rprev; // tranform to global coo for ith camera pose
	Vector3f tcurr = tprev;
	
	for(int level = LEVELS - 1; level >= 0; --level) {
	
		int iterations = icpIterations_[level];

		for(int iteration = 0; iteration < iterations; ++iteration) {
		
			device::Mat33& device_Rcurr = device_cast<device::Mat33> (Rcurr);
			float3& device_tcurr = device_cast<float3>(tcurr);
		
			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A;
			Eigen::Matrix<float, 6, 1> b;

			if(level == 2 && iteration == 0)			
				error_.create(rows_ * 4, cols_);

			device::estimateCombined (device_Rcurr, device_tcurr, vmaps_[level], nmaps_[level], device_Rprev_inv, device_tprev, intrinsics (level),
                          globalPreviousPointCloud->getVertexMaps()[level], globalPreviousPointCloud->getNormalMaps()[level], distThres_, angleThres_, 
						  gbuf_, sumbuf_, A.data (), b.data (), error_);

			//checking nullspace
			float det = A.determinant ();

			if (fabs (det) < 1e-15 || !pcl::device::valid_host (det)) {
				std::cout << "NaN value (ICP Failed)" << std::endl;
				return (false);
			}

			Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b);
			//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

			float alpha = result (0);
			float beta  = result (1);
			float gamma = result (2);

			Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
			Vector3f tinc = result.tail<3> ();

			//compose
			tcurr = Rinc * tcurr + tinc;
			Rcurr = Rinc * Rcurr;
		
		}
	
	}

	//save tranform
	Rcam[globalTime] = Rcurr;
	tcam[globalTime] = tcurr;
	return (true);

}

float MyPointCloud::computeFinalError() {
	
	pcl::PointCloud<PointXYZI>::Ptr error;
	DeviceArray2D<pcl::PointXYZI> errorInRGBDevice_;
	error = PointCloud<PointXYZI>::Ptr (new PointCloud<PointXYZI>);
	int cols;
	this->getHostErrorInRGB(errorInRGBDevice_);
	errorInRGBDevice_.download (error->points, cols);
	error->width = errorInRGBDevice_.cols ();
	error->height = errorInRGBDevice_.rows ();

	float error2 = 0;
	int count = 0;
	for(int point = 0; point < error->points.size(); point++) {
		if(error->points[point].intensity != -1) {
			error2 += error->points[point].intensity;
			count++;
		}
	}
	std::cout << count << std::endl;
	return error2/count;
}

void MyPointCloud::getHostErrorInRGB(DeviceArray2D<pcl::PointXYZI>& errorInRGB) {

	errorInRGB.create(rows_ , cols_);
	DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)errorInRGB;
	device::convertXYZI(error_, c);

}

void MyPointCloud::getLastFrameCloud(DeviceArray2D<pcl::PointXYZ>& cloud) {
	
	cloud.create (rows_, cols_);
	DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
	device::convert (vmaps_[0], c);

}


void MyPointCloud::getLastFrameNormals(DeviceArray2D<pcl::PointXYZ>& normals) {
	
	normals.create (rows_, cols_);
	DeviceArray2D<float4>& n = (DeviceArray2D<float4>&)normals;
	device::convert<float4>(nmaps_[0], n);

}

Eigen::Vector3f& MyPointCloud::getCentroid() {
	
	DeviceArray2D<pcl::PointXYZ> cloudDevice;
	pcl::PointCloud<PointXYZ>::Ptr hostFrameCloud;

	this->getLastFrameCloud(cloudDevice);

	int c;
	hostFrameCloud = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
	cloudDevice.download (hostFrameCloud->points, c);
	hostFrameCloud->width = cloudDevice.cols ();
	hostFrameCloud->height = cloudDevice.rows ();
	
	Eigen::Vector3f centroid;
	centroid.setZero();

	int count = 0;
	for(int point = 0; point < hostFrameCloud->points.size(); point++) {

		if(hostFrameCloud->points[point].z == hostFrameCloud->points[point].z) {
			
			if(hostFrameCloud->points[point].z != 0) {

				centroid(0) += hostFrameCloud->points[point].x;
				centroid(1) += hostFrameCloud->points[point].y;
				centroid(2) += hostFrameCloud->points[point].z;
				count++;
			
			}

		}

	}

	centroid /= count;
	return centroid;

}


void MyPointCloud::getDepthMap(unsigned short *depthMap) {
	
	for(int pixel = 0; pixel < (640 * 480); pixel++)
		depthMap[pixel] = 0;

	DeviceArray2D<pcl::PointXYZ> cloudDevice;
	pcl::PointCloud<PointXYZ>::Ptr hostFrameCloud;

	this->getLastFrameCloud(cloudDevice);

	int c;
	hostFrameCloud = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
	cloudDevice.download (hostFrameCloud->points, c);
	hostFrameCloud->width = cloudDevice.cols ();
	hostFrameCloud->height = cloudDevice.rows ();
	
	int count = 0;
	int x, y, pixel;
	for(int point = 0; point < hostFrameCloud->points.size(); point++) {

		if(hostFrameCloud->points[point].z == hostFrameCloud->points[point].z) {
			
			if(hostFrameCloud->points[point].z != 0) {
				x = (hostFrameCloud->points[point].x * 525.f / hostFrameCloud->points[point].z) + 320;
				y = (hostFrameCloud->points[point].y * 525.f / hostFrameCloud->points[point].z) + 240;
				pixel = y * 640 + x;
				if(pixel >= 0 && pixel < (640 * 480))
					depthMap[pixel] = hostFrameCloud->points[point].z;
			}

		}

	}

}
