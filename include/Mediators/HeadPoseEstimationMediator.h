#ifndef HEAD_POSE_ESTIMATION_MEDIATOR_H
#define HEAD_POSE_ESTIMATION_MEDIATOR_H

#include "HeadPoseEstimator/HeadPoseEstimator.h"
#include "Reconstruction.h"

class Reconstruction;
class HeadPoseEstimationMediator
{
public:
	HeadPoseEstimationMediator(char *HPEConfigFileName);
	~HeadPoseEstimationMediator();
	//initialGlobalTranslation = init_tcam_
	void stopTracking(bool stop, Reconstruction *reconstruction);
	void runHeadPoseEstimationPlusICP(Reconstruction *reconstruction);
private:
	HeadPoseEstimator *headPoseEstimator;
	char *HPEConfigFileName;
	Eigen::Matrix3f previousRotationMatrixEstimated;
	Eigen::Vector3f previousHeadCenter;
	
};

#endif