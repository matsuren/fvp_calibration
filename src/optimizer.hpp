#pragma once
#include <Eigen/Geometry>
#include "ocam_functions.hpp"
#include "reprojection_error.hpp"
#include "calibration.hpp"


class Optimizer
{
public:
	Optimizer(const std::string data_folder, const cv::Mat& obj_pts, const std::string origin_tag, const int camera_num, const CameraType cam_type);
	void optimize();
	void saveResults();
private:
	const std::string DATAFOLDER;
	const std::string ORIGIN_TAG;
	const int CAM_NUM;
	const CameraType cam_type_;

	std::map<std::string, std::array<double, 3>> tag_parameters_;
	std::map<std::string, std::array<double, 6>> camera_parameters_;

	Eigen::Matrix<double, 3, 4> obj_pts_eigen_;
	std::map<std::string, cv::Mat> observations_; // key:img0tag0
	std::map<std::string, ocam_model> ocam_models_;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
