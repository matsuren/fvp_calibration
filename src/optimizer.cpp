#include <ceres/ceres.h>
#include <Eigen/Geometry>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "ocam_functions.hpp"
#include "reprojection_error.hpp"
#include "calibration.hpp"
#include "optimizer.hpp"

Optimizer::Optimizer(const std::string data_folder, const cv::Mat& obj_pts, const std::string origin_tag, const int camera_num, const CameraType cam_type)
	:DATAFOLDER(data_folder), ORIGIN_TAG(origin_tag), CAM_NUM(camera_num), cam_type_(cam_type)
{
	/////////////////////////////////////////////
	// Load data
	/////////////////////////////////////////////
	// convert to eigen
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; j++)
		{
			obj_pts_eigen_(i, j) = obj_pts.at<double>(j, i);
		}
	}
	//std::cout << obj_pts_eigen_ << std::endl;

	// read initial parameters for tags
	cv::FileStorage fs_params(DATAFOLDER + "/initial_params.yml", cv::FileStorage::READ);
	if (!fs_params.isOpened())
		throw std::exception("Cannot open file: initial_params.yml!");
	for (auto &it = fs_params.root().begin(); it != fs_params.root().end(); ++it)
	{
		std::string key = (*it).name();
		tag_parameters_[key] = {};
		for (int i = 0; i < 3; i++)
		{
			tag_parameters_[key][i] = (*it)[i];
		}
	}
	// read initial poses for cameras
	std::map<std::string, cv::Mat> poses;
	cv::FileStorage fs_pose(DATAFOLDER + "/initial_camera_poses.yml", cv::FileStorage::READ);
	if (!fs_pose.isOpened())
		throw std::exception("Cannot open file: initial_camera_poses.yml!");
	for (size_t i = 0; i < CAM_NUM; ++i)
	{
		const std::string img_key = "img" + std::to_string(i);
		const std::string key = ORIGIN_TAG + img_key;
		cv::Mat pose, invpose,rvec, tvec;
		fs_pose[key] >> pose;
		invpose = pose.inv();
		cv::Rodrigues(invpose(cv::Rect(0, 0, 3, 3)), rvec);
		tvec = invpose(cv::Rect(3, 0, 1, 3));
		camera_parameters_[img_key] = {};
		for (int i = 0; i < 3; i++)
		{
			camera_parameters_[img_key][i] = rvec.at<double>(i);
			camera_parameters_[img_key][3 + i] = tvec.at<double>(i);
		}
	}
	// load points
	cv::FileStorage fs_pts(DATAFOLDER + "/tag_points.yml", cv::FileStorage::READ);
	if (!fs_pts.isOpened())
		throw std::exception("Cannot open file: tag_points.yml!");
	for (auto &fit = fs_pts.root().begin(); fit != fs_pts.root().end(); ++fit)
	{
		cv::FileNode item = *fit;
		std::string key = item.name();
		cv::Mat pts;
		fs_pts[key] >> pts;
		observations_[key] = pts;
	}

}


void Optimizer::optimize()
{
	ceres::Problem problem;
	/////////////////////////////////////////////
	// Add residual error
	/////////////////////////////////////////////
	for (const auto &it : observations_) {
		std::array<double, 8> observation;
		for (int i = 0; i < 4; i++)
		{
			observation[2 * i] = double(it.second.at<double>(i, 0));
			observation[2 * i + 1] = double(it.second.at<double>(i, 1));
		}
		const std::string img_key = it.first.substr(0, 4);
		const std::string tag_key = it.first.substr(4, 4);


		// load camera parameters
		ceres::CostFunction* cost_function;
		
		if (cam_type_ == CameraType::Perspective) {
			const std::string camera_model_file = DATAFOLDER + "/calib_perspective_results_" + img_key.substr(3, 1) + ".yml";
			//auto error = PerspectiveReprojectionError(observation, camera_model_file, obj_pts_eigen_);
			//double res[2];
			//error(camera_parameters_[img_key], tag_parameters_[tag_key], res);
			cost_function = PerspectiveReprojectionError::Create(observation, camera_model_file, obj_pts_eigen_);
		}
		else if (cam_type_ == CameraType::Fisheye) {
			const std::string camera_model_file = DATAFOLDER + "/calib_results_" + img_key.substr(3, 1) + ".txt";
			cost_function = OcamReprojectionError::Create(observation, camera_model_file, obj_pts_eigen_);
		}

		problem.AddResidualBlock(cost_function,
			NULL /* squared loss */, camera_parameters_[img_key].data(), tag_parameters_[tag_key].data());
	}
	// Set parameter for origin_tag as constant
	problem.SetParameterBlockConstant(tag_parameters_[ORIGIN_TAG].data());

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	std::cout << "Start optimization" << std::endl;
	ceres::Solve(options, &problem, &summary);
	std::cout << "Finished!" << std::endl;
	//std::cout << summary.FullReport() << "\n";
	std::cout << summary.BriefReport() << "\n";
}

void Optimizer::saveResults()
{
	// save results
	cv::FileStorage fs_pose(DATAFOLDER + "/refined_camera_poses.yml", cv::FileStorage::WRITE);
	for (const auto &it : camera_parameters_) {
		cv::Mat pose = cv::Mat::eye(cv::Size(4, 4), CV_64F);
		cv::Mat R;
		const std::string key = it.first;
		const auto camera = it.second;
		cv::Mat rvec = cv::Mat(3, 1, CV_64F);
		cv::Mat tvec = cv::Mat(3, 1, CV_64F);
		// first 3 elements are for rotation
		// last 3 elements are for translation
		for (int i = 0; i < 3; i++) {
			rvec.at<double>(i, 0) = camera[i];
			tvec.at<double>(i, 0) = camera[i + 3];
		}
		cv::Rodrigues(rvec, R);
		R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
		tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));
		fs_pose << key + "origin" << pose;
		fs_pose << "origin" + key << pose.inv();
	}
	// save tag parameters
	cv::FileStorage fs_params(DATAFOLDER + "/refined_params.yml", cv::FileStorage::WRITE);
	for (const auto &it : tag_parameters_) {
		const std::string key = it.first;
		const auto tag = it.second;
		fs_params << key << "[" << tag[0] << tag[1] << tag[2] << "]";
	}
	return;
}

//int main(int argc, char** argv) {
//
//
//	const double SQUARE_SIZE = 0.159; // size in m
//	// corners
//	const cv::Mat OBJ_PTS = (cv::Mat_<double>(4, 3) <<
//		0, 0, 0,
//		SQUARE_SIZE, 0, 0,
//		SQUARE_SIZE, SQUARE_SIZE, 0,
//		0, SQUARE_SIZE, 0);
//
//	google::InitGoogleLogging(argv[0]);
//
//	// GLOBAL
//	const std::string DATAFOLDER = "../data";
//	const std::string ORIGIN_TAG = "tag0";
//
//	const int CAM_NUM = 4;
//	//const CameraType cam_type = CameraType::Perspective; // CameraType::Fisheye; //
//	const CameraType cam_type = CameraType::Fisheye; // CameraType::Fisheye; //
//
//	Optimizer optimizer(DATAFOLDER, OBJ_PTS, ORIGIN_TAG, CAM_NUM, cam_type);
//	optimizer.optimize();
//	optimizer.saveResults();
//	return 0;
//
//}