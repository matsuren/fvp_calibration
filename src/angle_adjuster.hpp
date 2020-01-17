#pragma once
#include <string>
#include <opencv2/core.hpp>
#include "ocam_functions.hpp"

class AngleAdjuster
{
public:
	AngleAdjuster(const std::string data_folder, const int camera_num, const std::string & load_yml = "refined_camera_poses.yml");
	void generateBEV(cv::Mat & bev);
	void rotate(cv::Mat & img);
	void savePoses(const std::string save_yml = "final_camera_poses.yml");
private:
	int getClosestCamera(const cv::Mat &pt, const std::vector<cv::Mat> &poses_wc);

	int WIDTH = 512;
	int HEIGHT = 512;
	int ANGLE_MAX = 360;
	int current_angle = ANGLE_MAX / 2;

	std::vector<cv::Mat> poses_cw;
	std::vector<cv::Mat> poses_wc;
	std::vector<cv::Mat> frames;
	std::vector<ocam_model> fisheye_models;
	const std::string DATAFOLDER;
	const int CAM_NUM;
	double pixel_to_m; // [m]/[pixel]
	
};
