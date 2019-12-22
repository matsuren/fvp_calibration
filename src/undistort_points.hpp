#pragma once
#include <opencv2/core.hpp>
#include "ocam_functions.hpp"

void undistortPoints(const std::string camera_model_file, const cv::Mat &pts, cv::Mat &ray_dirs) {
	ocam_model fisheye_model;
	int ret = get_ocam_model(&fisheye_model, camera_model_file.c_str());
	if (ret == -1)
		throw std::exception("Can't open ocamcalib file!");

	ray_dirs = cv::Mat::zeros(cv::Size(4, 3), pts.type());
	for (int i = 0; i < 4; ++i) {
		// Be careful about coordinate in OCamCalib
		double img_pt[2] = { pts.at<double>(i, 1), pts.at<double>(i, 0) };
		double world_pt[3];
		cam2world(world_pt, img_pt, &fisheye_model);
		double x, y, z;
		x = world_pt[1];
		y = world_pt[0];
		z = -world_pt[2];
		//std::cout << x << ", " << y << "," << z << ", ";
		ray_dirs.at<double>(0, i) = x;
		ray_dirs.at<double>(1, i) = y;
		ray_dirs.at<double>(2, i) = z;
	}
	//std::cout << std::endl;
	return;
}

void undistortPoints(const std::string camera_model_file, const cv::Mat &pts, cv::Mat &undist_pts, cv::Mat &Rv) {
	// calculate ray direction
	cv::Mat undist_worldpts;
	undistortPoints(camera_model_file, pts, undist_worldpts);

	// define new coordinate
	cv::Mat ray0 = undist_worldpts.colRange(0, 1);
	cv::Mat ray1 = undist_worldpts.colRange(1, 2);
	cv::Mat axis_z = ray0;
	cv::Mat axis_x = ray1 / ray0.dot(ray1) - axis_z;
	axis_x /= cv::norm(axis_x);
	cv::Mat axis_y = axis_z.cross(axis_x);
	cv::vconcat(std::vector<cv::Mat>{ axis_x.t() , axis_y.t(), axis_z.t() }, Rv);
	undist_worldpts = Rv * undist_worldpts;

	undist_pts = cv::Mat::zeros(pts.size(), pts.type());
	for (int i = 0; i < 4; ++i) {
		double scale = undist_worldpts.at<double>(2, i);
		undist_pts.at<double>(i, 0) = undist_worldpts.at<double>(0, i) / scale;
		undist_pts.at<double>(i, 1) = undist_worldpts.at<double>(1, i) / scale;
	}
	return;
}