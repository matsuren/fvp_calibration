#include <iostream>
#include <string>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "angle_adjuster.hpp"
#include "ocam_functions.hpp"


AngleAdjuster::AngleAdjuster(const std::string data_folder, const int camera_num, const std::string & load_yml)
	:DATAFOLDER(data_folder), CAM_NUM(camera_num)
{

	cv::FileStorage fs_pose(DATAFOLDER + "/" + load_yml, cv::FileStorage::READ);
	if (!fs_pose.isOpened()) {
		std::string msg = "Cannot open file: " + load_yml;
		throw std::exception(msg.c_str());
	}
	// centering cameras
	cv::Mat centers = cv::Mat::zeros(3, 1, CV_64F);
	for (size_t i = 0; i < CAM_NUM; ++i) {
		const std::string key = "originimg" + std::to_string(i);
		cv::Mat pose;
		fs_pose[key] >> pose;
		centers += pose(cv::Rect(3, 0, 1, 3));
	}
	centers /= CAM_NUM;
	// adjust center andd
	for (size_t i = 0; i < CAM_NUM; ++i) {
		const std::string key = "originimg" + std::to_string(i);
		cv::Mat pose;
		fs_pose[key] >> pose;
		// adjust only for x,y coord
		pose(cv::Rect(3, 0, 1, 2)) -= centers(cv::Rect(0, 0, 1, 2));
		poses_wc.push_back(pose);
		poses_cw.push_back(pose.inv());
	}
	// calculate pixel_to_m
	// if image_cam_dist_ratio is 4.0, 
	// bird's-eye view image size is four times bigger than distance between cameras
	constexpr double image_cam_dist_ratio = 10.0;
	cv::Mat cam_xy;
	double cam_xy_min, cam_xy_max;
	double cam_dist;
	for (const auto &it : poses_wc) {
		if (cam_xy.empty())
			cam_xy = it(cv::Rect(3, 0, 1, 2));
		else
			cv::hconcat(cam_xy, it(cv::Rect(3, 0, 1, 2)), cam_xy);
	}
	cv::minMaxLoc(cam_xy.row(0), &cam_xy_min, &cam_xy_max);
	cam_dist = cam_xy_max - cam_xy_min;
	cv::minMaxLoc(cam_xy.row(0), &cam_xy_min, &cam_xy_max);
	cam_dist = std::max(cam_dist, cam_xy_max - cam_xy_min);
	pixel_to_m = image_cam_dist_ratio * cam_dist / std::min(WIDTH, HEIGHT);

	// load calibration file
	for (size_t i = 0; i < CAM_NUM; ++i) {
		cv::Mat tmp = cv::imread(DATAFOLDER + "/img" + std::to_string(i) + ".jpg");
		if (tmp.empty())
			throw std::exception("Image is empty!");
		frames.push_back(tmp);

		std::string fname = DATAFOLDER + "/calib_results_" + std::to_string(i) + ".txt";
		ocam_model fisheye_model;
		int ret = get_ocam_model(&fisheye_model, fname.c_str());
		if (ret == -1)
			throw std::exception("Can't open ocamcalib file!");
		fisheye_models.push_back(fisheye_model);
	}
}

int AngleAdjuster::getClosestCamera(const cv::Mat &pt, const std::vector<cv::Mat> &poses_wc) {
	std::vector<double> dist;
	for (size_t i = 0; i < poses_wc.size(); ++i) {
		cv::Mat cam_pos = poses_wc[i](cv::Rect(3, 0, 1, 3));
		dist.push_back(cv::norm(cam_pos - pt.rowRange(0, 3)));
	}
	int arg_min = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
	return arg_min;
}

void AngleAdjuster::generateBEV(cv::Mat & bev) {
	// bird's-eye view
	bev = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	double uc = WIDTH / 2.0;
	double vc = HEIGHT / 2.0;
	double M[3];
	double m[2];
	std::cout << "Generating bird's-eye view,,," << std::endl;
	cv::Mat world_pt = (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 1);
	for (int v = 0; v < HEIGHT; v++) {
		for (int u = 0; u < WIDTH; u++)
		{
			world_pt.at<double>(0) = (u - uc)*pixel_to_m;
			world_pt.at<double>(1) = (-v + vc )*pixel_to_m;
			world_pt.at<double>(2) = 0.0;
			world_pt.at<double>(3) = 1.0;

			// get the closest camera
			int cam_id = getClosestCamera(world_pt, poses_wc);
			cv::Mat src = frames[cam_id];
			ocam_model fisheye_model = fisheye_models[cam_id];
			cv::Mat pose = poses_cw[cam_id];
			// world coord to camera coord # TODO use Eigen here (OpenCV multiply is too slow)
			world_pt = pose * world_pt;
			//world_pt = world_pt / world_pt.at<double>(3, 0);
			M[0] = world_pt.at<double>(1, 0);
			M[1] = world_pt.at<double>(0, 0);
			M[2] = -world_pt.at<double>(2, 0);
			world2cam(m, M, &fisheye_model);

			int u_int = int(m[1]);
			int v_int = int(m[0]);
			if (0 <= u_int && u_int < src.cols && 0 <= v_int && v_int < src.rows)
				bev.at<cv::Vec3b>(v, u) = src.at<cv::Vec3b>(v_int, u_int);
		}
	}
	auto fs = cv::FileStorage(DATAFOLDER + "/bev_info.yml", cv::FileStorage::WRITE);
	fs << "pixel_to_m" << pixel_to_m;
	cv::imwrite(DATAFOLDER + "/bev.jpg", bev);
}

void AngleAdjuster::rotate(cv::Mat & img)
{
	std::string winname = "Rotate images";
	cv::Point2f center(img.cols/2, img.rows / 2);
	std::cout << "Please rotate image using 'a', 's'" << std::endl;
	std::cout << "After finishing rotating image, press ESC or q:" << std::endl;
	while (true)
	{
		cv::Mat dst;
		double angle = current_angle - ANGLE_MAX / 2;
		cv::Mat R = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::warpAffine(img, dst, R, img.size());
		cv::imshow(winname, dst);
		int k = cv::waitKey(10);
		if (k == 'a')
			current_angle++;
		if (k == 's')
			current_angle--;
		if (k == 27 || k=='q')
			break;
	}
	cv::destroyAllWindows();

	// Adjust camera position
	std::vector<cv::Mat> new_poses_wc;
	cv::Mat T = cv::Mat::eye(cv::Size(4, 4), CV_64F);
	double angle = current_angle - ANGLE_MAX / 2;
	// Rotation center is (0,0) 
	cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0, 0), -angle, 1.0);
	R.copyTo(T(cv::Rect(0, 0, 3, 2)));
	for (int i = 0; i < poses_wc.size(); i++) {
		cv::Mat pose = poses_wc[i];
		new_poses_wc.push_back(T*pose);
	}
	// update poses
	poses_wc = new_poses_wc;
	poses_cw.clear();
	for (const auto &it : poses_wc) {
		poses_cw.push_back(it.inv());
	}


}

void AngleAdjuster::savePoses(const std::string save_yml)
{
	cv::FileStorage fs_pose(DATAFOLDER + "/" + save_yml, cv::FileStorage::WRITE);
	for (size_t i = 0; i < CAM_NUM; ++i) {
		const std::string key = "originimg" + std::to_string(i);
		cv::Mat pose = poses_wc[i];
		fs_pose << key << pose;
		const auto inv_key = "img" + std::to_string(i) + "origin";
		fs_pose << inv_key << pose.inv();
	}
}

//int main(int argc, char *argv[]) {
//	////////////////////////////
//	// Adjust angle
//	AngleAdjuster adjuster("../data", 4);
//	cv::Mat out;
//	adjuster.generateBEV(out);
//	adjuster.rotate(out);
//	adjuster.savePoses();
//	adjuster.generateBEV(out);
//	cv::imshow("final image", out);
//	cv::waitKey();
//	cv::destroyAllWindows();
//
//	return 0;
//}
