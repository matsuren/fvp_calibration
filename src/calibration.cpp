#include <iostream>
#include <string>
#include <map>
#include <queue>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "calibration.hpp"
#include "detect_tag.hpp"
#include "undistort_points.hpp"


#include "angle_adjuster.hpp"

//
#include "optimizer.hpp"

// GLOBAL
const std::string DATAFOLDER = "../data";
const std::string ORIGIN_TAG = "tag0";
const int CAM_NUM = 4;
const double SQUARE_SIZE = 0.159; // size in m
// corners
const cv::Mat OBJ_PTS = (cv::Mat_<double>(4, 3) <<
	0, 0, 0,
	SQUARE_SIZE, 0, 0,
	SQUARE_SIZE, SQUARE_SIZE, 0,
	0, SQUARE_SIZE, 0);

//const CameraType cam_type = CameraType::Perspective; // CameraType::Fisheye; //
const CameraType cam_type = CameraType::Fisheye; // CameraType::Fisheye; //


/////////////////////////////////////////////////////////////////////////////
int detectCorner()
{
	DetectTag detector;

	std::vector<cv::Mat> frames;
	for (size_t i = 0; i < CAM_NUM; ++i) {
		cv::Mat tmp = cv::imread(DATAFOLDER + "/img" + std::to_string(i) + ".jpg");
		if (tmp.empty())
			throw std::exception("Image is empty!");
		frames.push_back(tmp);
	}

	cv::FileStorage fs(DATAFOLDER + "/tag_points.yml", cv::FileStorage::WRITE);
	for (size_t img_id = 0; img_id < frames.size(); ++img_id) {
		std::map<int, cv::Mat> tagid_pts;
		cv::Mat draw_tag;
		detector.detect(frames.at(img_id), tagid_pts, draw_tag);

		for (const auto &it : tagid_pts) {
			std::stringstream ss_ptid;
			ss_ptid << "img" << img_id;
			ss_ptid << "tag" << it.first;
			fs << ss_ptid.str() << it.second;
		}
		cv::imwrite(DATAFOLDER + "/detected_img" + std::to_string(img_id) + ".jpg", draw_tag);
	}
	return 0;
}

// BFS for estimating pose
cv::Mat BFSEstimatePoses(
	const std::string tag, std::map<std::string, cv::Mat>& poses)
{
	// calculate gragh
	static std::map<std::string, std::vector<std::string>> tag_dirs;
	if (tag_dirs.size()==0)
	{
		for (auto iter = poses.begin(); iter != poses.end(); ++iter)
		{
			std::string key = iter->first;
			const auto from_key = key.substr(4, 4);
			const auto to_key = key.substr(0, 4);
			if (tag_dirs.count(from_key) == 0) {
				tag_dirs[from_key] = { to_key };
			}
			else {
				tag_dirs[from_key].push_back(to_key);
			}
		}
	}


	// check keys
	if (poses.find(tag) != poses.end()) {
		return poses[tag];
	}

	// if can't find value, calculate using BFS
	const auto from_tag = tag.substr(4, 4);
	const auto to_tag = tag.substr(0, 4);

    // insert first elements
	std::queue<std::vector<std::string>> que;
	que.push({ from_tag});
	
	std::vector<std::string> tag_list;
	while (!que.empty())
	{
		tag_list = que.front();
		que.pop();
		// find answer
		if (tag_list.back() == to_tag)
			break;

		for (const auto& next_tag : tag_dirs[tag_list.back()]) {
			if (std::count(tag_list.begin(), tag_list.end(), next_tag) == 0){
				auto new_tag_list = tag_list;
				new_tag_list.push_back(next_tag);
				que.push(new_tag_list);
			}
		}
	}
	cv::Mat pose = cv::Mat::eye(cv::Size(4, 4), CV_64F);;
	for (size_t i = 0; i < tag_list.size() - 1; i++)
	{
		auto tmp_key = tag_list[i + 1] + tag_list[i];
		//std::cout << tmp_key << std::endl;
		pose = poses[tmp_key]* pose;
	}
	return pose;
}

/////////////////////////////////////////////////////////////////////////////
int calculatePoses() {
	cv::FileStorage fs(DATAFOLDER + "/tag_points.yml", cv::FileStorage::READ);
	if(!fs.isOpened())
		throw std::exception("Cannot open file: tag_points.yml!");
	// calculate camera poses based on origin_tag, and save
	std::map<std::string, cv::Mat> poses;
	const auto fn = fs.root();
	for (auto &fit = fn.begin(); fit != fn.end(); ++fit)
	{
		cv::FileNode item = *fit;
		std::string key = item.name();
		std::string cam_id_str = key.substr(3, 1);
		cv::Mat pts, undist_pts;
		item >> pts;
		cv::Mat R_tmp = cv::Mat::eye(cv::Size(3, 3), CV_64F);

		// Undistort points
		if (cam_type == CameraType::Perspective) {
			cv::Mat K, D;
			cv::FileStorage fs_cam(DATAFOLDER + "/calib_perspective_results_" + cam_id_str + ".yml", cv::FileStorage::READ);
			fs_cam["K"] >> K;
			fs_cam["D"] >> D;
			if (K.empty() || D.empty())
				throw std::exception("Camera matrix or dist coeffs is empty!");
			cv::undistortPoints(pts, undist_pts, K, D);
		}
		else if (cam_type == CameraType::Fisheye) {
			std::string fname = DATAFOLDER + "/calib_results_" + cam_id_str + ".txt";
			undistortPoints(fname, pts, undist_pts, R_tmp);
		}
		else {
			throw std::exception("NO camera type");
		}

		cv::Mat rvec, tvec, R;
		cv::solvePnP(OBJ_PTS, undist_pts, cv::Mat::eye(cv::Size(3, 3), CV_64F),
			cv::Mat::zeros(cv::Size(5, 1), CV_64F), rvec, tvec, false, cv::SOLVEPNP_IPPE);

		// camera pose
		cv::Mat pose = cv::Mat::eye(cv::Size(4, 4), CV_64F);
		cv::Rodrigues(rvec, R);
		R = R_tmp.t() * R;
		tvec = R_tmp.t() * tvec;
		R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
		tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));

		poses[key] = pose;
		const auto inv_key = key.substr(4, 4) + key.substr(0, 4);
		poses[inv_key] = pose.inv();
	}


	// calculate camera and tag poses based on origin_tag, and save
	cv::FileStorage fs_out(DATAFOLDER + "/initial_camera_poses.yml", cv::FileStorage::WRITE);
	for (size_t i = 0; i < CAM_NUM; ++i)
	{
		for (auto &it : std::vector<std::string>{ "img", "tag" }) {
			if (it == "tag" && i == std::stoi(ORIGIN_TAG.substr(3, 1)))
				continue;
			const std::string key = ORIGIN_TAG + it + std::to_string(i);
			fs_out << key << BFSEstimatePoses(key, poses);
		}
	}
	std::cout << "Done estimating to initial camera poses" << std::endl;
	return 0;
}

/////////////////////////////////////////////////////////////////////////////
int refinePosesInitialParams() {
	/////////////////////////////////////////////
	// estimate initial parameter for refinement

	// load initial poses
	std::map<std::string, cv::Mat> poses;
	cv::FileStorage fs_pose(DATAFOLDER + "/initial_camera_poses.yml", cv::FileStorage::READ);
	if (!fs_pose.isOpened())
		throw std::exception("Cannot open file: initial_camera_poses.yml!");
	for (size_t i = 0; i < CAM_NUM; ++i)
	{
		const std::string key = ORIGIN_TAG + "img" + std::to_string(i);
		cv::Mat pose;
		fs_pose[key] >> pose;
		poses[key] = pose;
		const auto inv_key = key.substr(4, 4) + key.substr(0, 4);
		poses[inv_key] = pose.inv();
	}
	// load points and calculate ray directions
	cv::FileStorage fs_pts(DATAFOLDER + "/tag_points.yml", cv::FileStorage::READ);
	if (!fs_pts.isOpened())
		throw std::exception("Cannot open file: tag_points.yml!");
	//cv::FileStorage fs_rays(DATAFOLDER + "/ray_directions.yml", cv::FileStorage::WRITE);
	cv::FileStorage fs_params(DATAFOLDER + "/initial_params.yml", cv::FileStorage::WRITE);
	for (auto &fit = fs_pts.root().begin(); fit != fs_pts.root().end(); ++fit)
	{
		cv::FileNode item = *fit;
		std::string key = item.name();
		std::string cam_id_str = key.substr(3, 1);
		std::string tag_str = key.substr(4, 4);

		cv::Mat pts, ray_dirs;
		fs_pts[key] >> pts;

		// Undistort points
		if (cam_type == CameraType::Perspective) {
			cv::Mat K, D;
			cv::FileStorage fs_cam(DATAFOLDER + "/calib_perspective_results_" + cam_id_str + ".yml", cv::FileStorage::READ);
			fs_cam["K"] >> K;
			fs_cam["D"] >> D;
			if (K.empty() || D.empty())
				throw std::exception("Camera matrix or dist coeffs is empty!");
			cv::undistortPoints(pts, ray_dirs, K, D);
			std::vector<cv::Mat> split_mat;
			cv::split(ray_dirs, split_mat);
			split_mat.push_back(cv::Mat::ones(cv::Size(1, 4), CV_64F));
			cv::hconcat(split_mat, ray_dirs);
			ray_dirs = ray_dirs.t();
		}
		else if (cam_type == CameraType::Fisheye) {
			std::string fname = DATAFOLDER + "/calib_results_" + cam_id_str + ".txt";
			undistortPoints(fname, pts, ray_dirs);
		}
		else {
			throw std::exception("NO camera type");
		}
		// save ray directions (tmp)
		//fs_rays << key << ray_dirs;

		// camera coord to world (ORIGIN_TAG) coord
		cv::Mat pose = poses[ORIGIN_TAG + key.substr(0, 4)];
		ray_dirs = pose(cv::Rect(0, 0, 3, 3))*ray_dirs;
		// project onto xy plane
		cv::Mat factor = -pose.at<double>(2, 3) / ray_dirs.row(2);
		factor = cv::repeat(factor, 3, 1);
		cv::Mat tvec = pose(cv::Rect(3, 0, 1, 3));
		cv::Mat proj_pt = cv::repeat(tvec, 1, 4) + factor.mul(ray_dirs);

		// estimate initial parameter
		// offset x,y: location pt0
		// angle: angle[rad] of vector(pt1 - pt0)
		double offset_x, offset_y, angle;
		offset_x = proj_pt.at<double>(0, 0);
		offset_y = proj_pt.at<double>(1, 0);
		angle = std::atan2(
			proj_pt.at<double>(1, 1) - proj_pt.at<double>(1, 0),
			proj_pt.at<double>(0, 1) - proj_pt.at<double>(0, 0));
		//std::cout << angle << "," << std::cos(angle) << "," << std::sin(angle) << std::endl;
		fs_params << tag_str << "[" << offset_x << offset_y << angle << "]";
	}
	return 0;
}

void optimization() {
	// Optimization by Ceres solver
	//google::InitGoogleLogging(argv[0]);
	Optimizer optimizer(DATAFOLDER, OBJ_PTS, ORIGIN_TAG, CAM_NUM, cam_type);
	optimizer.optimize();
	optimizer.saveResults();
}

void adjustAngle() {
	// Adjust angle
	AngleAdjuster adjuster(DATAFOLDER, CAM_NUM);
	cv::Mat out;
	adjuster.generateBEV(out);
	adjuster.rotate(out);
	adjuster.savePoses();
	adjuster.generateBEV(out);
	cv::imshow("final image", out);
	cv::waitKey();
	cv::destroyAllWindows();
}

int main(int argc, char *argv[]) {

	try {
		// detect Apriltag corner
		std::cout << "---detectCorner---" << std::endl;
		detectCorner();

		// calculate camera poses based on the detected corners
		std::cout << "---calculatePoses---" << std::endl;
		calculatePoses();

		// refine camera poses by LM method
		std::cout << "---refinePoses---" << std::endl;
		refinePosesInitialParams();
		optimization();

		// adjust angle
		std::cout << "---adjustAngle---" << std::endl;
		adjustAngle();

	}
	catch (const std::exception& ex) {
		std::cerr << "!!!!!!! --- error --- !!!!!!!" << std::endl;
		std::cerr << ex.what() << std::endl;
	}
	return 0;

}
