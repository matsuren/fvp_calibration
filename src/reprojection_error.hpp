#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>


template <typename T>
Eigen::Matrix<T, 3, 3> RotationMatrix2D(T yaw_radians) {
	const T cos_yaw = ceres::cos(yaw_radians);
	const T sin_yaw = ceres::sin(yaw_radians);

	Eigen::Matrix<T, 3, 3> rotation;
	rotation << cos_yaw, -sin_yaw, T(0.0), sin_yaw, cos_yaw, T(0.0), T(0.0), T(0.0), T(1.0);
	return rotation;
}

template <typename T>
void ceres_world2cam(T point2D[2], const T point3D[3], const ocam_model *myocam_model)
{
	const double *invpol = myocam_model->invpol;
	T xc = T(myocam_model->xc);
	T yc = T(myocam_model->yc);
	double c = (myocam_model->c);
	double d = (myocam_model->d);
	double e = (myocam_model->e);
	int    width = (myocam_model->width);
	int    height = (myocam_model->height);
	int length_invpol = (myocam_model->length_invpol);
	T norm = ceres::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
	T theta = ceres::atan(point3D[2] / norm);
	T t, t_i;
	T rho, x, y;
	T invnorm;
	int i;

	if (norm != T(0.0))
	{
		invnorm = 1.0 / norm;
		t = theta;
		rho = T(invpol[0]);
		t_i = T(1);

		for (i = 1; i < length_invpol; i++)
		{
			t_i *= t;
			rho += t_i * invpol[i];
		}

		x = point3D[0] * invnorm*rho;
		y = point3D[1] * invnorm*rho;

		// OcamCalib coord to OpenCV coord
		point2D[1] = x * c + y * d + xc;
		point2D[0] = x * e + y + yc;
	}
	else
	{
		// OcamCalib coord to OpenCV coord
		point2D[1] = xc;
		point2D[0] = yc;
	}
}


struct PerspectiveReprojectionError {
	PerspectiveReprojectionError(const std::array<double, 8> observed, const std::string camera_model_file, const Eigen::Matrix<double, 3, 4> obj_pts_eigen)
		: observed_(observed), obj_pts_eigen_(obj_pts_eigen) {
		load_model(camera_model_file);
	}

	template <typename T>
	bool operator()(const T* const camera, const T* const params, T* residuals) const {

		// calculalte obj_pts location based on parameters
		Eigen::Matrix<T, 3, 1> tag_t(params[0], params[1], T(0.0));
		Eigen::Matrix<T, 3, 3> tag_rot = RotationMatrix2D(params[2]);
		Eigen::Matrix<T, 3, 4> transformed_pts = tag_rot * obj_pts_eigen_ + tag_t.replicate(1, 4);
		//std::cout << "trans:\n" << transformed_pts << std::endl;

		// project obj_pts on image
		Eigen::Matrix<T, 3, 3> cam_R;
		Eigen::Matrix<T, 3, 1> cam_t(camera[3], camera[4], camera[5]);
		ceres::AngleAxisToRotationMatrix(camera, &cam_R(0, 0));
		//std::cout << "cam_R:\n" << cam_R;

		Eigen::Matrix<T, 3, 4> cam_pts = cam_R * transformed_pts + cam_t.replicate(1, 4);
		//std::cout << "cam_pts:\n" << cam_pts;

		for (size_t i = 0; i < 4; i++)
		{
			T proj_pts[2];
			T x = cam_pts(0, i) / cam_pts(2, i);
			T y = cam_pts(1, i) / cam_pts(2, i);

			T r_sq = x * x + y * y;
			T factor = (1.0 + k1 * r_sq + k2 * r_sq*r_sq + k3 * r_sq*r_sq*r_sq);
			T x_ = x * factor + 2.0 * p1*x*y + p2 * (r_sq + 2.0 * x*x);
			T y_ = y * factor + p1 * (r_sq + 2.0 * y*y) + 2.0 * p2*x*y;
			proj_pts[0] = x_ * fx + cx;
			proj_pts[1] = y_ * fy + cy;

			//std::cout << "proj_pts:\n" << proj_pts[0] << "," << proj_pts[1] << std::endl;
			residuals[2 * i] = proj_pts[0] - observed_[2 * i];
			residuals[2 * i + 1] = proj_pts[1] - observed_[2 * i + 1];
		}
		//std::cout << "observed" << std::endl;
		//for (size_t i = 0; i < 8; i++)
		//	std::cout << observed_[i] << ", ";
		//std::cout << std::endl;
		//std::cout << "residuals" << std::endl;
		//for (size_t i = 0; i < 8; i++)
		//	std::cout << residuals[i] << ", ";
		//std::cout << std::endl;
		return true;
	}


	void load_model(const std::string camera_model_file) {

		cv::Mat K, D;
		cv::FileStorage fs_cam(camera_model_file, cv::FileStorage::READ);
		if (!fs_cam.isOpened())
			throw std::exception("Can't open calibration file!");
		fs_cam["K"] >> K;
		fs_cam["D"] >> D;
		if (K.empty() || D.empty())
			throw std::exception("Camera matrix or dist coeffs is empty!");
		fx = K.at<double>(0, 0);
		fy = K.at<double>(1, 1);
		cx = K.at<double>(0, 2);
		cy = K.at<double>(1, 2);
		k1 = D.at<double>(0, 0);
		k2 = D.at<double>(1, 0);
		p1 = D.at<double>(2, 0);
		p2 = D.at<double>(3, 0);
		k3 = D.at<double>(4, 0);
		return;
	}

	static ceres::CostFunction* Create(
		const std::array<double, 8> observed, const std::string camera_model_file, const Eigen::Matrix<double, 3, 4> obj_pts_eigen) {
		return (new ceres::AutoDiffCostFunction<PerspectiveReprojectionError, 8, 6, 3>(
			//return (new ceres::NumericDiffCostFunction<OcamReprojectionError, ceres::CENTRAL, 8, 6, 3>(
			new PerspectiveReprojectionError(observed, camera_model_file, obj_pts_eigen)));
	}

	const std::array<double, 8> observed_; // x: 2*i, y:2*i+1
	const Eigen::Matrix<double, 3, 4> obj_pts_eigen_;

	// camera matrix
	double fx, fy, cx, cy;
	double k1, k2, p1, p2, k3;
};

struct OcamReprojectionError {
	OcamReprojectionError(const std::array<double, 8> observed, const std::string camera_model_file, const Eigen::Matrix<double, 3, 4> obj_pts_eigen)
		: observed_(observed), fisheye_model_(load_model(camera_model_file)), obj_pts_eigen_(obj_pts_eigen) {}

	template <typename T>
	bool operator()(const T* const camera, const T* const params, T* residuals) const {

		// calculalte obj_pts location based on parameters
		Eigen::Matrix<T, 3, 1> tag_t(params[0], params[1], T(0.0));
		Eigen::Matrix<T, 3, 3> tag_rot = RotationMatrix2D(params[2]);
		Eigen::Matrix<T, 3, 4> transformed_pts = tag_rot * obj_pts_eigen_ + tag_t.replicate(1, 4);
		//std::cout << "trans:\n" << transformed_pts << std::endl;

		// project obj_pts on image
		Eigen::Matrix<T, 3, 3> cam_R;
		Eigen::Matrix<T, 3, 1> cam_t(camera[3], camera[4], camera[5]);
		ceres::AngleAxisToRotationMatrix(camera, &cam_R(0, 0));
		//std::cout << "cam_R:\n" << cam_R;

		Eigen::Matrix<T, 3, 4> cam_pts = cam_R * transformed_pts + cam_t.replicate(1, 4);
		//std::cout << "cam_pts:\n" << cam_pts;

		for (size_t i = 0; i < 4; i++)
		{
			T cam_ocam_pts[3], proj_pts[2];
			cam_ocam_pts[0] = cam_pts(1, i);
			cam_ocam_pts[1] = cam_pts(0, i);
			cam_ocam_pts[2] = -cam_pts(2, i);
			ceres_world2cam(proj_pts, cam_ocam_pts, &fisheye_model_);
			//std::cout << "proj_pts:\n" << proj_pts[0] << "," << proj_pts[1] << std::endl;
			residuals[2 * i] = proj_pts[0] - observed_[2 * i];
			residuals[2 * i + 1] = proj_pts[1] - observed_[2 * i + 1];
		}

		//std::cout << "observed" << std::endl;
		//for (size_t i = 0; i < 8; i++)
		//	std::cout << observed_[i] << ", ";
		//std::cout << std::endl;
		//std::cout << "residuals" << std::endl;
		//for (size_t i = 0; i < 8; i++)
		//	std::cout << residuals[i] << ", ";
		//std::cout << std::endl;
		return true;
	}


	ocam_model load_model(const std::string camera_model_file) {
		ocam_model fisheye_model;
		int ret = get_ocam_model(&fisheye_model, camera_model_file.c_str());
		if (ret == -1)
			throw std::exception("Can't open ocamcalib file!");
		return fisheye_model;
	}

	static ceres::CostFunction* Create(
		const std::array<double, 8> observed, const std::string camera_model_file, const Eigen::Matrix<double, 3, 4> obj_pts_eigen) {
		return (new ceres::AutoDiffCostFunction<OcamReprojectionError, 8, 6, 3>(
			//return (new ceres::NumericDiffCostFunction<OcamReprojectionError, ceres::CENTRAL, 8, 6, 3>(
			new OcamReprojectionError(observed, camera_model_file, obj_pts_eigen)));
	}

	const std::array<double, 8> observed_; // x: 2*i, y:2*i+1
	const ocam_model fisheye_model_;
	const Eigen::Matrix<double, 3, 4> obj_pts_eigen_;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};