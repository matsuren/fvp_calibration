#pragma once
enum class CameraType { Perspective, Fisheye };
// GLOBAL
extern const std::string DATAFOLDER;
extern const std::string ORIGIN_TAG;
extern const int CAM_NUM;
extern const double SQUARE_SIZE; // size in m
// corners
extern const cv::Mat OBJ_PTS;
extern const CameraType cam_type;