#pragma once

#include <iostream>
#include <string>
#include <map>
#include <opencv2/imgproc.hpp>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
	//#include "tagCircle21h7.h"
}

class DetectTag
{
public:
	DetectTag();
	~DetectTag();
	int detect(const cv::Mat &src, std::map<int, cv::Mat> &tagid_pts,  cv::Mat &draw_tag);

private:
	apriltag_family_t *tf;
	apriltag_detector_t *td;
};

DetectTag::DetectTag()
{
	// Initialize tag detector with options
	tf = tag36h11_create(); // tagCircle21h7_create()
	td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);
	td->quad_decimate = 1;
	td->quad_sigma = 0.0;
	td->nthreads = 1;
	td->debug = 0;
	td->refine_edges = 0;
}

DetectTag::~DetectTag()
{
	apriltag_detector_destroy(td); //tagCircle21h7_destroy(tf);
	tag36h11_destroy(tf);
}

int DetectTag::detect(const cv::Mat &src, std::map<int, cv::Mat> &tagid_pts, cv::Mat &draw_tag) {
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	draw_tag = src.clone();
	// Make an image_u8_t header for the Mat data
	image_u8_t im = { gray.cols, // .width
		 gray.rows, // .height 
		 gray.cols, // .stride
		 gray.data  // .buf 
	};

	zarray_t *detections = apriltag_detector_detect(td, &im);
	std::cout << zarray_size(detections) << " tags detected" << std::endl;

	// Draw detection outlines
	for (int i = 0; i < zarray_size(detections); i++) {
		constexpr int thickness = 1;
		apriltag_detection_t *det;
		zarray_get(detections, i, &det);
		printf("detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
			i, det->family->nbits, det->family->h, det->id, det->hamming, det->decision_margin);
		cv::line(draw_tag, cv::Point(det->p[0][0], det->p[0][1]),
			cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 0xff, 0), thickness);
			
		cv::line(draw_tag, cv::Point(det->p[0][0], det->p[0][1]),
			cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 0, 0xff), thickness);
			
		cv::line(draw_tag, cv::Point(det->p[1][0], det->p[1][1]),
			cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0xff, 0, 0), thickness);
			
		cv::line(draw_tag, cv::Point(det->p[2][0], det->p[2][1]),
			cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0xff, 0xff, 0), thickness);

		std::stringstream ss;
		ss << det->id;
		cv::String text = ss.str();
		int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontscale = 1.0;
		int baseline;
		cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2, &baseline);
		cv::putText(draw_tag, text, cv::Point(det->c[0] - textsize.width / 2,
			det->c[1] + textsize.height / 2), fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);

		// convert to cv::Mat
		cv::Mat cv_pts(4, 2, CV_64F, det->p);
		tagid_pts[det->id] = cv_pts.clone();
	}
	apriltag_detections_destroy(detections);
	return 0;
}
//