/*
 * Copyright (C) 2021 Lara Kermarec <lara.git@kermarec.bzh> for CNRS
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <imgwarp_mls_rigid.h>
#include <opencv2/opencv.hpp>


/* From https://stackoverflow.com/a/52377891/5309963 */
class Subdiv2DIdx : public cv::Subdiv2D
{
	public :
		Subdiv2DIdx(cv::Rect rectangle);

		/* Source code of Subdiv2D:
		 * https://github.com/opencv/opencv/blob/master/modules/imgproc/src/subdivision2d.cpp#L762
		 * The implementation tweaks getTrianglesList() so that only the indice of the
		 * triangle inside the image are returned */
		void getTrianglesAndIndices(std::vector<cv::Vec6f> &triangles,
				std::vector<cv::Vec3i> &indices) const;
};

/* 1€ Filter, as described in: http://cristal.univ-lille.fr/~casiez/1euro/ and
 * implemented by taking inspiration from:
 * https://jaantollander.com/post/noise-filtering-using-one-euro-filter/ */
class OneEuroFilter
{
	public:
		OneEuroFilter(float beta, float fc);
		void update(float Te, const std::vector<cv::Point2f>& curPts,
				std::vector<cv::Point2f>& newPts);

		float beta;
		float fc;
	private:
		inline float smoothing_factor(const float Te, const float fc_d);
		inline cv::Point2f mix(const float alpha,
				const cv::Point2f prev, const cv::Point2f cur);

		std::vector<cv::Point2f> prevPos;
		std::vector<cv::Point2f> prevVel;
		float fc_d = 1.f;
};

struct Deformation {
	size_t group;
	size_t shapeIdx;
	cv::Vec3i triangle;
	cv::Vec3f weights;
};

enum LogLevel {
	LOG_ERROR,
	LOG_WARNING,
	LOG_DEBUG,
};


void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color);
void match_points(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& ref,
		std::vector<cv::Point2f>& dst);
void get_deformation_groups(const std::vector<Deformation>& def,
		const std::vector<cv::Point2f>& src, const float alpha,
		std::vector<std::vector<cv::Point2f>>& srcGroups,
		std::vector<std::vector<cv::Point2f>>& dstGroups);
bool compute_MLS_on_ROI(cv::Mat& frame, ImgWarp_MLS_Rigid& mls,
		std::vector<cv::Point2f> src, std::vector<cv::Point2f> dst);
void find_faces(const cv::Mat& frame, dlib::frontal_face_detector& detector,
		const double confidenceThresh, std::vector<dlib::rectangle>& detections,
		cv::Rect& probableROI);
void find_landmarks(const cv::Mat& frame, dlib::shape_predictor& sp,
		std::vector<dlib::rectangle>& dets, std::vector<std::vector<cv::Point2f>>& facesPts);
void draw_debug(cv::Mat& frame, const std::vector<std::vector<cv::Point2f>>& srcGroups,
		const std::vector<std::vector<cv::Point2f>>& dstGroups);
void perf_start();
void perf_split(std::string msg);
double perf_print();
bool import_deformations(const char* file, std::vector<Deformation>& deformations);
void log(LogLevel lvl, const char* file, const char* func,
		int line, const char* str, ...);


#ifdef USE_GST_LOGGING
	#include <gst/gst.h>
	void init_mozza(GstDebugCategory* cat);
#endif


#define ERROR(...) log(LOG_ERROR, __FILE__, __func__, __LINE__, __VA_ARGS__)
#define WARN(...)  log(LOG_WARNING, __FILE__, __func__, __LINE__, __VA_ARGS__)
#define DEBUG(...) log(LOG_DEBUG, __FILE__, __func__, __LINE__, __VA_ARGS__)


const int MLS_GRID_SIZE = 5;
