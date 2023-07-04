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

#include <dlib/opencv.h>

#include "mozza.h"


struct TimingMsg {
	std::string msg;
	double split;
};

struct PerfCounter {
	cv::TickMeter meter;
	std::vector<TimingMsg> splits;
};


static PerfCounter perf;


#ifdef USE_GST_LOGGING
	#include <gst/gst.h>
	static GstDebugCategory* gstCat = NULL;

	void init_mozza(GstDebugCategory* cat)
	{
		gstCat = cat;
	}
#endif

/* Log using Gstreamer's own logging system, or if not compiled with gstreamer
 * support, fall back to writing to stderr */
void log(LogLevel lvl, const char* file, const char* func,
		int line, const char* str, ...)
{
	va_list ap;
#ifdef USE_GST_LOGGING
	GstDebugLevel gstLvl = GST_LEVEL_DEBUG;
	switch (lvl) {
	case LOG_ERROR: gstLvl = GST_LEVEL_ERROR; break;
	case LOG_WARNING: gstLvl = GST_LEVEL_WARNING; break;
	case LOG_DEBUG: gstLvl = GST_LEVEL_DEBUG; break;
	}

	if (gstCat) {
		va_start(ap, str);
		gst_debug_log_valist(gstCat, gstLvl, file, func, line, NULL, str, ap);
		va_end(ap);
	}
#else
	switch (lvl) {
	case LOG_ERROR: fprintf(stderr, "[ERROR]: "); break;
	case LOG_WARNING: fprintf(stderr, "[WARNING]: "); break;
	case LOG_DEBUG: fprintf(stderr, "[DEBUG]: "); break;
	}

	fprintf(stderr, "%s:%d:%s() ", file, line, func);

	va_start(ap, str);
	vfprintf(stderr, str, ap);
	va_end(ap);
	fprintf(stderr, "\n");
#endif
}


/* Draw delaunay triangles, removing the triangles that go out of frame */
void draw_delaunay( cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color )
{
	std::vector<cv::Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<cv::Point> pt(3);
	cv::Size size = img.size();
	cv::Rect rect(0,0, size.width, size.height);

	for( size_t i = 0; i < triangleList.size(); i++ ) {
		cv::Vec6f t = triangleList[i];
		pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));
		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			cv::line(img, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
			cv::line(img, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
			cv::line(img, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
		}
	}
}


/* Align a set of facial landmarks to another, so they can be compared
 * regardless of head position
 *
 * `src`: set of points that will be aligned
 * `ref`: set of points used as reference for the alignment
 * `dst`: output. `src` points aligned in `ref` frame of reference */
void match_points(std::vector<cv::Point2f>& src,
		std::vector<cv::Point2f>& ref,
		std::vector<cv::Point2f>& dst)
{
	if (src.size() < 68 || ref.size() < 68) {
		DEBUG("MatchPoints failure: not enough points\n");
		return;
	}

	// We only use the face contours, eybrows, and nose to find the homography
	std::vector<cv::Point2f> homographySrc(src.begin(), src.begin() + 36);
	std::vector<cv::Point2f> homographyRef(ref.begin(), ref.begin() + 36);

	cv::Mat H = cv::findHomography(homographySrc, homographyRef, 0);  //0 => Least Squares
	cv::perspectiveTransform(src, dst, H);
}


/* Get groups of control and deformed points from the list of deformations and
 * a set of facial landmarks
 *
 * `def`: list of deformations
 * `src`: facial landmarks
 * `alpha`: deformation multiplicator
 * `srcGroups`: sets of control points
 * `dstGroups`: corresponding sets of deformed points */
void get_deformation_groups(const std::vector<Deformation>& def,
		const std::vector<cv::Point2f>& src, const float alpha,
		std::vector<std::vector<cv::Point2f>>& srcGroups,
		std::vector<std::vector<cv::Point2f>>& dstGroups)
{
	srcGroups.clear();
	dstGroups.clear();

	for (Deformation d : def) {
		/* Compute deformed point from barycentric coordinates */
		cv::Point2f v1 = src[d.triangle[0]];
		cv::Point2f v2 = src[d.triangle[1]];
		cv::Point2f v3 = src[d.triangle[2]];
		cv::Vec3f w = d.weights;
		cv::Point2f target = v1 * w[0] + v2 * w[1] + v3 * w[2];

		/* Compute the relative deformation to be able to scale it */
		cv::Point2f vec = target - src[d.shapeIdx];

		/* FIXME For now, there is no guarantee that imported group IDs are
		 * sequential and start at 0 */
		if (d.group + 1 > srcGroups.size()) {
			srcGroups.resize(d.group + 1);
			dstGroups.resize(d.group + 1);
		}

		srcGroups[d.group].push_back(src[d.shapeIdx]);
		dstGroups[d.group].push_back(src[d.shapeIdx] + vec * alpha);
	}
}


/* Apply an MLS Rigid deformation on `frame` using respectively the control and
 * target points, `src` and `dst`. The ROI is computed from the boudning box of
 * the points used*/
bool compute_MLS_on_ROI(cv::Mat& frame,
		ImgWarp_MLS_Rigid& mls,
		std::vector<cv::Point2f> src,
		std::vector<cv::Point2f> dst)
{
	cv::Rect roi;
	cv::Rect roiMin;

	if (src.size() != dst.size()) {
		DEBUG("compute_MLS_on_ROI: src and dst vectors differ in size\n");
		return false;
	}

	if (src.empty()) {  /* Nothing to be done */
		return false;
	}

	roi = cv::boundingRect(src) | cv::boundingRect(dst);
	roi -= cv::Point(roi.width, roi.height) / 2;
	roi += cv::Size(roi.width, roi.height);

	/* Make sure the ROI is big enough to apply the MLS */
	roiMin = cv::Rect(roi.x + roi.width / 2 - MLS_GRID_SIZE,
			roi.y + roi.height / 2 - MLS_GRID_SIZE,
			2 * MLS_GRID_SIZE + 1, 2 * MLS_GRID_SIZE + 1);
	roi |= roiMin;

	/* Make sure no part of the ROI is outside the image */
	cv::Rect full(0, 0, frame.cols, frame.rows);
	roi &= full;

	if (roi.empty()) {
		DEBUG("MLS ROI empty");
		return false;
	}

	for (size_t i = 0; i < src.size(); i++) {
		src[i] -= cv::Point2f(roi.tl());
		dst[i] -= cv::Point2f(roi.tl());
	}

	// Add fixed control points all around the BBox to prevent a discontinuity
	int step = MLS_GRID_SIZE * 2;
	for (int x = 0; x < roi.width; x += step) {
		src.push_back(cv::Point2f(x, 0));
		dst.push_back(src.back());
		src.push_back(cv::Point2f(x, roi.height));
		dst.push_back(src.back());
	}
	for (int y = step; y < roi.height - step; y += step) {
		src.push_back(cv::Point2f(0, y));
		dst.push_back(src.back());
		src.push_back(cv::Point2f(roi.width, y));
		dst.push_back(src.back());
	}

	cv::Mat frameROI = mls.setAllAndGenerate(frame(roi), src, dst, roi.width, roi.height);
	frameROI.copyTo(frame(roi));

	return true;
}


/* Efficiently find faces in an image
 *
 * `frame`: the image to search
 * `detector`: dlib's face detector
 * `confidenceThresh`: face detector confidence threshold
 * `detections`: bounding boxes of the faces identified by dlib
 * `resizeRate`: optimimal resizeRate to find a face in the image in minimal
 *               time. The value is updated when the function returns. When
 *               in doubt, use 1.f as a default */
void find_faces(const cv::Mat& frame, dlib::frontal_face_detector& detector,
		const double confidenceThresh, std::vector<dlib::rectangle>& detections,
		cv::Rect& probableROI)
{
	const float HOG_TRAIN_SIZE = 80.f;
	float resizeRate = 1.f;
	cv::Mat cropped, resized;
	cv::Rect fullFrameRect(0, 0, frame.cols, frame.rows);
	dlib::cv_image<dlib::bgr_pixel> dImg;

	/* First try to detect faces using the provided resizeRate (most likely computed during
	 * the previous run) */
	probableROI &= fullFrameRect;  /* ROI is cropped to fit inside the image size */
	if (!probableROI.empty() && probableROI.size().width > 0 && probableROI.size().height > 0) {
		cropped = frame(probableROI);
		resizeRate = std::floor(std::min(probableROI.width, probableROI.height) / (HOG_TRAIN_SIZE * 1.2f));
		if (resizeRate > 1.f)
			cv::resize(cropped, resized, cv::Size(cropped.cols / resizeRate, cropped.rows / resizeRate));
		else
			resized = cropped;
		dImg = dlib::cv_image<dlib::bgr_pixel>(resized);
		//dImg = dlib::cv_image<dlib::bgr_pixel>(cropped);
		detections = detector(dImg, confidenceThresh);
	}

	/* Then if we get nothing, fall back on doing detection on the full image */
	/* TODO Maybe set a minimum resizeRate > 1 */
	if (detections.empty()) {
		dImg = dlib::cv_image<dlib::bgr_pixel>(frame);
		detections = detector(dImg, confidenceThresh);
		probableROI = fullFrameRect;
		resizeRate = 1.f;
	}

	/* Rescale the detections back to full frame size */
	for (dlib::rectangle& r : detections)
		//r = dlib::translate_rect(r, probableROI.x, probableROI.y);
		r = dlib::translate_rect(dlib::scale_rect(r, resizeRate), probableROI.x, probableROI.y);

	/* TODO Use a better metric than simply taking the first detection */
	if (!detections.empty()) {
		const dlib::rectangle& r = detections[0];
		cv::Size size(r.width(), r.height());
		cv::Point tl(r.left(), r.top());
		cv::Rect newROI(tl, size);
		newROI += size / 4;
		newROI -= cv::Point(size / 8);
		probableROI |= newROI;
		if (probableROI.area() > 3 * newROI.area())
			probableROI = newROI;
	}
}


/* Find facial landmarks inside the bounding boxes provided
 *
 * `frame`: image to search
 * `sp`: dlib's facial shape predictor
 * `dets`: face detection bounding boxes, usually provided by `find_faces()`
 * `facesPts`: sets of facial landmarks for each bounding box provided*/
void find_landmarks(const cv::Mat& frame, dlib::shape_predictor& sp,
		std::vector<dlib::rectangle>& dets, std::vector<std::vector<cv::Point2f>>& facesPts)
{
	dlib::full_object_detection shape;
	dlib::cv_image<dlib::bgr_pixel> dImg(frame);

	facesPts.resize(dets.size());

	for (size_t i = 0; i < dets.size(); i++) {
		shape = sp(dImg, dets[i]);

		facesPts[i].resize(shape.num_parts());

		for (size_t j = 0; j < shape.num_parts(); j++)
			facesPts[i][j] = cv::Point2f(shape.part(j).x(), shape.part(j).y());
	}
}


void draw_debug(cv::Mat& frame, const std::vector<std::vector<cv::Point2f>>& srcGroups,
		const std::vector<std::vector<cv::Point2f>>& dstGroups)
{
	for (size_t i = 0; i < srcGroups.size(); i++) {
		for (size_t j = 0; j < srcGroups[i].size(); j++) {
			cv::circle(frame, srcGroups[i][j], 2, cv::Scalar(0, 0, 255));
			cv::line(frame, srcGroups[i][j], dstGroups[i][j], cv::Scalar(0, 255, 0));
		}
	}
}


void perf_start()
{
	perf.meter.reset();
	perf.meter.start();
}

void perf_split(std::string msg)
{
	perf.meter.stop();
	perf.splits.push_back({msg, perf.meter.getTimeMilli()});
	perf.meter.start();
}

double perf_print()
{
	perf.meter.stop();
	double total = perf.meter.getTimeMilli();
	double acc = 0.0;
	fprintf(stderr, "Time: ");
	for (TimingMsg m : perf.splits) {
		fprintf(stderr, " %s %5.1fms;", m.msg.c_str(), m.split - acc);
		acc = m.split;
	}
	fprintf(stderr, " Total %5.1fms\n", total);
	perf.splits.clear();
	return total;
}


bool import_deformations(const char* file, std::vector<Deformation>& deformations)
{
	FILE* f;
	char* buf = NULL;
	size_t len = 0;
	ssize_t read;
	size_t group;
	size_t shape_idx;
	int i1, i2, i3;
	float alpha, beta, gamma;
	int n;
	int line_number = 0;
	Deformation d;


	if (!(f = fopen(file, "r")))
		return false;

	deformations.clear();

	while ((read = getline(&buf, &len, f)) > 0) {
		line_number++;
		/* Skip comment and empty lines */
		if (len > 0 && (buf[0] == '#' || buf[0] == '\n'))
			continue;

		if ((n = sscanf(buf, "%lu,%lu,%d,%d,%d,%f,%f,%f", &group, &shape_idx, &i1, &i2, &i3, &alpha, &beta, &gamma)) != 8) {
			ERROR("While reading '%s', line %d: cannot parse deformation: '%s'", file, line_number, buf);
			deformations.clear();
			break;
		}

		d.group = group;
		d.shapeIdx = shape_idx;
		d.triangle = cv::Vec3i(i1, i2, i3);
		d.weights = cv::Vec3f(alpha, beta, gamma);
		deformations.push_back(d);

		DEBUG("Read deformation: group=%lu, idx=%lu, triangle=(%d, %d, %d), weights=(%f, %f, %f)",
				d.group, d.shapeIdx, d.triangle[0], d.triangle[1], d.triangle[2],
				d.weights[0], d.weights[1], d.weights[2]);
	}

	if (buf)
		free(buf);

	fclose(f);

	return true;
}


OneEuroFilter::OneEuroFilter(float beta, float fc): beta(beta), fc(fc) {}

void OneEuroFilter::update(float Te,
		const std::vector<cv::Point2f>& curPts,
		std::vector<cv::Point2f>& newPts)
{
	if (prevPos.empty()) {  /* First run */
		this->prevPos = curPts;
		this->prevVel.resize(curPts.size());
	} else if (curPts.size() != this->prevPos.size()) {
		ERROR("Size mismatch between update() calls");
		return;
	}
	newPts.clear();

	/* Filter points velocity */
	float alpha_d = this->smoothing_factor(Te, this->fc_d);
	std::vector<cv::Point2f> velocity(curPts.size());
	for (size_t i = 0; i < curPts.size(); i++) {
		cv::Point2f v = (curPts[i] - this->prevPos[i]) / Te;
		velocity[i] = this->mix(alpha_d, this->prevVel[i], v);
	}

	/* Filter points positions */
	newPts.resize(curPts.size());
	for (size_t i = 0; i < curPts.size(); i++) {
		float cutoff = this->fc + this->beta * norm(velocity[i]);
		float alpha = this->smoothing_factor(Te, cutoff);
		newPts[i] = this->mix(alpha, this->prevPos[i], curPts[i]);
	}

	this->prevVel = velocity;
	this->prevPos = newPts;
}

inline float OneEuroFilter::smoothing_factor(const float Te, const float fc)
{
	const float r = 2 * M_PI * fc * Te;
	return r / (r + 1);
}

inline cv::Point2f OneEuroFilter::mix(const float alpha,
		const cv::Point2f prev, const cv::Point2f cur)
{
	return alpha * cur + (1.f - alpha) * prev;
}


/* From https://stackoverflow.com/a/52377891/5309963 */
Subdiv2DIdx::Subdiv2DIdx(cv::Rect rectangle) : Subdiv2D{rectangle}
{
}

void Subdiv2DIdx::getTrianglesAndIndices(std::vector<cv::Vec6f> &triangles,
		std::vector<cv::Vec3i> &indices) const
{
	triangles.clear();
	indices.clear();
	int i, total = (int)(qedges.size() * 4);
	std::vector<bool> edgemask(total, false);
	const bool filterPoints = true;
	cv::Rect2f rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);

	for (i = 4; i < total; i += 2)
	{
		if (edgemask[i])
			continue;
		cv::Point2f a, b, c;
		int edge_a = i;
		int indexA = edgeOrg(edge_a, &a) -4;
		if (filterPoints && !rect.contains(a))
			continue;
		int edge_b = getEdge(edge_a, NEXT_AROUND_LEFT);
		int indexB = edgeOrg(edge_b, &b) - 4;
		if (filterPoints && !rect.contains(b))
			continue;
		int edge_c = getEdge(edge_b, NEXT_AROUND_LEFT);
		int indexC = edgeOrg(edge_c, &c) - 4;
		if (filterPoints && !rect.contains(c))
			continue;
		edgemask[edge_a] = true;
		edgemask[edge_b] = true;
		edgemask[edge_c] = true;

		indices.push_back(cv::Vec3i(indexA, indexB, indexC));
		triangles.push_back(cv::Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
	}
}
