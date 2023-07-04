/*
 * mozza-templater
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

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2/opencv.hpp>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>

#include <imgwarp_mls_rigid.h>

#include <iostream>

#include "mozza.h"


float signed_area (cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
bool point_in_triangle(cv::Point2f pt, cv::Vec6f triangle, cv::Vec3f& bary);
void compute_deformations(const std::vector<cv::Point2f>& neutral, const std::vector<cv::Point2f>& smile,
		const std::vector<std::vector<size_t>>& groups, const cv::Rect& imgSize,
		std::vector<Deformation>& deformations);
void export_deformations(const std::vector<Deformation>& deformations, const std::string& file);
void compute_deformations_from_imgs(const cv::Mat& neutral, const cv::Mat& smile,
		dlib::frontal_face_detector& fd, dlib::shape_predictor& sp,
		const double confidenceThresh, std::vector<Deformation>& deformations);
void usage(char* name);
void draw_text_box(cv::Mat& frame, cv::Point origin,
		const std::vector<std::string>& lines, bool topRightOrigin=false);


static const std::vector<size_t> MOUTH_INDICES = {
//		48, 49, 50, 52, 53, 54, 57};	// Mouth corners and bottom
		48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,  // Outer mouth
		/* 60, 61, 62, 63, 64, 65, 66, 67 */};           // Inner mouth
static const std::vector<size_t> LEFT_EYE_INDICES = {36, 37, 38, 39, 40, 41}; // previously {37, 38, 40, 41};
static const std::vector<size_t> RIGHT_EYE_INDICES = {42, 43, 44, 45, 46, 47}; // previously {43, 44, 46, 47};
static const std::vector<std::vector<size_t>> INDICES_GROUPS = {
		MOUTH_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES};
static const std::vector<std::string> HELP_LINES = {
	"space : play/pause",
	"L : single step",
	"",
	"D : toggle debug overlay",
	"H : show/hide help",
	"T : toggle Delaunay triangulation",
	"",
	"J : decrease alpha by 0.1",
	"K : increase alpha by 0.1",
	"A : decrease 1E beta by 0.01",
	"Z : decrease 1E beta by 0.01",
	"E : increase 1E Fc by 0.1",
	"R : increase 1E Fc by 0.1",
	"M : toggle MLS",
	"",
	"N : capture neutral face",
	"S : capture smiling face",
	"O : export deformation file",
	"I : import deformation file",
};
static const cv::Size BLUR_SIZE = cv::Size(5, 5);


int main(int argc, char** argv)
{
	std::string modelFile;
	std::string videoFile;
	std::string deformationFile = "./default.dfm";
	std::string neutralImgFile;
	std::string smileImgFile;
	bool paused = false;
	bool singleStep = false;
	bool enableMLS = true;
	bool enableDelaunay = false;
	bool dbgDisplay = true;
	bool displayHelp = true;
	bool batchMode = false;
	float alpha = 1.f;
	double confidenceThresh = 0.25;
	double frameTime = 16.0;
	double targetFrameTime;
	std::vector<cv::Point2f> neutralShape;
	std::vector<cv::Vec6f> neutralTriangles;
	std::vector<cv::Vec3i> neutralTrianglesIndices;
	std::vector<cv::Point2f> smilingShape;
	std::vector<Deformation> smileDeformations;
	dlib::frontal_face_detector faceDetector;
	dlib::shape_predictor shapePredictor;
	cv::VideoCapture cap;
	cv::Mat frame, frameBlurred, frameBGR, frameResized, frameDeformed;
	cv::Rect probableFaceROI;
	ImgWarp_MLS_Rigid mls;


	if (argc < 2)
		usage(argv[0]);

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-m") && i + 1 < argc) {
			modelFile = std::string(argv[++i]);
		} else if (!strcmp(argv[i], "-f") && i + 1 < argc) {
			videoFile = std::string(argv[++i]);
		} else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
			deformationFile = std::string(argv[++i]);
		} else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
			neutralImgFile = std::string(argv[++i]);
		} else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
			smileImgFile = std::string(argv[++i]);
		} else if (!strcmp(argv[i], "-b")) {
			batchMode = true;
		} else if (!strcmp(argv[i], "-h")) {
			usage(argv[0]);
		} else {
			fprintf(stderr, "Unrecognized argument: '%s'\n", argv[i]);
		}
	}

	if (modelFile.empty()) {
		fprintf(stderr, "Missing shape predictor model file.\n");
		return 1;
	}

	// We need a face detector.  We will use this to get bounding boxes for
	// each face in an image.
	faceDetector = dlib::get_frontal_face_detector();
	// And we also need a shape_predictor.  This is the tool that will predict face
	// landmark positions given an image and face bounding box.  Here we are just
	// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
	// as a command line argument.
	dlib::deserialize(modelFile) >> shapePredictor;


	if (batchMode) {
		cv::Mat neutralImg = cv::imread(neutralImgFile);
		cv::Mat smileImg = cv::imread(smileImgFile);
		if (neutralImg.empty()) {
			ERROR("failed to load image file: '%s'", neutralImgFile.c_str());
			return 1;
		} else if (smileImgFile.empty()) {
			ERROR("failed to load image file: '%s'", smileImgFile.c_str());
			return 1;
		} else if (deformationFile.empty()) {
			ERROR("no deformation file specified");
			return 1;
		}
		compute_deformations_from_imgs(neutralImg, smileImg, faceDetector,
				shapePredictor, confidenceThresh, smileDeformations);
		export_deformations(smileDeformations, deformationFile);
		return 0;
	} else if (!neutralImgFile.empty() || !smileImgFile.empty()) {
		WARN("-n and -s arguments are only read in batch mode (-b)");
	}

	// Load video from file, or defaults to webcam
	if (!videoFile.empty()) {
		cap = cv::VideoCapture(videoFile);
		if (!cap.isOpened())
			fprintf(stderr, "Couldn't load '%s'\n", videoFile.c_str());
	} else {
		cap = cv::VideoCapture(0);
	}
	if (!cap.isOpened()) {
		fprintf(stderr, "Failed to open video capture.\n");
		return 1;
	}

	const float CAP_FPS = cap.get(cv::CAP_PROP_FPS);
	targetFrameTime = 1000.0 / CAP_FPS;

	mls.gridSize = MLS_GRID_SIZE;
	mls.preScale = true;
	mls.alpha = 1.4;  // Important, no sane default for this one

	std::string winName = "Mozza-templater";
	cv::namedWindow(winName);
	OneEuroFilter oeFilter(0.1f, 5.0f);

	while (1) {
		perf_start();

		if (!paused || singleStep || frame.empty()) {
			cap >> frame; // get a new frame from camera

			if (frame.empty()) {
				if (!videoFile.empty()) {
					return 0;
				} else {
					fprintf(stderr, "Webcam stream ended unexpectedly: exiting...\n");
					return 1;
				}
			}

			if (videoFile.empty())  // Mirror webcam image
				cv::flip(frame, frame, 1);

			singleStep = false;
		}

		cv::Rect fullFrameRect(0, 0, frame.cols, frame.rows);

		perf_split("capture");


		// Blur to try and counteract the CCD noise
		cv::GaussianBlur(frame, frameBlurred, BLUR_SIZE, 0);
		dlib::cv_image<dlib::bgr_pixel> dlibBlurred(frameBlurred);

		perf_split("preprocess");


		std::vector<dlib::rectangle> dets;
		find_faces(frame, faceDetector, confidenceThresh, dets, probableFaceROI);
		std::cerr << "Probable ROI: " << probableFaceROI << std::endl;

		if (dets.empty())
			fprintf(stderr, "Failed to detect any faces\n");
		else if (dets.size() > 1)  // TODO support simultaneous detections
			fprintf(stderr, "More than one face detected, things might break\n");

		perf_split("face detect");


		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector<std::vector<cv::Point2f>> facesPts;
		find_landmarks(frameBlurred, shapePredictor, dets, facesPts);

		perf_split("shape predict");


		/* The shape predictor seem to be able to fit points outside of the image,
		 * so we extend the aera in which the compute the Delaunay triangulation. */
		cv::Rect big = fullFrameRect + fullFrameRect.size() * 2;
		big -= cv::Point(fullFrameRect.width, fullFrameRect.height);
		Subdiv2DIdx subdiv(big);
		std::vector<cv::Vec6f> triangles;
		std::vector<cv::Vec3i> trianglesIndices;
		if (!facesPts.empty()) {
			try {
				subdiv.insert(facesPts[0]);
				subdiv.getTrianglesAndIndices(triangles, trianglesIndices);
			} catch (cv::Exception& e) {
				std::cerr << "Something went wrong while creating Delaunay triangulation: " << e.what() << std::endl;
			}
		}

		perf_split("Delaunay");


		// Try to remove jitter from the facial landmarks
		std::vector<cv::Point2f> landmarks;
		if (!facesPts.empty())
			oeFilter.update(frameTime / 1000.f, facesPts[0], landmarks);

		perf_split("Stabilization");


		std::vector<std::vector<cv::Point2f>> srcGroups;
		std::vector<std::vector<cv::Point2f>> dstGroups;
		if (!landmarks.empty())
			get_deformation_groups(smileDeformations, landmarks, alpha, srcGroups, dstGroups);

		frame.copyTo(frameDeformed);
		if (!dstGroups.empty() && enableMLS) {
			for (size_t i = 0; i < srcGroups.size(); i++) {
				compute_MLS_on_ROI(frameDeformed, mls, srcGroups[i], dstGroups[i]);
			}
		}

		perf_split("MLS");


		if (dbgDisplay && !facesPts.empty()) {
			if (enableDelaunay)
				draw_delaunay(frameDeformed, subdiv, cv::Scalar(255, 225, 255));
			// for (cv::Point2f pt : facesPts[0])
			// 	cv::circle(frameDeformed, pt, 2, cv::Scalar(0, 0, 255));
			for (cv::Point2f pt : landmarks)
				cv::circle(frameDeformed, pt, 2, cv::Scalar(255, 128, 128));
			for (size_t i = 0; i < srcGroups.size(); i++)
				for (size_t j = 0; j < srcGroups[i].size(); j++)
				 	cv::line(frameDeformed, srcGroups[i][j], dstGroups[i][j], cv::Scalar(0, 255, 0));
			for (const dlib::rectangle& r : dets) {
				cv::Rect faceRect(r.left(), r.top(), r.width(), r.height());
				cv::rectangle(frameDeformed, faceRect, cv::Scalar(0, 128, 0));
			}
			cv::rectangle(frameDeformed, probableFaceROI, cv::Scalar(128, 0, 128));
		}

		if (displayHelp) {
			std::vector<std::string> hud_lines = {
				"alpha = " + std::to_string(alpha),
				"1E beta = " + std::to_string(oeFilter.beta),
				"1E Fc = " + std::to_string(oeFilter.fc),
			};
			draw_text_box(frameDeformed, cv::Point(20, 20), hud_lines, true);
			draw_text_box(frameDeformed, cv::Point(20, 20), HELP_LINES);
		}

		cv::imshow(winName, frameDeformed);

		/* Ensure framerate is stable when playing a video */
		frameTime = perf_print();
		int idleTime = targetFrameTime - frameTime;
		int key = cv::waitKey(idleTime > 1 ? idleTime : 1);

		if (key == 113) { // Q
			break;
		} else if (key == ' ') {
			paused = !paused;
		} else if (key == 'l') {
			singleStep = true;
		} else if (key == 'm') {
			enableMLS = !enableMLS;
		} else if (key == 'd') {
			dbgDisplay = !dbgDisplay;
		} else if (key == 'h') {
			displayHelp = !displayHelp;
		} else if (key == 'j') {
			alpha -= 0.1f;
		} else if (key == 'k') {
			alpha += 0.1f;
		} else if (key == 'a') {
			oeFilter.beta -= 0.01f;
		} else if (key == 'z') {
			oeFilter.beta += 0.01f;
		} else if (key == 'e') {
			oeFilter.fc -= 0.1f;
		} else if (key == 'r') {
			oeFilter.fc += 0.1f;
		} else if (key == 'n') {  // Capture neutral template
			// FIXME
			if (!facesPts.empty()) {
				neutralShape = facesPts[0];
			}
		} else if (key == 's') {  // Capture smiling template
			if (!neutralShape.empty() && !facesPts.empty())
				match_points(facesPts[0], neutralShape, smilingShape);
			else
				fprintf(stderr, "Warning: no neutral reference to compute smile deformation\n");
			if (!smilingShape.empty()) {
				cv::Rect imgSize(0, 0, frame.cols, frame.rows);
				compute_deformations(neutralShape, smilingShape, INDICES_GROUPS,
						imgSize, smileDeformations);
			} else {
				fprintf(stderr, "Warning: point matching failed\n");
			}
		} else if (key == 't') {
			enableDelaunay = ! enableDelaunay;
		} else if (key == 'o') {
			export_deformations(smileDeformations, deformationFile);
		} else if (key == 'i') {
			import_deformations(deformationFile.c_str(), smileDeformations);
		}
	}

	cv::destroyAllWindows();

	return 0;

}


float signed_area (cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}


bool point_in_triangle(cv::Point2f pt, cv::Vec6f triangle, cv::Vec3f& bary)
{
	bool b1, b2, b3;
	cv::Point2f v1(triangle[0], triangle[1]);
	cv::Point2f v2(triangle[2], triangle[3]);
	cv::Point2f v3(triangle[4], triangle[5]);

	/* Compute barycentric coordinates */
	float area = signed_area(v1, v2, v3);
	bary[0] = signed_area(pt, v2, v3);
	bary[1] = signed_area(pt, v3, v1);
	bary[2] = signed_area(pt, v1, v2);
	bary /= area;

	/* Check that all barycentric coordinates have the same sign */
	b1 = bary[0] < 0.0f;
	b2 = bary[1] < 0.0f;
	b3 = bary[2] < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}


/* - `neutral` and `smile`: respectively contain the facial landmark
 *   points of a neutral and a smiling face, previously aligned using the
 *   `match_points()` function.
 *
 * - `groups`: vector of vectors of indices, corresponding to points in
 *   `neutral` or `smile`, to define groups of points.
 *
 * - `imgSize`: dimensions of the full frame
 *
 * - `deformations`: output vector */
void compute_deformations(const std::vector<cv::Point2f>& neutral, const std::vector<cv::Point2f>& smile,
		const std::vector<std::vector<size_t>>& groups, const cv::Rect& imgSize,
		std::vector<Deformation>& deformations)
{
	deformations.clear();

	/* The shape predictor seem to be able to fit points outside of the image,
	 * so we extend the aera in which the compute the Delaunay triangulation. */
	cv::Rect big = imgSize + imgSize.size() - cv::Point(imgSize.width, imgSize.height) / 2;

	Subdiv2DIdx subdiv(big);
	std::vector<cv::Vec6f> triangles;
	std::vector<cv::Vec3i> trianglesIdx;
	subdiv.insert(neutral);
	subdiv.getTrianglesAndIndices(triangles, trianglesIdx);

	for (size_t group = 0; group < groups.size(); group++) {
		for (size_t i : groups[group]) {
			Deformation d;
			d.shapeIdx = i;
			d.group = group;
			//d.vec = smile[i] - neutral[i];

			// For each deformation point, find which triangle it belongs to
			for (size_t t = 0; t < triangles.size(); t++) {
				cv::Vec3f barycentricCoords;
				if (point_in_triangle(smile[i], triangles[t], barycentricCoords)) {
					d.triangle = trianglesIdx[t];
					d.weights = barycentricCoords;
					break;
				}
			}

			deformations.push_back(d);
		}
	}
}


void export_deformations(const std::vector<Deformation>& deformations, const std::string& file)
{
	FILE* f;

	if (deformations.empty()) {
		fprintf(stderr, "No deformations to export\n");
		return;
	}

	if (!(f = fopen(file.c_str(), "w"))) {
		fprintf(stderr, "Couldn't open '%s' for writing\n", file.c_str());
		return;
	}

	fprintf(f, "# group, index, triangle indices {0, 1, 2}, barycentric coords weights {alpha, beta, gamma}\n");
	for (const Deformation& d : deformations) {
		fprintf(f, "%lu,%lu,%d,%d,%d,%f,%f,%f\n", d.group, d.shapeIdx,
				d.triangle[0], d.triangle[1], d.triangle[2],
				d.weights[0], d.weights[1], d.weights[2]);
	}

	fclose(f);

	fprintf(stderr, "Exported deformation info to '%s'\n", file.c_str());
}


void compute_deformations_from_imgs(const cv::Mat& neutral, const cv::Mat& smile,
		dlib::frontal_face_detector& fd, dlib::shape_predictor& sp,
		const double confidenceThresh, std::vector<Deformation>& deformations)
{
	std::vector<dlib::rectangle> neutralDets;
	std::vector<dlib::rectangle> smileDets;
	std::vector<std::vector<cv::Point2f>> neutralPts;
	std::vector<std::vector<cv::Point2f>> smilePts;
	std::vector<cv::Point2f> matchedSmilePts;
	cv::Rect imgSize(0, 0, neutral.cols, neutral.rows);

	deformations.clear();

	find_faces(neutral, fd, confidenceThresh, neutralDets, imgSize);
	/* reset the resize rate, we can't assume the smiling face will have the
	 * same dimensions */
	imgSize = cv::Rect(0, 0, neutral.cols, neutral.rows);
	find_faces(neutral, fd, confidenceThresh, smileDets, imgSize);

	if (neutralDets.empty() || smileDets.empty()) {
		WARN("Failed to detect any faces in at least one image");
		return;
	} else if (neutralDets.size() > 1 || smileDets.size() > 1) {  // TODO support simultaneous detections
		WARN("More than one face detected in at least one image. Using first detection");
	}

	find_landmarks(neutral, sp, neutralDets, neutralPts);
	find_landmarks(smile, sp, smileDets, smilePts);

	match_points(smilePts[0], neutralPts[0], matchedSmilePts);

	compute_deformations(neutralPts[0], matchedSmilePts, INDICES_GROUPS, imgSize, deformations);
}


void usage(char* name)
{
	fprintf(stderr, "Usage: %s -m MODEL_FILE [-f VIDEO_FILE] [-o EXPORT_FILE]\n", name);
	fprintf(stderr, "       %s -b -m MODEL_FILE -n NEUTRAL_IMG -s SMILING_IMG -o EXPORT_FILE\n\n", name);
	fprintf(stderr, "  Interactive program to generate smile deformation vectors.\n\n");
	fprintf(stderr, "  -b\tEnable batch mode. Compute deformations from image files.\n");
	fprintf(stderr, "    \tRequires -m, -n, -s ,and -o\n");
	fprintf(stderr, "  -f\tInput video file. If none is provided, defaults to the webcam.\n");
	fprintf(stderr, "  -h\tDisplay this help.\n");
	fprintf(stderr, "  -m\tShape predictor model file. Mandatory.\n");
	fprintf(stderr, "  -n\tImage file containing a face with a neutral expression.\n");
	fprintf(stderr, "  -o\tFile to export the deformation info to.\n");
	fprintf(stderr, "  -s\tImage file containing a face with a smiling expression.\n");
	exit(0);
}


void draw_text_box(cv::Mat& frame, cv::Point origin,
		const std::vector<std::string>& lines, bool topRightOrigin)
{
	const int thickness = 1;
	const cv::HersheyFonts font = cv::FONT_HERSHEY_DUPLEX;
	const double fontScale = 0.6;

	std::vector<cv::Point> origins;
	cv::Point org, offset;
	cv::Rect bbox;
	int baseline;

	/* Compute BBox for the  whole text box */
	for (size_t i = 0; i < lines.size(); i++) {
		cv::Size txtSize = cv::getTextSize(lines[i], font, fontScale, thickness, &baseline);
		bbox |= cv::Rect(org, txtSize + cv::Size(0, baseline));
		org += cv::Point(0, txtSize.height);
		origins.push_back(org);
		org += cv::Point(0, baseline);
	}

	if (!topRightOrigin)
		offset = origin;
	else
		offset = cv::Point(frame.cols - origin.x - bbox.width, origin.y);

	/* Create a dark background for the text box */
	bbox = bbox - cv::Point(10, 10) + cv::Size(20, 20) + offset;
	frame(bbox) /= 2;

	for (size_t i = 0; i < lines.size(); i++)
		cv::putText(frame, lines[i], origins[i] + offset, font, fontScale,
				cv::Scalar(0xf2, 0xf8, 0xf8), thickness, cv::LINE_AA);
}
