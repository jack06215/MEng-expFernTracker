#ifndef __ferns__
#define __ferns__

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
//#include "utils.h"
#include "training.h"

using namespace std;
using namespace cv;

class FernUtils
{
public:
	static const int MIN_NUM_MATCHES = 10;
	static const int INDEX_MAX_VALUE = 8192;
	//static const int BINARY_THRESHOLD = 48 to 60 are good values
	static const int BINARY_THRESHOLD = 48;
	static const int MAX_POINTS = 100;
	static const uint64_t dict[256];
	static const char bits[256][8];
	/*
	*  Util function to check the homography
	*  From "Multiple View Geometry"
	*  If the determinant of the top-left 2x2 matrix is > 0 the transformation is orientation
	*  preserving. Else is <0 it's orientation reversing (bad homography).
	*/
	static bool goodHomography(const cv::Mat &H);

	static int  computeHomography(const vector<Point2f> &obj_pts,
		const vector<Point2f> &scn_pts,
		const vector<Point2f> &corners,
		Mat &H,
		vector<Point2f> &dstCorners);

	static int  computeFundamentalMatrix(const vector<Point2f> &obj_pts,
		const vector<Point2f> &scn_pts,
		const vector<Point2f> &corners,
		Mat &F,
		vector<Point2f> &obj_inliers,
		vector<Point2f> &scn_inliers,
		vector<Vec3f> &cLines);

	static Point pointOnALine(const Vec3f &line,
		const Point2f &pt);

	static Point twoLinesIntersection(const Point2f &sl,
		const Point2f &el,
		const Point2f &sl2,
		const Point2f &el2);

	static Vec3f getLine(const Point2f &p1,
		const Point2f &p2);

	static double radiusOfInscribedCircled(const Point center,
		const vector<Point2f> &corners);
	static bool sortByQueryId(const DMatch &left, const DMatch &right);
	static bool sortByDistance(const DMatch &left, const DMatch &right);

	static int  numInliers(const vector<Point2f> &obj_pts,
		const vector<Point2f> &scn_pts);

	static void crossProdut(const Vec3f &a, const Vec3f &b, Vec3f &pt);



	/*Some drawing utilities*/
	/*
	* Draws target corners as their convex hull.
	* hShift = horizontal displacement (translation)
	* vShift = vertical displacement (translation)
	*/
	static void drawPoint(Mat &frameOut,
		Point2f &pt,
		Scalar &color,
		Point2f &shift,
		int thickness = 10)
	{
		line(frameOut, pt + shift, pt + shift, color, thickness);
	}

	static void drawCorners(Mat &frameOut,
		const vector<Point2f> &corners,
		Scalar &color,
		Point2f &shift,
		int thickness = 4);

	static void drawPoints(Mat &frameOut,
		const vector<Point2f> &corners,
		Scalar &color,
		Point2f &shift,
		int thickness = 4);

	static void drawELines(Mat &frameOut,
		const vector<Vec3f> &eLines,
		Scalar &color,
		Point2f &shift,
		int width = 200);
	static void refineCornersWithEpipolarLines(const vector<Vec3f> &eLines,
		const vector<Point2f> &corners,
		vector<Point2f> &refined);

	static void drawObject(Mat &frameOut,
		const vector<Point2f> &corners,
		Scalar &color,
		Scalar &topColor,
		Point2f &shift,
		float percent = 50,
		int thickness = 4,
		float alpha = .4);

	static void drawPercentCircle(Mat &frameOut,
		const Point2f &center,
		int radius,
		float percent,
		float alpha);

};

#endif
