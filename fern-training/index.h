#ifndef __index__
#define __index__

#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0)

using namespace std;
using namespace cv;

class Index
{
public:
	static uint16_t getDescIndex(const Mat& img, const vector<uint16_t> &positions);
};

#endif