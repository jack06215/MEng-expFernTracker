#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>


/* Feature class structure */
struct Space
{
public:
	double min;
	double max;
	double step;
	Space(double _min, double _max, double _step) :
		min(_min), max(_max), step(_step) {}
};

struct Feat
{
private:
	static std::string names[5];
public:
	enum Code { ORB, FAST, MSER, SIFT, SURF };

	static std::string getName(Feat::Code code);
	static cv::Ptr<cv::FeatureDetector>  create(Feat::Code code);
};

struct Desc
{
private:
	static std::string names[6];
public:
	enum Code { ORB, BRIEF, SIFT, SURF, FREAK, BRISK };

	static std::string getName(Desc::Code code);
	static cv::Ptr<cv::DescriptorExtractor> create(Desc::Code code);
};

struct ViewParams
{
public:

	double fov;
	Space  yaw; //Vertical Axis
	Space  pitch; //Horizontal Axis
	Space  roll;  //Depth Axis
	Space  scale;
	int    numberOfViewPoints;
	int    numberOfPointsPerViewPoints;
	Feat::Code detector;
	Desc::Code extractor;

	ViewParams(double _fov,
		Space _yaw,
		Space _pitch,
		Space _roll,
		Space _scale,
		int _nOfViewPoints,
		int _nOfPointsPerView,
		Feat::Code _detector,
		Desc::Code _extractor)
		:fov(_fov), yaw(_yaw), pitch(_pitch), roll(_roll), scale(_scale),
		numberOfViewPoints(_nOfViewPoints),
		numberOfPointsPerViewPoints(_nOfPointsPerView),
		detector(_detector),
		extractor(_extractor) {}
};

struct StablePoint
{
public:
	cv::Point pt;
	std::vector<cv::KeyPoint> viewPointPts;
	std::vector<int>       imageNumber;
	std::vector<uint16_t>      indices;
	cv::Mat				viewPointPtDescriptors;

	StablePoint() :
		pt(cv::Point(0, 0)), viewPointPtDescriptors(0, 0, CV_8UC1) {}

	StablePoint(cv::Point _pt, cv::KeyPoint view, int image, cv::Mat &desc, uint16_t index) :
		pt(_pt), viewPointPtDescriptors(0, desc.cols, CV_8UC1)
	{
		viewPointPts.push_back(view);
		imageNumber.push_back(image);
		viewPointPtDescriptors.push_back(desc);
		indices.push_back(index);
	}
};

