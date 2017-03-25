#pragma once
#include "feature.h"
#include "index.h"


/* Training class structure */
class Train
{
public:
	static std::string featureName(Feat::Code code);
	static std::string descriptorName(Desc::Code code);

	static cv::Ptr<cv::FeatureDetector>     create(Feat::Code code);
	static cv::Ptr<cv::DescriptorExtractor> create(Desc::Code code);


	static void trainFromReferenceImage(cv::Mat &image, ViewParams &params, std::vector<std::vector<StablePoint> > &database);
	
	
	// They should be private, but for debugging purpose so they are public.
	static vector<uint16_t> computeDescIndices(vector<vector<StablePoint>> &database, int size);
	static bool sortByProbNearToDot5(std::pair<int, float> &first, std::pair<int, float> &second);
	static int getNearbyPoint(const cv::Point& p, const vector<vector<int> > &grid);
	static void computeIndices(vector<vector<StablePoint>> &database, vector<uint16_t> &indices);



	static const int minDistance2ReferencePoint;
	static const int minDistance2ReferencePointSqr;
	static const int HALF_PATCH_WIDTH;
	static const int SIGMA_BLUR;

private:
	static double rad2Deg(double rad);
	static double deg2Rad(double deg);
	static void warpMatrix(cv::Size sz, double yaw, double pitch, double roll, double scale, double fovy, cv::Mat &M, vector<cv::Point2f>* corners);
	
	static cv::Mat keyPoint2Mat(const vector<cv::KeyPoint >& keypoints);
	static std::vector<cv::Point> mat2Points(const cv::Mat& stub);
	static void projectKeypoints(const vector<cv::KeyPoint> &original, const cv::MatExpr M, vector<cv::Point> &transformedPoints);

	static void selectPoints(cv::Mat &image, cv::Mat &M, vector<cv::Point2f> &corners, cv::Size refImgSize, cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &extractor, vector<cv::KeyPoint> &kps, vector<uint16_t> &indxs, cv::Mat &desc);
	static void warpImage(const cv::Mat &src, double yaw, double pitch, double roll, double scale, double fovy, cv::Mat &dst, cv::Mat &M, std::vector<cv::Point2f> &corners);
	static bool keyPointOrderingByResponse(const cv::KeyPoint& k1, const cv::KeyPoint& k2);
	static void filterKeyPoints(cv::Size targetImgSize, const vector<cv::Point2f> &corners, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Point> &pointsTransformed);
	static void selectMostStableKeypoints(const vector<vector<cv::KeyPoint>>& keypoints, const vector<vector<uint16_t>> &indices, const vector<cv::Mat> &descriptors, const vector<cv::Mat>&transfMatrix, const vector < vector < cv::Point2f > > & corners, int maxPointsPerView, cv::Size refImgSize, cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &extractor, vector<StablePoint> &bestPoints);
	static bool compareStablePoints(const StablePoint &x1, const StablePoint &x2);
	
	
};