#include "feature.h"

std::string Desc::names[6] = { "ORB", "BRIEF", "SIFT", "SURF", "FREAK", "BRISK" };
std::string Feat::names[5] = { "ORB", "FAST", "MSER", "SIFT", "SURF" };

std::string Feat::getName(Feat::Code code)
{
	return names[code];
}

cv::Ptr<cv::FeatureDetector>  Feat::create(Feat::Code code)
{
	switch (code)
	{
	case Feat::ORB:
		return cv::ORB::create(1000);
		break;
	case Feat::FAST:
		return cv::FastFeatureDetector::create();
		break;
	case Feat::MSER:
		return cv::MSER::create();
		break;
	case Feat::SIFT:
		return cv::xfeatures2d::SIFT::create();
		break;
	case Feat::SURF:
		return cv::xfeatures2d::SURF::create();
		break;
	default:
		return cv::FastFeatureDetector::create();
		break;
	}
}

std::string Desc::getName(Desc::Code code)
{
	return names[code];
}
cv::Ptr<cv::DescriptorExtractor> Desc::create(Desc::Code code)
{
	switch (code)
	{
	case Desc::ORB:
		return cv::ORB::create();
		break;
	case Desc::BRIEF:
		return cv::xfeatures2d::BriefDescriptorExtractor::create();
		break;
	case Desc::SIFT:
		return cv::xfeatures2d::SiftDescriptorExtractor::create();
		break;
	case Desc::SURF:
		return cv::xfeatures2d::SurfDescriptorExtractor::create();
		break;
	case Desc::FREAK:
		return cv::xfeatures2d::FREAK::create();
	case Desc::BRISK:
		return cv::BRISK::create();
	default:
		return cv::xfeatures2d::BriefDescriptorExtractor::create();
		break;
	}
}