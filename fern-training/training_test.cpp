#include <iostream>
#include <opencv2/opencv.hpp>

#include "training.h"
#include "classifier.h"


int main(void)
{
	// feature detector and descriptor
	Feat::Code feat = Feat::FAST;
	Desc::Code desc = Desc::BRIEF;

	// training parameter: minimum, maximum and step
	Space yaw(-30, 30, 60);
	Space pitch(-30, 30, 60);
	Space roll(-10, 10, 20);
	Space scale(1 * pow(.8, 7), 1.0, .8);
	// field of view, number of viewpoints per training set and
	double fov = 37;
	int numViewPoints = 1000;
	int numKeyPointsPerViewPoint = 70;
	
	// image and its grayscale container
	cv::Mat img = cv::imread("texture2.jpg");
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_RGB2GRAY);
	
	// training database
	std::vector<std::vector<StablePoint>> database;

	// fern classifier
	Classifier   classifiers;



	ViewParams params(fov,
					  yaw,
					  pitch,
			    	  roll,
					  scale,
					  numViewPoints,
					  numKeyPointsPerViewPoint,
					  feat,
				      desc);


	cv::Ptr<cv::FeatureDetector> featDetector = Train::create(feat);
	cv::Ptr<cv::DescriptorExtractor> descExtractor = Train::create(desc);

	cv::imshow("Training image", img);
	cv::waitKey(1);

	// start building up the descriptor database for each training set
	Train::trainFromReferenceImage(img_gray, params, database);
	std::cout << "trainFromReferenceImage() finished\n";
	
	// construct a histogram for all population in the database and extract the indeices for the most frequent 13 elements
	std::vector<uint16_t> index = Train::computeDescIndices(database, 13);
	std::cout << "computeDescIndices() finished\n";

	classifiers.setIndexPositions(index);
	Train::computeIndices(database, index);


	classifiers.initialize(img_gray.size(), database, feat, desc);
	classifiers.saveModel("training.model");

	return 0;
}
