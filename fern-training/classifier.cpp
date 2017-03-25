#include "classifier.h"
#include <map>
#include <utility>
#include <algorithm>

bool comparePairs(const  std::pair<uint16_t, uint32_t> &l,
	const  std::pair<uint16_t, uint32_t> &r)
{
	return l.second > r.second;
}

/*
*  Create a database entry (i.e. Feature) from the stable point.
*  it merges all the descriptors from the stable points (i.e. computed from different views)
*  to create an unique descriptor.
*/
void Classifier::createFeatureFromStablePoint(const StablePoint &sp)
{
	std::map<uint16_t, uint32_t> indexHistogram;
	uint16_t descSize = sp.viewPointPtDescriptors.cols * 8;
	vector<int> bitHistogram(descSize, 0);

	int totalDesc = sp.viewPointPtDescriptors.rows;
	int colSize = sp.viewPointPtDescriptors.cols;

	Mat finalDescriptor = Mat::zeros(1, colSize, CV_8UC1);
	for (int d = 0; d < totalDesc; ++d)
	{
		Mat desc = sp.viewPointPtDescriptors.row(d);
		indexHistogram[sp.indices[d]]++;

		for (int col = 0; col < colSize; ++col)
			for (int bit = 7; bit >= 0; --bit)
			{
				uchar tmp = (1 << bit);
				if ((desc.at<uchar>(0, col) & tmp) > 0)
				{
					bitHistogram[col * 8 + (7 - bit)]++;
				}
			}
	}

	for (int col = 0; col < colSize; ++col)
		for (int bit = 7; bit >= 0; --bit)
		{
			uchar tmp = (1 << bit);
			if (bitHistogram[col * 8 + (7 - bit)] >
				STABILITY_OF_BITS * totalDesc)
			{
				finalDescriptor.at<uchar>(0, col) |= tmp;
			}
		}

	vector< std::pair<uint16_t, uint32_t> > mPairs;
	float sum = 0;
	std::map<uint16_t, uint32_t>::iterator mapIt;
	for (mapIt = indexHistogram.begin(); mapIt != indexHistogram.end(); ++mapIt)
	{
		mPairs.push_back(*mapIt);
		sum += mapIt->second;
	}
	sort(mPairs.begin(), mPairs.end(), comparePairs);

	float tmpSum = 0;
	for (int i = 0; (i < mPairs.size() && (tmpSum / sum < .75)); ++i)
	{
		uint16_t index = mPairs[i].first;
		uint32_t value = mPairs[i].second;
		lookUpTable[index].push_back((uint32_t)featureTable.size());
		tmpSum += value;
	}

	FernFeat _feat;
	_feat.idx = (uint32_t)featureTable.size();
	_feat.descriptor = finalDescriptor.clone();
	_feat.x = sp.pt.x;
	_feat.y = sp.pt.y;
	featureTable.push_back(_feat);
}


Classifier::Classifier() :
	imgRefSize(Size(0, 0)),
	detectorCode(Feat::FAST),
	extractorCode(Desc::BRIEF),
	corners(),
	featureTable()
{
	for (int i = 0; i < FernUtils::INDEX_MAX_VALUE; ++i)
		lookUpTable[i] = vector<uint16_t>();
};

/*
*  Initializes Classifier using the list of most stable keypoints.
*/
void Classifier::initialize(const Size img,
	const vector<vector<StablePoint> > &database,
	Feat::Code    feat,
	Desc::Code desc)
{
	imgRefSize = img;
	detectorCode = feat;
	extractorCode = desc;
	createCorners();

	for (int v = 0; v < database.size(); ++v)
		for (int sp = 0; sp < database[v].size(); ++sp)
		{
			createFeatureFromStablePoint(database[v][sp]);
		}
}

/*
*  Private method to create the targets corners position
*  from the width and height.
*  The order of the points are Top Left, Top Right, Bottom Right, Bottom Left.
*/
void Classifier::createCorners()
{
	corners.clear();
	corners.push_back(Point2f(0, imgRefSize.height));//TL
	corners.push_back(Point2f(imgRefSize.width, imgRefSize.height));//TR
	corners.push_back(Point2f(imgRefSize.width, 0));//BR
	corners.push_back(Point2f(0, 0)); //BL
}

void Classifier::setIndexPositions(vector<uint16_t> &idxs)
{
	indexPositions = idxs;
}

/**
*  Writes a binary model from the filename.
*  @param filename string. Path to the binary file containing the model
*
*  Format:
*  4 bytes - Value of detectorCode used to generate the model
*  4 bytes - Value of extractorCode used to generate the model
*  4 bytes - Size in bytes of the descriptor used
*  8 bytes (4bytes, 4bytes) - Size of the target image used to train.
*  8 bytes - Length in bytes of the feature table
*  .........Table of Features (consecutive Features)
*      A Feature is:
*          X bytes, Descriptor bytes. The amount of X is the Size in bytes of
*                   the descriptor.
*          2 bytes, 'x' coordiate of the feature.
*          2 bytes, 'y' coordinate of the feature.
*          4 bytes, id of the feature.
*  ..........Lookup Table from 0 to 8192
*  index0: 8 bytes - Size in bytes of the first vector (So)
*  index0: So * 4 bytes to fillup the vector
*  index1: 8 bytes (S1)
*  index1: S1 * 4 bytes
*  index2: 8 bytes (S2)
*  index2: S2 * 4 bytes
*  .
*  .
*  .
*
**/
bool Classifier::write(const string &filename)
{
	FILE* dfile = fopen(filename.c_str(), "w");
	printf("Saving model in %s \n", filename.c_str());

	fwrite(&detectorCode, sizeof(uint32_t), 1, dfile);
	fwrite(&extractorCode, sizeof(uint32_t), 1, dfile);
	uint32_t dSize;
	switch (featureTable[0].descriptor.depth())
	{
	case CV_8U:
		dSize = featureTable[0].descriptor.cols;
		break;
	case CV_32F:
		dSize = featureTable[0].descriptor.cols * sizeof(float);
		break;
	default:
		dSize = featureTable[0].descriptor.cols;
		break;
	}
	fwrite(&dSize, sizeof(uint32_t), 1, dfile);
	fwrite(&imgRefSize, sizeof(Size), 1, dfile);
	uint16_t amountOfElements = featureTable.size();
	fwrite(&amountOfElements, sizeof(uint16_t), 1, dfile);
	for (uint16_t e = 0; e < amountOfElements; ++e)
	{
		fwrite(featureTable[e].descriptor.data, sizeof(uchar), dSize, dfile);
		fwrite(&featureTable[e].x, sizeof(uint16_t), 1, dfile);
		fwrite(&featureTable[e].y, sizeof(uint16_t), 1, dfile);
		fwrite(&featureTable[e].idx, sizeof(uint32_t), 1, dfile);

	}

	uint16_t indices = 0;
	for (uint16_t i = 0; i < 8192; i++)
	{
		if (lookUpTable[i].size() > 0)
			indices++;
	}
	fwrite(&indices, sizeof(uint16_t), 1, dfile);

	//8192 = 2^13

	for (uint16_t i = 0; i < 8192; i++)
	{
		uint16_t numberOfElements = lookUpTable[i].size();
		if (numberOfElements > 0)
		{
			fwrite(&i, sizeof(uint16_t), 1, dfile);
			fwrite(&numberOfElements, sizeof(uint16_t), 1, dfile);
			fwrite(&lookUpTable[i][0], sizeof(uint16_t), numberOfElements, dfile);
		}
	}


	uint16_t index = indexPositions.size();
	fwrite(&index, sizeof(uint16_t), 1, dfile);
	fwrite(&indexPositions[0], sizeof(uint16_t), index, dfile);
	fclose(dfile);
	return true;
}

void Classifier::saveModel(const string &modelFilename)
{
	write(modelFilename);
}