#ifndef __detection3D__taylor__
#define __detection3D__taylor__

#include <iostream>
#include "training.h"
#include "bitcount.h"
#include "ferns.h"

struct FernFeat
{
	uint32_t idx;
	uint16_t x, y;
	Mat descriptor;
};

class Classifier
{
public:

	Feat::Code detectorCode;
	Desc::Code extractorCode;

	Classifier();

	void initialize(const Size targetSize,
		const vector<vector<StablePoint> > &database,
		Feat::Code feat,
		Desc::Code desc);

	void setIndexPositions(vector<uint16_t> &idxs);
	void saveModel(const string &modelFilename);

private:
	const float STABILITY_OF_BITS = .50;

	bool write(const string &filename);
	void createCorners();
	void createFeatureFromStablePoint(const StablePoint &sp);

	Size               imgRefSize;
	vector<Point2f>    corners;
	vector<FernFeat> featureTable;
	vector<uint16_t>   lookUpTable[FernUtils::INDEX_MAX_VALUE];
	vector<uint16_t>   indexPositions;

};


#endif /* defined(__detection3D__taylor__) */