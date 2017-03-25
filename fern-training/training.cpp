#define _USE_MATH_DEFINES
#include <cmath>
#include "training.h"

const int Train::minDistance2ReferencePoint = 2;
const int Train::minDistance2ReferencePointSqr = minDistance2ReferencePoint * minDistance2ReferencePoint;
const int Train::HALF_PATCH_WIDTH = 15;
const int Train::SIGMA_BLUR = 15;



double Train::rad2Deg(double rad)
{
	return rad*(180 / M_PI);
}
double Train::deg2Rad(double deg)
{
	return deg*(M_PI / 180);
}

bool Train::keyPointOrderingByResponse(const cv::KeyPoint& k1, const cv::KeyPoint& k2)
{
	return k1.response > k2.response;
}

bool Train::compareStablePoints(const StablePoint &x1, const StablePoint &x2)
{
	return x1.viewPointPts.size() > x2.viewPointPts.size();
}

void Train::warpMatrix(cv::Size sz,
	double yaw,
	double pitch,
	double roll,
	double scale,
	double fovy,
	cv::Mat &M,
	vector<cv::Point2f>* corners)
{
	double st = sin(deg2Rad(roll));
	double ct = cos(deg2Rad(roll));
	double sp = sin(deg2Rad(pitch));
	double cp = cos(deg2Rad(pitch));
	double sg = sin(deg2Rad(yaw));
	double cg = cos(deg2Rad(yaw));

	double halfFovy = fovy*0.5;
	double d = hypot(sz.width, sz.height);
	double sideLength = scale*d / cos(deg2Rad(halfFovy));
	double h = d / (2.0*sin(deg2Rad(halfFovy)));
	double n = h - (d / 2.0);
	double f = h + (d / 2.0);


	cv::Mat F = cv::Mat(4, 4, CV_64FC1);
	cv::Mat Rroll = cv::Mat::eye(4, 4, CV_64FC1);
	cv::Mat Rpitch = cv::Mat::eye(4, 4, CV_64FC1);
	cv::Mat Ryaw = cv::Mat::eye(4, 4, CV_64FC1);

	cv::Mat T = cv::Mat::eye(4, 4, CV_64FC1);
	cv::Mat P = cv::Mat::zeros(4, 4, CV_64FC1);


	Rroll.at<double>(0, 0) = Rroll.at<double>(1, 1) = ct;
	Rroll.at<double>(0, 1) = -st; Rroll.at<double>(1, 0) = st;

	Rpitch.at<double>(1, 1) = Rpitch.at<double>(2, 2) = cp;
	Rpitch.at<double>(1, 2) = -sp; Rpitch.at<double>(2, 1) = sp;

	Ryaw.at<double>(0, 0) = Ryaw.at<double>(2, 2) = cg;
	Ryaw.at<double>(0, 2) = sg; Ryaw.at<double>(2, 0) = sg;


	T.at<double>(2, 3) = -h;

	P.at<double>(0, 0) = P.at<double>(1, 1) = 1.0 / tan(deg2Rad(halfFovy));
	P.at<double>(2, 2) = -(f + n) / (f - n);
	P.at<double>(2, 3) = -(2.0*f*n) / (f - n);
	P.at<double>(3, 2) = -1.0;

	F = P*T*Rpitch*Rroll*Ryaw;

	double ptsIn[4 * 3];
	double ptsOut[4 * 3];
	double halfW = sz.width / 2, halfH = sz.height / 2;
	ptsIn[0] = -halfW; ptsIn[1] = halfH;
	ptsIn[3] = halfW; ptsIn[4] = halfH;
	ptsIn[6] = halfW; ptsIn[7] = -halfH;
	ptsIn[9] = -halfW; ptsIn[10] = -halfH;
	ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0;
	cv::Mat ptsInMat(1, 4, CV_64FC3, ptsIn); 
	cv::Mat ptsOutMat(1, 4, CV_64FC3, ptsOut);
	perspectiveTransform(ptsInMat, ptsOutMat, F);

	cv::Point2f ptsInPt2f[4]; 
	cv::Point2f ptsOutPt2f[4];
	for (int i = 0; i<4; i++)
	{
		cv::Point2f ptIn(ptsIn[i * 3 + 0], ptsIn[i * 3 + 1]);
		cv::Point2f ptOut(ptsOut[i * 3 + 0], ptsOut[i * 3 + 1]);
		ptsInPt2f[i] = ptIn + cv::Point2f(halfW, halfH);
		ptsOutPt2f[i] = (ptOut + cv::Point2f(1, 1))*(sideLength*0.5);
	}
	M = getPerspectiveTransform(ptsInPt2f, ptsOutPt2f);

	if (corners != NULL)
	{
		corners->clear();
		corners->push_back(ptsOutPt2f[0]);
		corners->push_back(ptsOutPt2f[1]);
		corners->push_back(ptsOutPt2f[2]);
		corners->push_back(ptsOutPt2f[3]);
	}
}

cv::Mat Train::keyPoint2Mat(const vector<cv::KeyPoint >& keypoints)
{
	cv::Mat stub((int)keypoints.size(), 1, CV_32FC2);
	for (unsigned int i = 0; i<keypoints.size(); i++)
		stub.at<cv::Vec2f>(i, 0) = keypoints[i].pt;
	return stub;
}

std::vector<cv::Point> Train::mat2Points(const cv::Mat& stub)
{
	std::vector<cv::Point> points(stub.rows);
	for (int i = 0; i<stub.rows; i++)
	{
		cv::Point2f pnt = stub.at<cv::Vec2f>(i, 0);
		points[i] = cv::Point(pnt.x, pnt.y);
	}
	return points;
}


void Train::projectKeypoints(const vector<cv::KeyPoint> &original, const cv::MatExpr M, vector<cv::Point> &transformedPoints)
{
	cv::Mat keypointMatIn = keyPoint2Mat(original);
	cv::Mat keypointMatOut;
	cv::perspectiveTransform(keypointMatIn, keypointMatOut, M);
	transformedPoints = mat2Points(keypointMatOut);
}


void Train::warpImage(const cv::Mat &src,
	double yaw,
	double pitch,
	double roll,
	double scale,
	double fovy,
	cv::Mat &dst,
	cv::Mat &M,
	std::vector<cv::Point2f> &corners)
{
	//Warp Image
	//Half of vertical field of view
	double halfFovy = fovy * 0.5;
	//Compute d
	double d = hypot(src.cols, src.rows);
	//Compute side length of square
	double sideLength = scale * d / cos(deg2Rad(halfFovy));
	//Compute warp matrix and set vector of corners
	warpMatrix(src.size(), yaw, pitch, roll, scale, fovy, M, &corners);
	//Perform actual warp to finish the method
	cv::warpPerspective(src, dst, M, cv::Size(sideLength, sideLength));
}

void Train::trainFromReferenceImage(cv::Mat &image, ViewParams &params, std::vector<std::vector<StablePoint> > &database)
{
	CV_Assert(image.channels() == 1);

	cv::RNG rng(0xFFFFFFFF);
	int viewPointBin = 0, totalBins = 0;
	totalBins = ((int)floor(1.0 + abs(log(params.scale.min / params.scale.max) / log(params.scale.step)))) *
		((int)ceil(abs((params.yaw.max - params.yaw.min) / params.yaw.step))) *
		((int)ceil(abs((params.pitch.max - params.pitch.min) / params.pitch.step))) *
		((int)ceil(abs((params.roll.max - params.roll.min) / params.roll.step)));

	double rYaw, rPitch, rRoll;
	cv::namedWindow("Keypoints", cv::WINDOW_NORMAL);
	cv::Ptr<cv::FeatureDetector> detector = Train::create(params.detector);
	cv::Ptr<cv::DescriptorExtractor> extractor = Train::create(params.extractor);

	// for each step of rotation in yaw
	for (double yaw = params.yaw.min; yaw < params.yaw.max; yaw += params.yaw.step)
	{
		std::cout << "yaw: " << yaw << '\n';
		// for each step of rotation in pitch
		for (double pitch = params.pitch.min; pitch < params.pitch.max; pitch += params.pitch.step)
		{
			std::cout << "pitch: " << pitch << '\n';
			// for each step of rotation rotation in roll
			for (double roll = params.roll.min; roll < params.roll.max; roll += params.roll.step)
			{
				std::cout << "roll: " << roll << '\n';
				// for each scale of a specific viewing angle (yaw, pitch, roll)
				for (double scale = params.scale.max; scale >= params.scale.min; scale *= params.scale.step)
				{
					viewPointBin++;
					printf("Training %d / %d\n", viewPointBin, totalBins);

					std::vector<cv::Mat> /*images, */projMatrices;

					std::vector<std::vector<cv::KeyPoint>> keyPoints;
					std::vector<std::vector<uint16_t>> indxs;
					std::vector<cv::Mat> descriptors;
					std::vector<std::vector<cv::Point2f> > vecCorners;

					// perform x amount of random rotation (within the range of step)
					for (int viewPoint = 0; viewPoint < params.numberOfViewPoints; viewPoint++)
					{
						cv::Mat warpBlurred, warped, projecMatrix;

						std::vector<cv::Point2f> corners;

						// randomly rotate by the amount of each rotation parameter up to the maximum "step"
						rPitch = rng.uniform(pitch, pitch + params.pitch.step);
						rYaw = rng.uniform(yaw, yaw + params.yaw.step);
						rRoll = rng.uniform(roll, roll + params.roll.step);

						// Warp the image
						warpImage(image, rYaw, rPitch, rRoll,
							scale, params.fov, warped, projecMatrix, corners);

						warped.copyTo(warpBlurred);
						
						// Blur the image
						GaussianBlur(warped, warpBlurred,
							cv::Size(3, 3), SIGMA_BLUR, SIGMA_BLUR); //scale*sigma_blur


						std::vector<cv::KeyPoint> kps;
						std::vector<uint16_t> idx;
						cv::Mat desc;

						// Perform binary feature detection (e.g. FAST) and remove the invalid points (i.e. out of image corner)
						selectPoints(warpBlurred,
							projecMatrix,
							corners,
							image.size(),
							detector,
							extractor,
							kps,
							idx,
							desc);


						// Debug: Show output
						cv::drawKeypoints(warpBlurred, kps, warpBlurred);
						cv::imshow("Keypoints", warpBlurred);
						cv::waitKey(1);
						//std::cout << warpBlurred.size() << '\n';

						// Generate the database for later on stable feature point selection
						keyPoints.push_back(kps);				// keypoints: pts, size(diameter)=7 by default, response
						descriptors.push_back(desc);			// 32 (bytes) by num_features (feature index) descriptor
						indxs.push_back(idx);					// 1 by num_features of feature indeice (all 0 for now)
						projMatrices.push_back(projecMatrix);	// projection matrix of randomly select viewing angle
						vecCorners.push_back(corners);			// image rectangle corner
					}

					std::vector<StablePoint> viewStablePts;

					selectMostStableKeypoints(keyPoints,
						indxs,
						descriptors,
						projMatrices,
						vecCorners,
						params.numberOfPointsPerViewPoints,
						image.size(),
						detector,
						extractor,
						viewStablePts);

					

					database.push_back(viewStablePts);

					//free memory
					keyPoints.clear();
					indxs.clear();
					descriptors.clear();
					vecCorners.clear();
					projMatrices.clear();
					//std::cout << .size() << '\n';
				}
			}
		}
	}
}

bool Train::sortByProbNearToDot5(std::pair<int, float> &first, std::pair<int, float> &second)
{
	float f = std::abs(.5 - first.second);
	float s = std::abs(.5 - second.second);
	return f < s;
}

/*
	database: training dataset contains all the possible viewpoints
	size: size of the histogram, default is 13
*/
vector<uint16_t> Train::computeDescIndices(vector<vector<StablePoint>> &database, int size)
{
	std::map<int, float> bitHistograms;
	int totalDescriptors = 0;
	//For all list of stable points
	for (int i = 0; i < database.size(); ++i)
	{
		//Go through all the stable points
		for (int j = 0; j < database[i].size(); ++j)
		{
			cv::Mat descs = database[i][j].viewPointPtDescriptors;
			//All the descriptors associated to a point
			for (int r = 0; r < descs.rows; ++r)
			{
				uchar *row = descs.ptr<uchar>(r);
				//For every single col value in the descriptor
				for (int c = 0; c < descs.cols; c++)
				{
					//Take the bits positions
					for (int bit = 7; bit >= 0; --bit)
					{
						uchar tmp = (1 << bit);
						if ((row[c] & tmp) > 0)
						{
							// Creating a histogram of all 256 testing results from the entire database
							bitHistograms[c * 8 + (7 - bit)]++;
						}
					}
				}
				totalDescriptors++;
			}
		}
	}
	typedef std::map<int, float>::iterator iter;
	vector< pair<int, float> > result;
	for (iter it = bitHistograms.begin(); it != bitHistograms.end(); ++it)
	{
		it->second /= (float)totalDescriptors;
		result.push_back(pair<int, float>(it->first, it->second));
	}
	sort(result.begin(), result.end(), Train::sortByProbNearToDot5);
	vector<uint16_t> indices;
	int i = 0;
	typedef std::vector<pair<int, float>>::iterator iter2;
	for (iter2 it = result.begin(); it != result.end() && i < size; ++it, ++i)
	{
		indices.push_back(it->first);
	}
	sort(indices.begin(), indices.end());
	return indices;
}

void Train::filterKeyPoints(cv::Size targetImgSize, const vector<cv::Point2f> &corners, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Point> &pointsTransformed)
{
	vector<cv::Point> newPointsTransformed;
	newPointsTransformed.reserve(pointsTransformed.size());
	vector<cv::KeyPoint> newKeypoints;
	newKeypoints.reserve(keypoints.size());

	for (unsigned int i = 0; i<keypoints.size(); i++)
	{
		cv::Point& p = pointsTransformed[i];
		cv::KeyPoint& k = keypoints[i];
		//reject if transformed point is too close to the edge
		if (p.x < HALF_PATCH_WIDTH ||
			p.y < HALF_PATCH_WIDTH ||
			p.x >= targetImgSize.width - HALF_PATCH_WIDTH ||
			p.y >= targetImgSize.height - HALF_PATCH_WIDTH)
			continue;

		//reject if keypoint is too close to edge of the warped image
		if (cv::pointPolygonTest(corners, cv::Point2f(k.pt.x - HALF_PATCH_WIDTH, k.pt.y - HALF_PATCH_WIDTH), false) < 0 ||
			cv::pointPolygonTest(corners, cv::Point2f(k.pt.x - HALF_PATCH_WIDTH, k.pt.y + HALF_PATCH_WIDTH), false) < 0 ||
			cv::pointPolygonTest(corners, cv::Point2f(k.pt.x + HALF_PATCH_WIDTH, k.pt.y - HALF_PATCH_WIDTH), false) < 0 ||
			cv::pointPolygonTest(corners, cv::Point2f(k.pt.x + HALF_PATCH_WIDTH, k.pt.y + HALF_PATCH_WIDTH), false) < 0)
			continue;

		newPointsTransformed.push_back(p);
		newKeypoints.push_back(k);
	}
	//replace old points with new points
	pointsTransformed.clear();
	keypoints.clear();
	pointsTransformed = newPointsTransformed;
	keypoints = newKeypoints;

	CV_Assert(keypoints.size() == pointsTransformed.size());
	return;
}


void Train::selectPoints(cv::Mat &image, cv::Mat &M, vector<cv::Point2f> &corners, cv::Size refImgSize, cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &extractor, vector<cv::KeyPoint> &kps, vector<uint16_t> &indxs, cv::Mat &desc)
{
	unsigned int MAX_KEYPOINTS = 1000;
	detector->detect(image, kps);
	vector<cv::Point> bpKps;
	sort(kps.begin(), kps.end(), keyPointOrderingByResponse);
	kps.resize(std::min((unsigned int)MAX_KEYPOINTS, (unsigned int)kps.size()));
	projectKeypoints(kps, M.inv(), bpKps);
	filterKeyPoints(refImgSize, corners, kps, bpKps);



	extractor->compute(image, kps, desc);
	projectKeypoints(kps, M.inv(), bpKps);
	//for (int i = 0; i < kps.size(); i++)
	//{
	//	std::cout << kps.at(i).size << '\n';
	//}
	indxs.resize(kps.size(), 0);
}

int Train::getNearbyPoint(const cv::Point& p, const vector<vector<int> > &grid)
{
	int i, j, idx = grid[p.y][p.x];
	// Set initial threshold to one more than the acceptable threshold
	int dist, minDist = minDistance2ReferencePointSqr + 1;
	//the center point was already an existing point, so return its index
	if (idx != -1)
		return idx;

	/* Changed because the limits were being re-evaluated each time with the stdmin, and because
	* the distance is evaluated as distance squared the results are always integral, so floating
	* point is not needed.
	*/

	int xmin, ymin, xmax, ymax;
	xmin = std::max(p.x - minDistance2ReferencePoint, 0);
	ymin = std::max(p.y - minDistance2ReferencePoint, 0);
	xmax = std::min(p.x + minDistance2ReferencePoint, (int)grid.size() - 1);
	ymax = std::min(p.y + minDistance2ReferencePoint, (int)grid[0].size() - 1);

	for (i = ymin; i <= ymax; i++)
	{
		for (j = xmin; j <= xmax; j++)
		{
			dist = (p.y - i)*(p.y - i) + (p.x - j)*(p.x - j);

			if ((dist < minDist) && (grid[i][j] != -1))
			{
				minDist = dist; 
				idx = grid[i][j];
			}
		}
	}
	return idx;
}

void Train::selectMostStableKeypoints(const vector<vector<cv::KeyPoint>>& keypoints, const vector<vector<uint16_t>> &indices, const vector<cv::Mat> &descriptors, const vector<cv::Mat>&transfMatrix, const vector < vector < cv::Point2f > > & corners, int maxPointsPerView, cv::Size refImgSize, cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &extractor, vector<StablePoint> &bestPoints)
{
	// initalise all elements with value -1
	vector<vector<int> > grid(refImgSize.height, vector<int>(refImgSize.width, -1));

	// for each element of the vector of cv::KeyPoints
	for (int i = 0; i < keypoints.size(); ++i)
	{
		// for each viewpoint, get all the keypoints, backprojected to its original view, and grab the descriptor as well.

		// back projected all the viewKeyPoints vector
		const vector<cv::KeyPoint> &viewKeyPoints = keypoints[i];
		vector<cv::Point> backProjPoints;
		projectKeypoints(viewKeyPoints, transfMatrix[i].inv(), backProjPoints);
		
		
		const cv::Mat &viewDescriptors = descriptors[i];
		const vector<uint16_t> idxs = indices[i];
		
		// for each back projected point, we want to know whether each keypoint is roughly mapping to the same back
		// projected point

		// for each back projected point of cv::KeyPoints in the vector
		//find the points and update them
		for (int j = 0; j < backProjPoints.size(); j++)
		{
			cv::Point &p = backProjPoints[j];
			const cv::KeyPoint &k = viewKeyPoints[j];
			cv::Mat d = viewDescriptors.row(j);		// each row contains the corresponding descriptor
			const uint16_t index = idxs[j];			// each index contains the corresponding feature point index

			int idx = getNearbyPoint(p, grid);
			
			
			//no nearby model found, we have a new stable point
			if (idx == -1)
			{
				//add new point
				// i: imageIndex of the current viewing orientation
				// p: corresponding back projected keypoint at imageIndex[i]
				// k: corresponding detcted keypoint at imageIndex[i]
				// d: corresponding descriptor at imageIndex[i]
				// index: 
				bestPoints.push_back(StablePoint(p, k, i, d, index));
				idx = (int)bestPoints.size() - 1;
				grid[p.y][p.x] = idx;
			}
			// otherwise, update existing point about the current keypoint that maps to same existing location in the original view.
			else
			{
				bestPoints[idx].viewPointPts.push_back(k);
				bestPoints[idx].imageNumber.push_back(i);
				bestPoints[idx].viewPointPtDescriptors.push_back(d);
				bestPoints[idx].indices.push_back(index);
				continue;
			}
		}
	}


	/*
	* We now sort bestPoints based on the number of votes for each refPnt and select the TOP-N points
	*/

	sort(bestPoints.begin(), bestPoints.end(), compareStablePoints);
	bestPoints.resize(std::min((unsigned int)maxPointsPerView,
		(unsigned int)bestPoints.size()));

	//bestPoints.clear();
	// recycle memory
	for (unsigned int i = 0; i<grid.size(); i++)
		grid[i].clear();
	grid.clear();
	return;
}

void Train::computeIndices(vector<vector<StablePoint>> &database, vector<uint16_t> &indices)
{
	//For all list of stable points
	for (int i = 0; i < database.size(); ++i)
	{
		//Go through all the stable points
		for (int j = 0; j < database[i].size(); ++j)
		{
			cv::Mat descs = database[i][j].viewPointPtDescriptors;
			//All the descriptors associated to a point
			for (int r = 0; r < descs.rows; ++r)
			{
				cv::Mat desc = descs.row(r);
				database[i][j].indices[r] = Index::getDescIndex(desc, indices);
			}
		}
	}
}

std::string Train::featureName(Feat::Code code)
{
	return Feat::getName(code);
}

std::string Train::descriptorName(Desc::Code code)
{
	return Desc::getName(code);
}

cv::Ptr<cv::FeatureDetector>  Train::create(Feat::Code code)
{
	return Feat::create(code);
}

cv::Ptr<cv::DescriptorExtractor> Train::create(Desc::Code code)
{
	return Desc::create(code);
}

