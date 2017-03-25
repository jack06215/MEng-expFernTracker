#include "index.h"


uint16_t Index::getDescIndex(const Mat& img, const vector<uint16_t> &positions)
{
	// data is a 32 elements of 8 bits feature descriptor associated with a particular keypoint.
	uchar *data = img.data;
	uint16_t index = 0;
	int size = (int)positions.size();

	// for most frequently appearing index in the histogram, go to its corresponding elements (of each descriptor row) and its bit position
	// the result of this will compress a 32 * 8 bit descriptor to a uint_16 value
	for (int i = 0; i < size; i++)
	{
		uchar mask = 1 << positions[i] % CHAR_BIT; 		// shift mask to matches the corresponding bit position (from 0-7)       
		index |= (data[positions[i] / CHAR_BIT] & mask) ? 1 << i : 0; 		// at {position[i] / 8} element of {mask} bit, assign a '1' to that bit position or 0 otherwise (unchnaged) 

	}
	return index;
}