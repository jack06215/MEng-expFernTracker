#ifndef detection3D_hest_h
#define detection3D_hest_h

#ifdef __cplusplus
extern "C" {
#endif
	int hestRefC(float*   src,
		float*   dst,
		unsigned N,
		float    maxD,
		unsigned maxI,
		unsigned rConvg,
		double   cfd,
		unsigned minInl,
		float*   finalH);

	int hestSSE(float*   src,
		float*   dst,
		unsigned N,
		float    maxD,
		unsigned maxI,
		unsigned rConvg,
		double   cfd,
		unsigned minInl,
		float*   finalH);
#ifdef __cplusplus
}
#endif
#endif
