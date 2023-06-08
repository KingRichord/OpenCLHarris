#include "opencv2/opencv.hpp"
#include "Featuredetectors.h"
#ifndef OPENCL_HARRIS_TEST_HARIIS_CL_H
#define OPENCL_HARRIS_TEST_HARIIS_CL_H
class Harris_CL :public FeatureDetector {
public:
	Harris_CL(cv::Size2f image_size);
	~Harris_CL();
	virtual void compute();
private:
	float* filter;
	
	
	boost::compute::kernel smooth_kernel;
	boost::compute::kernel diffx_kernel;
	boost::compute::kernel diffy_kernel;
	boost::compute::kernel struct_kernel;
	boost::compute::kernel nms_kernel;
	
	
	boost::compute::image2d input_image;
	boost::compute::image2d smoothing_image;
	boost::compute::image2d diffx;
	boost::compute::image2d diffy;
	boost::compute::image2d structure_image;
	boost::compute::image2d nms_image;
};


#endif //OPENCL_HARRIS_TEST_HARIIS_CL_H
