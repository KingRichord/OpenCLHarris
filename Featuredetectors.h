#ifndef OPENCL_HARRIS_TEST_FEATUREDETECTOR_H
#define OPENCL_HARRIS_TEST_FEATUREDETECTOR_H
#include "boost/compute.hpp"
class FeatureDetector {
public:
	FeatureDetector(bool use_opencl= false);
	~FeatureDetector();
	bool opencl_enable() {return is_use_opencl_;}
	void compile(const char* code_txt);
	virtual void compute();
	
	boost::compute::device gpu = boost::compute::system::default_device();
	boost::compute::context context;
	boost::compute::command_queue queue;
	boost::compute::program filter_program;
	
private:
	bool is_use_opencl_;
};
#endif
