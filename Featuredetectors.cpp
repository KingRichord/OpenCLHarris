//
// Created by moi on 23-6-8.
//
#include "Featuredetectors.h"
FeatureDetector::FeatureDetector(bool use_opencl) :is_use_opencl_(use_opencl){
	if(gpu.name().empty()) {
		std::cout << " No computing equipment available" << std::endl;
		use_opencl = false;
	}
	else
	{
		std::cout << gpu.name() << std::endl;
		use_opencl = true;
	}
	if(use_opencl)
	{
		context = boost::compute::context(gpu);
		queue = boost::compute::command_queue(context, gpu);
	}
}
void FeatureDetector::compile(const char* code_txt)
{
	filter_program = boost::compute::program::create_with_source(code_txt, context);
	try
	{
		filter_program.build();
	}
	catch(boost::compute::opencl_error e)
	{
		std::cout<<"Build Error: "<<std::endl <<filter_program.build_log();
	}
}