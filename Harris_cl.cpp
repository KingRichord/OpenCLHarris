//
// Created by moi on 23-6-8.
//

#include "Hariis_cl.h"
Harris_CL::Harris_CL(cv::Size2f image_size) {
	cv::Mat
	if(opencl_enable()) {
		smooth_kernel = boost::compute::kernel(filter_program, "smoothing");
		diffx_kernel = boost::compute::kernel(filter_program, "diffx");
		diffy_kernel = boost::compute::kernel(filter_program, "diffy");
		struct_kernel = boost::compute::kernel(filter_program, "structure");
		nms_kernel = boost::compute::kernel(filter_program, "NonMaxSuppression");
		
		smoothing_image = boost::compute::image2d(
				context,
				input_image.width(),
				input_image.height(),
				input_image.format(),
				boost::compute::image2d::read_write
		);
		diffx = boost::compute::image2d(
				context,
				input_image.width(),
				input_image.height(),
				input_image.format(),
				boost::compute::image2d::read_write
		);
		
		diffy = boost::compute::image2d(
				context,
				input_image.width(),
				input_image.height(),
				input_image.format(),
				boost::compute::image2d::read_write
		);
		structure_image = boost::compute::image2d(
				context,
				input_image.width(),
				input_image.height(),
				input_image.format(),
				boost::compute::image2d::read_write
		);
		nms_image = boost::compute::image2d(
				context,
				input_image.width(),
				input_image.height(),
				input_image.format(),
				boost::compute::image2d::read_write
		);
	}
}
void Harris_CL::compute() {




}