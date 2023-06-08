#include <iostream>
#include <opencv2/opencv.hpp>
#include "boost/compute.hpp"
#include <boost/compute/interop/opencv/core.hpp>
#include <boost/compute/interop/opencv/highgui.hpp>
// Create sobel filter program
const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE (
		const sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		kernel void diffx (
				read_only image2d_t src,
				write_only image2d_t dest)
		{
			const int2 pos = {get_global_id(0), get_global_id(1)};
			float4 sum = read_imagef(src, reflect_sampler, pos - (int2)(1,0)) - read_imagef(src, reflect_sampler, pos + (int2)(1,0));
			write_imagef(dest, (int2)(pos.x, pos.y), sum);
		}
		// Computes dy image
		kernel void diffy (
				read_only image2d_t src,
				write_only image2d_t dest)
		{
			const int2 pos = {get_global_id(0), get_global_id(1)};
			float4 sum = read_imagef(src, reflect_sampler, pos - (int2)(0,1)) - read_imagef(src, reflect_sampler, pos + (int2)(0,1));
            write_imagef (dest, (int2)(pos.x, pos.y), sum);
		}
		kernel void smoothing (
				read_only image2d_t src,
				constant float* filterWeights,
				const float half_smoothing,
				write_only image2d_t dest) {
			
			const int2 pos = {get_global_id(0), get_global_id(1)};
			// printf("  %f  ",read_imagef(src, reflect_sampler, pos).x);
			float4 sum = (float4)(0.0f);
			int i = 0;
			for (int y = -half_smoothing; y <= half_smoothing; ++y) {
				for (int x = -half_smoothing; x <= half_smoothing; ++x) {
					sum += filterWeights[i++] * read_imagef(src, reflect_sampler, pos + (int2)(x,y));
				}
			}
			write_imagef(dest, pos, sum);
		}
		// Computes the structure tensor image from dx and dy images
		kernel void structure (
				read_only image2d_t i_x,
				read_only image2d_t i_y,
				constant float* filterWeights,
				write_only image2d_t dest)
		{
			int x = (int)get_global_id(0);
			int y = (int)get_global_id(1);
			const int2 pos = {x, y};
			float4 s = (float4)(0.0f);
			int i = 0;
			for (int y = -1; y <= 1; ++y) {
				for (int x = -1; x <= 1; ++x) {
					const int2 window_pos = pos + (int2)(x,y);
					const float s_x = read_imagef(i_x, reflect_sampler, window_pos).x * filterWeights[i];
					const float s_y = read_imagef(i_y, reflect_sampler, window_pos).x * filterWeights[i];
					s.x += s_x * s_x;
					s.y += s_y * s_y;
					s.z += s_x * s_y;
					i++;
				}
			}
            const float4 r;
            r.x = (s.x * s.y - s.z * s.z) - 0.04f * (s.x + s.y) * (s.x + s.y);
			write_imagef(dest, pos, r);
		}
		// Runs non-maximal suppression with a global minimum threshold
		kernel void NonMaxSuppression (
				read_only image2d_t src,
				const float src_max,
				const float half_suppression,
				write_only image2d_t dest,
				global int2 *data) {
			
			float threshold = src_max * 1.f;
			
			int x = (int)get_global_id(0);
			int y = (int)get_global_id(1);
			const int2 pos = {x, y};
			const float4 max = read_imagef(src, reflect_sampler, pos);
			
			if (max.x < threshold) {
				write_imagef(dest, pos, (float4)0.0f);
				return;
			}
			for (int y = -half_suppression; y <= half_suppression; y++) {
				for (int x = -half_suppression; x <= half_suppression; ++x) {
					const float4 r = read_imagef(src, reflect_sampler, pos + (int2)(x,y));
					if (r.x > max.x) {
						write_imagef(dest, pos, (float4)0.0f);
						return;
					}
				}
			}
			data[y*x+x] = pos;
			write_imagef(dest, pos, (float4)255.f);
		}
);


int main() {
	
	float min_th = 1000. ;  //最大响应值阈值
	float num_radius = 5;

	float filter[9] =  {
			0.05,      0.1,      0.05,
			0.1,       0.4,       0.1,
			0.05,      0.1,      0.05,
	};
	

	std::cout << "Hello, World!" << std::endl;
    boost::compute::device gpu = boost::compute::system::default_device();
	std::cout <<gpu.name() <<std::endl;
	boost::compute::context context(gpu);
    boost::compute::command_queue queue(context, gpu);
    boost::compute::program filter_program =
            boost::compute::program::create_with_source(source, context);
    try
    {
        filter_program.build();
    }
    catch(boost::compute::opencl_error e)
    {
        std::cout<<"Build Error: "<<std::endl <<filter_program.build_log();
    }
	boost::compute::kernel smooth_kernel(filter_program, "smoothing");
	boost::compute::kernel diffx_kernel(filter_program, "diffx");
	boost::compute::kernel diffy_kernel(filter_program, "diffy");
	boost::compute::kernel struct_kernel(filter_program, "structure");
	boost::compute::kernel nms_kernel(filter_program, "NonMaxSuppression");
	cv::Mat cv_mat;
    cv_mat = cv::imread("../0.png", cv::IMREAD_GRAYSCALE);
	cv::Mat input;
	cv_mat.convertTo(input, CV_32F);
	cv::imshow("origin",cv_mat);
	boost::compute::buffer dev_filter(context, sizeof(filter),
	                                  boost::compute::memory_object::read_only |
	                                  boost::compute::memory_object::copy_host_ptr,filter);
	size_t steps = input.cols* input.rows;
	std::vector<boost::compute::int2_> h_output(steps);
	boost::compute::vector<boost::compute::int2_> all_corner_pos(steps, context);
	boost::compute::fill(all_corner_pos.begin(), all_corner_pos.end(), (boost::compute::int2_)(-1), queue);
	// 将图像数据转换到GPU中
    boost::compute::image2d input_image =
            boost::compute::opencv_create_image2d_with_mat(
		            input,  boost::compute::image2d::read_only, queue);
	boost::compute::image2d smoothing_image(
			context,
			input_image.width(),
			input_image.height(),
			input_image.format(),
			boost::compute::image2d::read_write
	);
    boost::compute::image2d diffx(
            context,
            input_image.width(),
            input_image.height(),
            input_image.format(),
            boost::compute::image2d::read_write
    );
	
	boost::compute::image2d diffy(
			context,
			input_image.width(),
			input_image.height(),
			input_image.format(),
			boost::compute::image2d::read_write
	);
	boost::compute::image2d structure_image(
			context,
			input_image.width(),
			input_image.height(),
			input_image.format(),
			boost::compute::image2d::read_write
	);
	boost::compute::image2d nms_image(
			context,
			input_image.width(),
			input_image.height(),
			input_image.format(),
			boost::compute::image2d::read_write
	);
	
	
	size_t origin[2] = { 0, 0 };
	size_t region[2] = { input_image.width(),
	                     input_image.height() };
	
	diffx_kernel.set_arg(0, input_image);
	diffx_kernel.set_arg(1, diffx);
    queue.enqueue_nd_range_kernel(diffx_kernel, 2, origin, region, 0);
	// boost::compute::opencv_imshow("diffx_kernel Image", diffx, queue);
	diffy_kernel.set_arg(0, input_image);
	diffy_kernel.set_arg(1, diffy);
	queue.enqueue_nd_range_kernel(diffy_kernel, 2, origin, region, 0);
	
	struct_kernel.set_arg(0, diffx);
	struct_kernel.set_arg(1, diffy);
	struct_kernel.set_arg(2, dev_filter);
	struct_kernel.set_arg(3, structure_image);
	queue.enqueue_nd_range_kernel(struct_kernel, 2, origin, region, 0);
	// boost::compute::opencv_imshow("structure_image Image", structure_image, queue);

	 nms_kernel.set_arg(0, structure_image);
	 nms_kernel.set_arg(1, min_th);
	 nms_kernel.set_arg(2, num_radius);
	 nms_kernel.set_arg(3, nms_image);
	 nms_kernel.set_arg(4, all_corner_pos.get_buffer());
	 queue.enqueue_nd_range_kernel(nms_kernel, 2, origin, region, 0);
	 boost::compute::copy(all_corner_pos.begin(), all_corner_pos.end(), h_output.begin(), queue);
	cv::Mat result;
	cv::cvtColor(cv_mat, result, cv::COLOR_GRAY2RGB);
	for (auto & i : h_output) {
		if(i.x != -1||i.y != -1)
			circle(result, cv::Point2i (i.x,i.y), 4, cv::Scalar(0, 0, 255), 1);
	}
	cv::imshow("positions", result);
	// boost::compute::opencv_imshow("nms_image Image", nms_image, queue);
    cv::waitKey(0);
    return 0;
}