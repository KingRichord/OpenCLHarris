// Runs non-maximal suppression with a global minimum threshold
__kernel void NonMaxSuppression (
    __read_only image2d_t src,
    __constant float* src_max,
    __write_only image2d_t dest) {

    float threshold = src_max[0] * THRESHOLD_RATIO;
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const float4 max = read_imagef(src, clamp_sampler, pos);

    if (max.x < threshold) {
        write_imagef(dest, pos, (float4)0.0f);
        return;
    }

    for (int y = -HALF_SUPPRESSION; y <= HALF_SUPPRESSION; y++) {
        for (int x = -HALF_SUPPRESSION; x <= HALF_SUPPRESSION; ++x) {
            const float4 r = read_imagef(src, reflect_sampler, pos + (int2)(x,y));
            if (r.x > max.x) {
                write_imagef(dest, pos, (float4)0.0f);
                return;
            }
        }
    }

    write_imagef(dest, pos, max);
}