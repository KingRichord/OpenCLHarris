// OpenCL port 

inline int cornerScore(__global const uchar* img, int step)
{
    int k, tofs, v = img[0], a0 = 0, b0;
    int d[16];
    #define LOAD2(idx, ofs) \
        tofs = ofs; d[idx] = (short)(v - img[tofs]); d[idx+8] = (short)(v - img[-tofs])
    LOAD2(0, 3);
    LOAD2(1, -step+3);
    LOAD2(2, -step*2+2);
    LOAD2(3, -step*3+1);
    LOAD2(4, -step*3);
    LOAD2(5, -step*3-1);
    LOAD2(6, -step*2-2);
    LOAD2(7, -step-3);

    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int a = min((int)d[(k+1)&15], (int)d[(k+2)&15]);
        a = min(a, (int)d[(k+3)&15]);
        a = min(a, (int)d[(k+4)&15]);
        a = min(a, (int)d[(k+5)&15]);
        a = min(a, (int)d[(k+6)&15]);
        a = min(a, (int)d[(k+7)&15]);
        a = min(a, (int)d[(k+8)&15]);
        a0 = max(a0, min(a, (int)d[k&15]));
        a0 = max(a0, min(a, (int)d[(k+9)&15]));
    }

    b0 = -a0;
    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int b = max((int)d[(k+1)&15], (int)d[(k+2)&15]);
        b = max(b, (int)d[(k+3)&15]);
        b = max(b, (int)d[(k+4)&15]);
        b = max(b, (int)d[(k+5)&15]);
        b = max(b, (int)d[(k+6)&15]);
        b = max(b, (int)d[(k+7)&15]);
        b = max(b, (int)d[(k+8)&15]);

        b0 = min(b0, max(b, (int)d[k]));
        b0 = min(b0, max(b, (int)d[(k+9)&15]));
    }

    return -b0-1;
}

__kernel void FAST_findKeypoints(
    __global const uchar * _img, int step, int img_offset,
    int img_rows, int img_cols,
    volatile __global int* kp_loc,
    int max_keypoints, int threshold )
{
    int j = get_global_id(0) + 3;
    int i = get_global_id(1) + 3;

    if (i < img_rows - 3 && j < img_cols - 3)
    {
        __global const uchar* img = _img + mad24(i, step, j + img_offset);
        int v = img[0], t0 = v - threshold, t1 = v + threshold;
        int k, tofs, v0, v1;
        int m0 = 0, m1 = 0;

        #define UPDATE_MASK(idx, ofs) \
            tofs = ofs; v0 = img[tofs]; v1 = img[-tofs]; \
            m0 |= ((v0 < t0) << idx) | ((v1 < t0) << (8 + idx)); \
            m1 |= ((v0 > t1) << idx) | ((v1 > t1) << (8 + idx))

        UPDATE_MASK(0, 3);
        if( (m0 | m1) == 0 )
            return;

        UPDATE_MASK(2, -step*2+2);
        UPDATE_MASK(4, -step*3);
        UPDATE_MASK(6, -step*2-2);

        #define EVEN_MASK (1+4+16+64)

        if( ((m0 | (m0 >> 8)) & EVEN_MASK) != EVEN_MASK &&
            ((m1 | (m1 >> 8)) & EVEN_MASK) != EVEN_MASK )
            return;

        UPDATE_MASK(1, -step+3);
        UPDATE_MASK(3, -step*3+1);
        UPDATE_MASK(5, -step*3-1);
        UPDATE_MASK(7, -step-3);
        if( ((m0 | (m0 >> 8)) & 255) != 255 &&
            ((m1 | (m1 >> 8)) & 255) != 255 )
            return;

        m0 |= m0 << 16;
        m1 |= m1 << 16;

        #define CHECK0(i) ((m0 & (511 << i)) == (511 << i))
        #define CHECK1(i) ((m1 & (511 << i)) == (511 << i))

        if( CHECK0(0) + CHECK0(1) + CHECK0(2) + CHECK0(3) +
            CHECK0(4) + CHECK0(5) + CHECK0(6) + CHECK0(7) +
            CHECK0(8) + CHECK0(9) + CHECK0(10) + CHECK0(11) +
            CHECK0(12) + CHECK0(13) + CHECK0(14) + CHECK0(15) +

            CHECK1(0) + CHECK1(1) + CHECK1(2) + CHECK1(3) +
            CHECK1(4) + CHECK1(5) + CHECK1(6) + CHECK1(7) +
            CHECK1(8) + CHECK1(9) + CHECK1(10) + CHECK1(11) +
            CHECK1(12) + CHECK1(13) + CHECK1(14) + CHECK1(15) == 0 )
            return;

        {
            int idx = atomic_inc(kp_loc);
            if( idx < max_keypoints )
            {
                kp_loc[1 + 2*idx] = j;
                kp_loc[2 + 2*idx] = i;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// nonmaxSupression

__kernel
void FAST_nonmaxSupression(
    __global const int* kp_in, volatile __global int* kp_out,
    __global const uchar * _img, int step, int img_offset,
    int rows, int cols, int counter, int max_keypoints)
{
    const int idx = get_global_id(0);

    if (idx < counter)
    {
        int x = kp_in[1 + 2*idx];
        int y = kp_in[2 + 2*idx];
        __global const uchar* img = _img + mad24(y, step, x + img_offset);

        int s = cornerScore(img, step);

        if( (x < 4 || s > cornerScore(img-1, step)) +
            (y < 4 || s > cornerScore(img-step, step)) != 2 )
            return;
        if( (x >= cols - 4 || s > cornerScore(img+1, step)) +
            (y >= rows - 4 || s > cornerScore(img+step, step)) +
            (x < 4 || y < 4 || s > cornerScore(img-step-1, step)) +
            (x >= cols - 4 || y < 4 || s > cornerScore(img-step+1, step)) +
            (x < 4 || y >= rows - 4 || s > cornerScore(img+step-1, step)) +
            (x >= cols - 4 || y >= rows - 4 || s > cornerScore(img+step+1, step)) == 6)
        {
            int new_idx = atomic_inc(kp_out);
            if( new_idx < max_keypoints )
            {
                kp_out[1 + 3*new_idx] = x;
                kp_out[2 + 3*new_idx] = y;
                kp_out[3 + 3*new_idx] = s;
            }
        }
    }
}


///////gaussBlur 5x5
__kernel void cl_img_gaussian_blur(__global const uchar* imgbuf, int imgstep, int imgoffset,
  __global uchar *outbuf, 
  __global const uchar *gboxbuf,
  uint img_rows, uint img_cols)
{
    int i, j, offset=2;
    int x, y, summ;

    y = get_global_id(0);//row
    x = get_global_id(1);//col

    __global const uchar* _img = imgbuf + mad24(y, imgstep, x + imgoffset);//x,y

    /* ignore border pixels*/
    if (y - offset < 0 || y + offset > img_rows || x - offset < 0 || x + offset > img_cols) {
        outbuf[y*imgstep+x] =  _img[0];
        return;
    }

    summ = 0;

    for (j = -offset; j <= offset; j++) {   //row -2 2
        for (i = -offset; i <= offset; i++) {   //  //col
            summ+=_img[mad24(j,imgstep,i)] * gboxbuf[(j+offset)*5 + i+offset];
        }
    }

    outbuf[y*imgstep+x] = summ/331;//
}


// OpenCL port of the ORB feature detector and descriptor extractor
// Copyright (C) 2014, Itseez Inc. See the license at http://opencv.org
//
// The original code has been contributed by Peter Andreas Entschev, peter@entschev.com

#define LAYERINFO_SIZE 1
#define LAYERINFO_OFS 0
#define KEYPOINT_SIZE 3
#define ORIENTED_KEYPOINT_SIZE 4
#define KEYPOINT_X 0
#define KEYPOINT_Y 1
#define KEYPOINT_Z 2
#define KEYPOINT_ANGLE 3

/////////////////////////////////////////////////////////////

#ifdef ORB_RESPONSES

__kernel void
ORB_HarrisResponses(__global const uchar* imgbuf, int imgstep, int imgoffset0,
                    __global const int* layerinfo, __global const int* keypoints,
                    __global float* responses, int nkeypoints )
{
    int idx = get_global_id(0);
    if( idx < nkeypoints )
    {
        __global const int* kpt = keypoints + idx*KEYPOINT_SIZE;
        __global const int* layer = layerinfo + kpt[KEYPOINT_Z]*LAYERINFO_SIZE;
        __global const uchar* img = imgbuf + imgoffset0 + layer[LAYERINFO_OFS] +
            (kpt[KEYPOINT_Y] - blockSize/2)*imgstep + (kpt[KEYPOINT_X] - blockSize/2);

        int i, j;
        int a = 0, b = 0, c = 0;
        for( i = 0; i < blockSize; i++, img += imgstep-blockSize )
        {
            for( j = 0; j < blockSize; j++, img++ )
            {
                int Ix = (img[1] - img[-1])*2 + img[-imgstep+1] - img[-imgstep-1] + img[imgstep+1] - img[imgstep-1];
                int Iy = (img[imgstep] - img[-imgstep])*2 + img[imgstep-1] - img[-imgstep-1] + img[imgstep+1] - img[-imgstep+1];
                a += Ix*Ix;
                b += Iy*Iy;
                c += Ix*Iy;
            }
        }
        responses[idx] = ((float)a * b - (float)c * c - HARRIS_K * (float)(a + b) * (a + b))*scale_sq_sq;
    }
}

#endif

/////////////////////////ORB////////////////////////////////////

#ifdef ORB_ANGLES

#define _DBL_EPSILON 2.2204460492503131e-16f
#define atan2_p1 (0.9997878412794807f*57.29577951308232f)
#define atan2_p3 (-0.3258083974640975f*57.29577951308232f)
#define atan2_p5 (0.1555786518463281f*57.29577951308232f)
#define atan2_p7 (-0.04432655554792128f*57.29577951308232f)

inline float fastAtan2( float y, float x )
{
    float ax = fabs(x), ay = fabs(y);
    float a, c, c2;
    if( ax >= ay )
    {
        c = ay/(ax + _DBL_EPSILON);
        c2 = c*c;
        a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    else
    {
        c = ax/(ay + _DBL_EPSILON);
        c2 = c*c;
        a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    if( x < 0 )
        a = 180.f - a;
    if( y < 0 )
        a = 360.f - a;
    return a;
}


__kernel void
ORB_ICAngle(__global const uchar* imgbuf, int imgstep, int imgoffset0,
            __global const int* layerinfo, __global const int* keypoints,
            __global float* responses, const __global int* u_max,
            int nkeypoints, int half_k )
{
    int idx = get_global_id(0);
    if( idx < nkeypoints )
    {
        __global const int* kpt = keypoints + idx*KEYPOINT_SIZE;

        __global const int* layer = layerinfo + kpt[KEYPOINT_Z]*LAYERINFO_SIZE;
        __global const uchar* center = imgbuf + imgoffset0 + layer[LAYERINFO_OFS] +
            kpt[KEYPOINT_Y]*imgstep + kpt[KEYPOINT_X];

        int u, v, m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        for( u = -half_k; u <= half_k; u++ )
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for( v = 1; v <= half_k; v++ )
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for( u = -d; u <= d; u++ )
            {
                int val_plus = center[u + v*imgstep], val_minus = center[u - v*imgstep];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        // we do not use OpenCL's atan2 intrinsic,
        // because we want to get _exactly_ the same results as the CPU version
        responses[idx] = fastAtan2((float)m_01, (float)m_10);
    }
}

#endif

/////////////////////////////////////////////////////////////

#ifdef ORB_DESCRIPTORS

__kernel void
ORB_computeDescriptor(__global const uchar* imgbuf, int imgstep, int imgoffset0,
                      __global const int* layerinfo, __global const int* keypoints,
                      __global uchar* _desc, const __global int* pattern,
                      int nkeypoints, int dsize )
{
    int idx = get_global_id(0);
    if( idx < nkeypoints )
    {
        int i;
        __global const int* kpt = keypoints + idx*ORIENTED_KEYPOINT_SIZE;

        __global const int* layer = layerinfo + kpt[KEYPOINT_Z]*LAYERINFO_SIZE;
        __global const uchar* center = imgbuf + imgoffset0 + layer[LAYERINFO_OFS] +
                                kpt[KEYPOINT_Y]*imgstep + kpt[KEYPOINT_X];
        float angle = as_float(kpt[KEYPOINT_ANGLE]);
        angle *= 0.01745329251994329547f;

        float cosa;
        float sina = sincos(angle, &cosa);

        __global uchar* desc = _desc + idx*dsize;

        #define GET_VALUE(idx) \
            center[mad24(convert_int_rte(pattern[(idx)*2] * sina + pattern[(idx)*2+1] * cosa), imgstep, \
                        convert_int_rte(pattern[(idx)*2] * cosa - pattern[(idx)*2+1] * sina))]

        for( i = 0; i < dsize; i++ )
        {
            int val;
        #if WTA_K == 2
            int t0, t1;

            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;

            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;

            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;

            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;

            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;

            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;

            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;

            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            pattern += 16*2;

        #elif WTA_K == 3
            int t0, t1, t2;

            t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
            val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

            t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

            t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

            t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

            pattern += 12*2;

        #elif WTA_K == 4
            int t0, t1, t2, t3, k;
            int a, b;

            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            t2 = GET_VALUE(2); t3 = GET_VALUE(3);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val = k;

            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            t2 = GET_VALUE(6); t3 = GET_VALUE(7);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val |= k << 2;

            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            t2 = GET_VALUE(10); t3 = GET_VALUE(11);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val |= k << 4;

            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            t2 = GET_VALUE(14); t3 = GET_VALUE(15);
            a = 0, b = 2;
            if( t1 > t0 ) t0 = t1, a = 1;
            if( t3 > t2 ) t2 = t3, b = 3;
            k = t0 > t2 ? a : b;
            val |= k << 6;

            pattern += 16*2;
        #else
            #error "unknown/undefined WTA_K value; should be 2, 3 or 4"
        #endif
            desc[i] = (uchar)val;
        }
    }
}

#endif
