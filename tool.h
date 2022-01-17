#ifndef TOOL_H
#define TOOL_H

#define HAVE_AVX512 true


#include <x86intrin.h>
#include <NTL/LLL.h>
using namespace NTL;

RR dot(Vec<RR>& u, Vec<RR>& v);
void red(Vec<RR>& u, Vec<RR>& v, RR& q);

void print(float *v, long nn);
void print(double *v, long nn);
void print(short *v,long nn);
void print(int *v,long nn);
void print(uint16_t *v,long nn);
void print(uint32_t *v, long nn);
void printsqrt(double *v, long nn);
void printsqrt(float *v, long nn);
void print(float *b, long nn, long mm);
void print(uint32_t *b, long nn, long mm);
void print(double **b, long nn, long mm);
void print(float **b, long nn, long mm);
void print_vec(float *res, long CSD, long int_bias);

double gh_coeff(long n);

uint64_t rand_uint64_t();
long discrete_gaussian(double t,double sigma2);

/*---------------------vecops---------------------*/
//nn should be divided by 16.
inline void set_zero(float *dst, long nn){
    if (HAVE_AVX512){
        __m512 r;
        r =_mm512_setzero_ps();
        for (long i = 0; i < nn/16; i++){
            _mm512_store_ps(dst+i*16, r);
        }
        return;
    }
    __m256 r;
    r = _mm256_setzero_ps();
    for (long i = 0; i < nn/8; i++){
        _mm256_store_ps(dst+i*8, r);
    }
    return;
}
inline void red(float *dst, float *src, float q, long nn){
    if (HAVE_AVX512){
        __m512 q1 = _mm512_set1_ps(q);
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(dst+i*16);
            _mm512_store_ps(dst+i*16, _mm512_fnmadd_ps(_mm512_load_ps(src+i*16), q1, x0));
        }
        return;
    }
    __m256 q1 = _mm256_set1_ps(q);
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(dst+i);
        _mm256_store_ps(dst+i, _mm256_fnmadd_ps(_mm256_load_ps(src+i), q1, x0));
    }
    return;
}
inline float norm(float *a, long nn){
    if (HAVE_AVX512){
        __m512 r0 = _mm512_setzero_ps();
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(a+16*i);
            r0 = _mm512_fmadd_ps(x0, x0, r0);
        }

        __m256 r256 = _mm256_add_ps(_mm512_castps512_ps256(r0), _mm512_extractf32x8_ps(r0, 1));
        __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r256), _mm256_extractf32x4_ps(r256, 1));
        r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
        r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
        return _mm_cvtss_f32(r128);
    }
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    for (long i = 0; i < nn/8; i++){
        x0 = _mm256_load_ps(a+8*i);
        r0 = _mm256_fmadd_ps(x0, x0, r0);
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
}
inline float dot(float *a, float *b, long nn){
    if (HAVE_AVX512){
        if (nn != 80){
            __m512 r0 = _mm512_setzero_ps();
            __m512 x0;
            for (long i = 0; i < nn/16; i++){
                x0 = _mm512_load_ps(a+16*i);
                r0 = _mm512_fmadd_ps(x0, _mm512_load_ps(b+16*i), r0);
            }

            __m256 r256 = _mm256_add_ps(_mm512_castps512_ps256(r0), _mm512_extractf32x8_ps(r0, 1));
            __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r256), _mm256_extractf32x4_ps(r256, 1));
            r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
            r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
            return _mm_cvtss_f32(r128);
        }else{
            __m512 x0 = _mm512_load_ps(a);
            __m512 y0 = _mm512_load_ps(b);
            __m512 x1 = _mm512_load_ps(a+16);
            __m512 y1 = _mm512_load_ps(b+16);
            __m512 x2 = _mm512_load_ps(a+32);
            __m512 y2 = _mm512_load_ps(b+32);
            __m512 x3 = _mm512_load_ps(a+48);
            __m512 y3 = _mm512_load_ps(b+48);
            __m512 x4 = _mm512_load_ps(a+64);
            __m512 y4 = _mm512_load_ps(b+64);
            __m512 r0 = _mm512_mul_ps(x0, y0);
            __m512 r1 = _mm512_mul_ps(x2, y2);
            r0 = _mm512_fmadd_ps(x1, y1, r0);
            r1 = _mm512_fmadd_ps(x3, y3, r1);
            r0 = _mm512_fmadd_ps(x4, y4, r0);
            r0 = _mm512_add_ps(r1, r0);
            __m256 r256 = _mm256_add_ps(_mm512_castps512_ps256(r0), _mm512_extractf32x8_ps(r0, 1));
            __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r256), _mm256_extractf32x4_ps(r256, 1));
            r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
            r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
            return _mm_cvtss_f32(r128);
        }
        
    }
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    for (long i = 0; i < nn/8; i++){
        x0 = _mm256_load_ps(a+8*i);
        r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(b+8*i), r0);
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
}
inline void add(float *dst, float *src1, float *src2, long nn){
    if (HAVE_AVX512){
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(src1+i*16);
            _mm512_store_ps(dst+i*16, _mm512_add_ps(_mm512_load_ps(src2+i*16), x0));
        }
        return;
    }
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(src1+i);
        _mm256_store_ps(dst+i, _mm256_add_ps(_mm256_load_ps(src2+i), x0));
    }
    return;
}
inline void add(float *dst, float *src, long nn){
    if (HAVE_AVX512){
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(src+i*16);
            _mm512_store_ps(dst+i*16, _mm512_add_ps(_mm512_load_ps(dst+i*16), x0));
        }
        return;
    }
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(src+i);
        _mm256_store_ps(dst+i, _mm256_add_ps(_mm256_load_ps(dst+i), x0));
    }
    return;
}
inline void sub(float *dst, float *src1, float *src2, long nn){
    if (HAVE_AVX512){
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(src1+i*16);
            _mm512_store_ps(dst+i*16, _mm512_sub_ps(x0, _mm512_load_ps(src2+i*16)));
        }
        return;
    }
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(src1+i);
        _mm256_store_ps(dst+i, _mm256_sub_ps(x0, _mm256_load_ps(src2+i)));
    }
    return;
}
inline void sub(float *dst, float *src, long nn){
    if (HAVE_AVX512){
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(src+i*16);
            _mm512_store_ps(dst+i*16, _mm512_sub_ps(_mm512_load_ps(dst+i*16), x0));
        }
        return;
    }
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(src+i);
        _mm256_store_ps(dst+i, _mm256_sub_ps(_mm256_load_ps(dst+i), x0));
    }
    return;
}
inline void copy(float *dst, float *src, long nn){
    if (HAVE_AVX512){
        __m512 x0;
        for (long i = 0; i < nn/16; i++){
            x0 = _mm512_load_ps(src+i*16);
            _mm512_store_ps(dst + i * 16, x0);
        }
        return;
    }
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(src+i);
        _mm256_store_ps(dst+i, x0);
    }
    return;
}



inline void add(short *dst, short *src1, short *src2, long n){
    if (HAVE_AVX512){
        __m512i x0;
        for (long i = 0; i < n; i++){
            x0 = _mm512_load_epi32(src1 + i * 32);
            _mm512_store_epi32(dst+i*32, _mm512_add_epi16(x0, _mm512_load_epi32(src2 + i * 32)));
        }
        return;
    }
    for (long i = 0; i < n * 32; i++){
        dst[i] = src1[i] +src2[i];
    }
}
inline void sub(short *dst, short *src1, short *src2, long n){
    if (HAVE_AVX512){
        __m512i x0;
        for (long i = 0; i < n; i++){
            x0 = _mm512_load_epi32(src1 + i * 32);
            _mm512_store_epi32(dst+i*32, _mm512_sub_epi16(x0, _mm512_load_epi32(src2 + i * 32)));
        }
        return;
    }
    for (long i = 0; i < n * 32; i++){
        dst[i] = src1[i] -src2[i];
    }
}
inline void red(short *dst, short *src, short q, long n){
    if (false){
        //todo
        return;
    }
    for (long i = 0; i < n*32; i++){
        dst[i] -= q *src[i];
    }
    return;
}
inline void add(short *dst, short *src, long n){
    if (HAVE_AVX512){
        __m512i x0;
        for (long i = 0; i < n; i++){
            x0 = _mm512_load_epi32(src + i * 32);
            _mm512_store_epi32(dst+i*32, _mm512_add_epi16(x0, _mm512_load_epi32(dst + i * 32)));
        }
        return;
    }
    for (long i = 0; i < n * 32; i++){
        dst[i] = dst[i] +src[i];
    }
}
inline void sub(short *dst, short *src, long n){
    if (HAVE_AVX512){
        __m512i x0;
        for (long i = 0; i < n; i++){
            x0 = _mm512_load_epi32(src + i * 32);
            _mm512_store_epi32(dst+i*32, _mm512_sub_epi16(_mm512_load_epi32(dst + i * 32), x0));
        }
        return;
    }
    for (long i = 0; i < n * 32; i++){
        dst[i] = dst[i]-src[i];
    }
}
inline void copy(short *dst, short *src, long n){
    if (HAVE_AVX512){
        for (long i = 0; i < n; i++){
            _mm512_store_epi32(dst + i * 32, _mm512_load_epi32(src + i * 32));
        }
        return;
    }
    
    for (long i = 0; i < n*32; i++){
        dst[i] = src[i];
    }
}


inline void add_coeff(float *dst, float *src1, float *src2, long int_bias, long n){
    if (HAVE_AVX512){
        __m512i x0;
        for (long i = 2; i <= int_bias/16; i++){
            x0 = _mm512_load_epi32(src1-i*16);
            _mm512_store_epi32(dst-i*16, _mm512_add_epi16(x0, _mm512_load_epi32(src2-i*16)));
        }
        return;
    }
    short *x = (short *)(&dst[-int_bias]);
    short *y = (short *)(&src1[-int_bias]);
    short *z = (short *)(&src2[-int_bias]);
    for(long i = 0; i < n; i++){
        x[i] = y[i]+z[i];
    }
    return;
}
inline void sub_coeff(float *dst, float *src1, float *src2, long int_bias, long n){
    if (HAVE_AVX512){
        __m512i x0;
        for (long i = 2; i <= int_bias/16; i++){
            x0 = _mm512_load_epi32(src1-i*16);
            _mm512_store_epi32(dst-i*16, _mm512_sub_epi16(x0, _mm512_load_epi32(src2-i*16)));
        }
        return;
    }
    short *x = (short *)(&dst[-int_bias]);
    short *y = (short *)(&src1[-int_bias]);
    short *z = (short *)(&src2[-int_bias]);
    for(long i = 0; i < n; i++){
        x[i] = y[i]-z[i];
    }
    return;
}
inline void copy_vec(float *dst, float *src, long int_bias, long vec_length){
    if (HAVE_AVX512){
        for (long i = -int_bias/16; i < vec_length/16; i++){
            _mm512_store_ps(dst+i*16, _mm512_load_ps(src+i*16));
        }
        return;
    }
    for (long i = -int_bias/8; i < vec_length/8; i++){
        _mm256_store_ps(dst+i*8, _mm256_load_ps(src+i*8));
    }
    return;
}

class gaussian_sampler{
    public:
        gaussian_sampler(){holdrand = 0;};
        gaussian_sampler(int seed){holdrand = seed;};
        ~gaussian_sampler(){};
        inline int discrete_gaussian(double t, double sigma2);
        void set_seed(int seed){holdrand = seed;};
        inline int myrand();
    private:
        int holdrand;   
};

inline int gaussian_sampler::discrete_gaussian(double t, double sigma2){
    long x,tt = floor(t+.5);
    double p,r,dt = t-tt;
    long BB = floor(sqrt(sigma2))+1;
    while(1){
        x = ((long) myrand()) % (2*BB+1);
        x -= BB;
        p = exp(-((x-dt)*(x-dt)) / (2*sigma2));
        r = ((int) myrand() % 100000) / 100000;
        if (r<p) return x+tt;
    }
}
inline int gaussian_sampler::myrand(){
    return(((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0x7fff);
}


#endif