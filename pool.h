#ifndef POOL_H
#define POOL_H

#include <omp.h>
#include <fstream>
#include "tool.h"
#include "basis.h"
#include "UidHashTable.h"
#include "thread_pool.hpp"
#include "parallel_algorithms.hpp"



#define cvec_size 6
#define XPC_BUCKET_THRESHOLD 102
#define XPC_THRESHOLD 96




class coeff_buffer{
    public:
        coeff_buffer(){};
        coeff_buffer(long coeffsize, long maxsize);
        ~coeff_buffer();
        void buffer_setup(long coeffsize, long maxsize);

        __attribute__ ((aligned (128))) long size = 0;
        long coeff_size;
        long max_size;
        short *buffer;
        short *buffer_store = NULL;
};
struct size{
    __attribute__ ((aligned (128))) int a = 0;
};



struct NVSieve_params{
    double alpha = 0.3;
    double one_epoch_ratio = 0.05;
    double saturation_ratio = 0.5;
    double saturation_radius = 4.0/3.0;
    double improve_ratio = 0.65;
    double resort_ratio = 0.85;
};

struct three_Sieve_params{
    double alpha = 0.3;
    double one_epoch_ratio = 0.05;
    double saturation_ratio = 0.5;
    double saturation_radius = 4.0/3.0;
    double improve_ratio = 0.7;
    double resort_ratio = 0.85;
};


class Pool {
    public:
        //basis
        Basis *basis;
        double **b = NULL;
        double **miu = NULL;
        double *B = NULL;
        double *b_store = NULL;
        double *miu_store = NULL;
        float **b_local = NULL;
        float *b_local_store = NULL;
        long index_l;
        long index_r;
        float gh2;
        
        //Sieving Status
        long dim;   //full dimension
        long MSD;   //maximal sieving dimension
        long CSD;   //current sieving dimension
        long vec_size;
        long int_bias;
        long vec_length;
        
        //pool
        long max_pool_size;
        long num_vec = 0;
        float *vec_store = NULL;
        float *vec;   
        long *cvec_store = NULL;         
        long *cvec;                     //compressed vectors

        //Simhash and uid
        uint32_t* compress_pos = NULL;
        UidHashTable uid;

        //basic operations
        Pool();
        Pool(Basis& bs);
        Pool(const char *file_name);
        ~Pool();
        void clear_all();
        void clear_pool();

        //setup
        void set_num_threads(long n);
        void set_basis(Basis& bs); 
        void set_MSD(long msd);
        void set_max_pool_size(long N);
        void set_sieving_context(long l, long r);
        void compute_gh2();
        
        //pool operations
        bool gaussian_sampling(long N);
        void extend_left();
        void sort_cvec();
        void show_pool_status();
        bool check_pool_status();
        void print_random_vec(long num);
        void show_vec_length();
        bool sieve_is_over(double saturation_radius, double saturation_ratio);
        void shrink(long N);
        void shrink_left();
        void insert(long index);
        void LLL_ZZ(double delta);
        void store(const char *file_name);
        void load(const char *file_name);
        void store_vec(const char *file_name);

        //Sieving
        void NVSieve(NVSieve_params params);
        void three_Sieve(three_Sieve_params params);
    private:
        long num_threads = 1;
        long sorted_index = 0;
        void Simhash_setup();
        void basis_setup(Basis& bs); 
        void update_b_local();
        void gaussian_sampling(float *res, long *cres, gaussian_sampler &R);
        inline void compute_vec(float *res);
        inline void compute_Simhash(float *res);
        inline void compute_uid(float *res);
        inline void compute_cvec(float *res, long *cres);
        thread_pool::thread_pool threadpool;
};


inline void Pool::compute_vec(float *res){
    short *x = (short *)(&res[-int_bias]);
    set_zero(res, vec_length);
    long CSD16 = ((CSD+15)/16)*16;
    for (long i = 0; i < CSD; i++){
        red(res, b_local[i], -x[CSD-1-i], CSD16);
    }
    compute_uid(res);
    compute_Simhash(res);
    res[-1] = norm(res, vec_length);
}
inline void Pool::compute_Simhash(float *res){
    uint64_t c0, c1, c2, c3;
    float a[4];
    uint32_t *a0 = (uint32_t *)(&a[0]);
    uint32_t *a1 = (uint32_t *)(&a[1]);
    uint32_t *a2 = (uint32_t *)(&a[2]);
    uint32_t *a3 = (uint32_t *)(&a[3]);

    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;
    for (long i = 0; i < 64; i++){
        a[0]   = res[compress_pos[24*i+0]];
        a[0]  += res[compress_pos[24*i+1]];
        a[0]  += res[compress_pos[24*i+2]];
        a[0]  -= res[compress_pos[24*i+3]];
        a[0]  -= res[compress_pos[24*i+4]];
        a[0]  -= res[compress_pos[24*i+5]];

        a[1]   = res[compress_pos[24*i+6]];
        a[1]  += res[compress_pos[24*i+7]];
        a[1]  += res[compress_pos[24*i+8]];
        a[1]  -= res[compress_pos[24*i+9]];
        a[1]  -= res[compress_pos[24*i+10]];
        a[1]  -= res[compress_pos[24*i+11]];

        a[2]   = res[compress_pos[24*i+12]];
        a[2]  += res[compress_pos[24*i+13]];
        a[2]  += res[compress_pos[24*i+14]];
        a[2]  -= res[compress_pos[24*i+15]];
        a[2]  -= res[compress_pos[24*i+16]];
        a[2]  -= res[compress_pos[24*i+17]];

        a[3]   = res[compress_pos[24*i+18]];
        a[3]  += res[compress_pos[24*i+19]];
        a[3]  += res[compress_pos[24*i+20]];
        a[3]  -= res[compress_pos[24*i+21]];
        a[3]  -= res[compress_pos[24*i+22]];
        a[3]  -= res[compress_pos[24*i+23]];

        c0 |= (uint64_t)((a0[0] & 0x80000000) >> 31) << i;
        c1 |= (uint64_t)((a1[0] & 0x80000000) >> 31) << i;
        c2 |= (uint64_t)((a2[0] & 0x80000000) >> 31) << i;
        c3 |= (uint64_t)((a3[0] & 0x80000000) >> 31) << i;
    } 
    *((uint64_t *)(&res[-16])) = c0;
    *((uint64_t *)(&res[-14])) = c1;
    *((uint64_t *)(&res[-12])) = c2;
    *((uint64_t *)(&res[-10])) = c3;
}
inline void Pool::compute_uid(float *res){
    short *x = (short *)(&res[-int_bias]);
    uint64_t u = 0;
    for (long i = 0; i < CSD; i++){
        u += x[i] * uid.uid_coeffs[i];
    }
    *((uint64_t *)(&res[-4])) = u;
}
inline void Pool::compute_cvec(float *res, long *cres){
    *((uint64_t *)(&cres[0])) = *((uint64_t *)(&res[-16]));
    *((uint64_t *)(&cres[1])) = *((uint64_t *)(&res[-14]));
    *((uint64_t *)(&cres[2])) = *((uint64_t *)(&res[-12]));
    *((uint64_t *)(&cres[3])) = *((uint64_t *)(&res[-10]));
    *((float *)(&cres[4])) = res[-1];
    *((float **)(&cres[5])) = res;
}






#endif