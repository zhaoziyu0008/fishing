#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <pthread.h>
#include <x86intrin.h>
#include <NTL/LLL.h>
#include <sys/time.h>

using namespace std;
using namespace NTL;

#define NUM_COMPONENT 3
#define BUCKET_INIT_SIZE 7
#define BUCKET_ENLARGE_SIZE 3
#define XOR_POPCNT_THRESHOLD 50
#define deux28 (((long long) 2) << 28)

#define SHOW_DETAILS true
#define REPORT_INTERVAL 5.0
#define HAVE_AVX512 false

long n;             //maximal sieving dimension
long m;             //length of local codes
long m8;
long n8;

//the original basis
Mat<RR> miu_RR;
Mat<RR> b_star_RR;
Vec<RR> B_RR;

// the projected basis for sampling
float **b;         //the projected lattice
double **miu;       //miu of b
double *B;          //B of b
float *b_store;
double *miu_store;
Mat<RR> U;          //base change matrix

//bucketing setup
float alpha;
float beta;
float* Polytope;
float* Polytope_store;
uint32_t** Bucket;
uint16_t* Bucket_size;
uint16_t* Bucket_max_size;
long num_buckets;
long TotalSize;

//Simhash setup
uint32_t* compress_pos;

//pool setup
long vec_size;
long vec_length;
long int_bias;
float *vec_store;
float *vec_start;
float **empty;
float **update;
long num_used;
long num_empty;
long num_update;

//running
float min_norm = 900000000;
float* min_vec;
long num_try_to_reduce = 0;
long num_pass_Simhash = 0;
long num_reduce = 0;

pthread_spinlock_t update_lock= 1;
pthread_spinlock_t empty_lock = 1;
pthread_spinlock_t pool_lock = 1;
pthread_spinlock_t Bucket_lock = 1;
pthread_spinlock_t min_lock = 1;


/*---------------------prints---------------------*/
void print(float *v, long nn){
    cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << v[i] << " ";
    }
    cout << v[nn-1] << "]\n";
}
void print(double *v, long nn){
    cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << v[i] << " ";
    }
    cout << v[nn-1] << "]\n";
}
void print(int *v, long nn){
    cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << v[i] << " ";
    }
    cout << v[nn-1] << "]\n";
}
void print(uint16_t *v,long nn){
    cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << v[i] << " ";
    }
    cout << v[nn-1] << "]\n";
}
void print(uint32_t *v, long nn){
     cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << v[i] << " ";
    }
    cout << v[nn-1] << "]\n";
}
void print(long long *v, long nn){
     cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << v[i] << " ";
    }
    cout << v[nn-1] << "]\n";
}
void printsqrt(double *v, long nn){
    cout << "[";
    for (long i = 0; i < nn-1; i++){
        cout << sqrt(v[i]) << " ";
    }
    cout << sqrt(v[nn-1]) << "]\n";
}
void print(float *b, long nn, long mm){
    cout << "[";
    for (long i = 0; i < nn; i++){
        print(b+i*mm, mm);
    }
    cout << "]";
}
void print(uint32_t *b, long nn, long mm){
    cout << "[";
    for (long i = 0; i < nn; i++){
        print(b+i*mm, mm);
    }
    cout << "]";
}
void print(long long *b, long nn, long mm){
    cout << "[";
    for (long i = 0; i < nn; i++){
        print(b+i*mm, mm);
    }
    cout << "]";
}
void print(double **b, long nn, long mm){
    cout << "[";
    for (long i = 0; i < nn; i++){
        print(b[i], mm);
    }
    cout << "]";
}
void print_vec(float *a){
    cout << "[";
    for (long i = 0; i < vec_length-1; i++){
            cout << a[i]<<" ";
    }
    cout << a[vec_length-1]<<"]\n";
    short *x = (short *)(&a[-int_bias]);
    cout << "length = "<<a[-1]<<", Status = "<<*((uint64_t *)(&a[-4])) << ", SimHash = ["<< *((uint64_t *)(&a[-8])) << " " <<*((uint64_t *)(&a[-6]))<<"], ";
    cout << "coeff = [";
    for (long i = 0; i < n-1; i++){
            cout << x[i]<<" ";
    }
    cout << x[n-1]<<"]\n";
}
void print_bucket(long *I){
    if (I[0]==0){
        cout << "[]\n";
        return;
    }
    cout << "[";
    for (long i = 1; i < I[0]; i++){
        cout << I[i] << " ";
    }
    cout << I[I[0]]<<"]\n";
}


/*-----------------------tools-----------------------*/
float dot_slow(float *a, float *b, long nn){
    float x = 0.0;
    for (long i = 0; i < nn; i++){
        x += a[i]*b[i];
    }
    return x;
}
float norm_slow(float *a, long nn){
    float x = 0.0;
    for (long i = 0; i < nn; i++){
        x += a[i]*a[i];
    }
    return x;
}
double gh_coeff(long nn){
    double a = 1.0;
    if (nn%2 == 0){
        long m = nn/2;
        for (long i = 1; i < m+1; i++){
            a = a * pow(i,1.0/nn);
        }
    }else{
        long m = (nn-1)/2;
        for (long i = 0; i < m+1; i++){
            a = a * pow(i+0.5,1.0/nn);
        }
        a = a * pow(3.14159265357989324,0.5/nn);
    }
    a = a / sqrt(3.14159265357989324);
    return a;
}


/*---------------------RR linalg---------------------*/
RR dot(Vec<RR>& u, Vec<RR>& v){
    RR x;
    long nn = u.length();
    for (long i = 0; i < nn; i++){
        x += u[i]*v[i];
    }
    return x;
}
void red(Vec<RR>& u, Vec<RR>& v, RR& q){
    long nn = u.length();
    for (long i = 0; i < nn; i++){
        u[i] -= v[i]*q;
    }
    return;
}
Mat<RR> random_orth_matrix(long k){
    Mat<RR> C;
    Mat<RR> miu;
    Vec<RR> B;
    C.SetDims(k,k);
    for (long i = 0; i < k; i++){
        for (long j = 0; j < k; j++){
            C[i][j] = to_RR(rand()%10000);
        }
    }

    miu.SetDims(k,k);
    B.SetLength(k);
    B[0] = dot(C[0],C[0]);
	for (long i = 0; i < k-1; i++){
		for (long j = i+1; j < k; j++){
			miu[j][i] = dot(C[j],C[i])/B[i];
            red(C[j],C[i],miu[j][i]);
		}
		B[i+1] = dot(C[i+1],C[i+1]);
	}
    for (long i = 0; i < k; i++){
        RR x;
        x = dot(C[i],C[i]);
        x = 1.0/sqrt(x);
        C[i] = C[i]*x;
    }
    return C;
}
Mat<RR> matmul(Mat<RR>& A, Mat<RR>& B){
    Mat<RR> C;
    long nn = A.NumRows();
    long mm = B.NumCols();
    long ll = A.NumCols();
    if (B.NumRows() != ll){
        cerr << "in matmul, size mismatch!\n";
        exit(1);
    }
    C.SetDims(nn,mm);
    for (long i = 0; i < nn; i++){
        for(long j = 0; j < mm; j++){
            RR x;
            for (long k = 0; k < ll; k++){
                x = x +A[i][k]*B[k][j];
            }
            C[i][j] = x;
        }
    }
    return C;
}


/*----------------------fast ops----------------------*/
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
inline float dot8(float *a, float *b, long nn){
    if (nn == 24){
        __m256 r0 = _mm256_setzero_ps();
        __m256 x0,x1,x2,y0,y1,y2;
        x0 = _mm256_load_ps(a);
        x1 = _mm256_load_ps(a+8);
        x2 = _mm256_load_ps(a+16);
        r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(b), r0);
        r0 = _mm256_fmadd_ps(x1, _mm256_load_ps(b+8), r0);
        r0 = _mm256_fmadd_ps(x2, _mm256_load_ps(b+16), r0);
        __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
        r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
        r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
        return _mm_cvtss_f32(r128);
    }
    if (nn == 32){
        __m256 r0 = _mm256_setzero_ps();
        __m256 x0,x1,x2,x3,y0,y1,y2;
        x0 = _mm256_load_ps(a);
        x1 = _mm256_load_ps(a+8);
        x2 = _mm256_load_ps(a+16);
        x3 = _mm256_load_ps(a+24);
        r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(b), r0);
        r0 = _mm256_fmadd_ps(x1, _mm256_load_ps(b+8), r0);
        r0 = _mm256_fmadd_ps(x2, _mm256_load_ps(b+16), r0);
        r0 = _mm256_fmadd_ps(x3, _mm256_load_ps(b+24), r0);
        __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
        r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
        r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
        return _mm_cvtss_f32(r128);
    }
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    for (long i = 0; i < nn; i+=8){
        x0 = _mm256_load_ps(a+i);
        r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(b+i), r0);
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
}
inline void copy(float *dst, float *src, long nn){
    if (HAVE_AVX512){
        for (long i = 0; i < nn; i+=16){
            _mm512_store_ps(dst+i, _mm512_load_ps(src+i));
        }
        return;
    }
    for (long i = 0; i < nn; i+=8){
        _mm256_store_ps(dst+i, _mm256_load_ps(src+i));
    }
    return;
}
inline void mul(float *dst, float x, long nn){
    if (HAVE_AVX512){
        __m512 x0 = _mm512_set1_ps(x);
        for (long i = 0; i < nn; i+=16){
            _mm512_store_ps(dst+i, _mm512_mul_ps(_mm512_load_ps(dst+i), x0));
        }
        return;
    }
    __m256 x0 = _mm256_set1_ps(x);
    for (long i = 0; i < nn; i+=8){
        _mm256_store_ps(dst+i, _mm256_mul_ps(_mm256_load_ps(dst+i), x0));
    }
    return;
}
inline void copy_vec(float *dst, float *src){
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
inline void add_coeff(float *dst, float *src1, float *src2){
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
inline void sub_coeff(float *dst, float *src1, float *src2){
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
void compute_Simhash(float *res){
    uint64_t clo, chi;
    float a[2];
    uint32_t *alo = (uint32_t *)(&a[0]);
    uint32_t *ahi = (uint32_t *)(&a[1]);

    clo = 0;
    chi = 0;
    for (long i = 0; i < 64; i++){
        a[0]   = res[compress_pos[8*i+0]];
        a[0]  -= res[compress_pos[8*i+1]];
        a[0]  += res[compress_pos[8*i+2]];
        a[0]  -= res[compress_pos[8*i+3]];

        a[1]   = res[compress_pos[8*i + 4]];
        a[1]  -= res[compress_pos[8*i + 5]];
        a[1]  += res[compress_pos[8*i + 6]];
        a[1]  -= res[compress_pos[8*i + 7]];

        clo |= (uint64_t)((alo[0] & 0x80000000) >> 31) << i;
        chi |= (uint64_t)((ahi[0] & 0x80000000) >> 31) << i;
    } 
    *((uint64_t *)(&res[-8])) = clo;
    *((uint64_t *)(&res[-6])) = chi;
}
void compute_vec(float *res){
    short *x = (short *)(&res[-int_bias]);
    set_zero(res, vec_length);
    for (long i = 0; i < n; i++){
        /*for (long j = 0; j < vec_length; j++){
            res[j]+= x[i]*b[i][j];
        }*/
        red(res, b[i], -x[i], vec_length);
    }
    *((uint64_t *)(&res[-4])) = 1;
    compute_Simhash(res);
    res[-1] = norm(res, vec_length)+100.0;
}


/*----------------------sampling----------------------*/
long discrete_gaussian(double sigma2){
    long x, BB = floor(5*sqrt(sigma2))+1;
    double p,r,y = 0.5/sigma2;
    while(1){
        x = ((long) rand()) % (2*BB+1);
        x -= BB;
        p = exp(-x*x*y);
        r = ((long) rand() % 100000) / 100000;
        if (r < p) return x;
    }
}
long discrete_gaussian(double t,double sigma2){
    long x,tt = floor(t+.5);
    double p,r,dt = t-tt;
    long BB = floor(sqrt(sigma2))+1;
    while(1){
        x = ((long) rand()) % (2*BB+1);
        x -= BB;
        p = exp(-((x-dt)*(x-dt)) / (2*sigma2));
        r = ((int) rand() % 100000) / 100000;
        if (r<p) return x+tt;
    }
}
void spherical_sampling(float *res, long nn){
    long i,j;
    for(i = 0; i < nn; i++){
        res[i] = (float) discrete_gaussian(1000000);
    }
    float sqn=norm_slow(res, nn);
    float x = 1/sqrt(sqn);
    for(i = nn - 1; i >= 0; i--){
        res[i] *= x;
    }

}
bool gaussian_sampling_on_dim(long current_dim, float *res){
    int coeff[n];
    double sigma2 = B[current_dim/2];
    set_zero(res, vec_length);
    if (current_dim > n) return false;
    for (long i = current_dim - 1; i >= 0; i--){
        coeff[i] = discrete_gaussian(res[i],sigma2/B[i]);
        for(long j = 0; j < i; j++){
            res[j] -= coeff[i]*miu[i][j];
        }
    }
    for (long i = 0; i < current_dim; i++){
        ((short*)(&res[-int_bias]))[i] = coeff[i];
    }
    for (long i = current_dim; i < n; i++){
        ((short*)(&res[-int_bias]))[i] = 0;
    }
    compute_vec(res);
}


/*-----------------------setup-----------------------*/
void compute_basis_gso(Mat<double>& L){
    RR::SetPrecision(150);
    miu_RR.SetDims(L.NumRows(),L.NumRows());
    B_RR.SetLength(L.NumRows());
	b_star_RR.SetDims(L.NumRows(),L.NumCols());
    for (long i = 0; i < L.NumRows(); i++){
        for (long j = 0; j < L.NumCols(); j++){
            b_star_RR[i][j] = to_RR(L[i][j]);
        }
    }
    B_RR[0] = dot(b_star_RR[0],b_star_RR[0]);
	for (long i = 0; i < L.NumRows()-1; i++){
		for (long j = i+1; j < L.NumRows(); j++){
			miu_RR[j][i] = dot(b_star_RR[j],b_star_RR[i])/B_RR[i];
            red(b_star_RR[j],b_star_RR[i],miu_RR[j][i]);
		}
		B_RR[i+1] = dot(b_star_RR[i+1],b_star_RR[i+1]);
	}
    if (SHOW_DETAILS){
        cout << "gso data computing done!\n";
        //cout << "B = "<<B_RR<<endl;
        //cout << "miu = "<<miu_RR << endl<<endl;
    }
}
/* we don't use the basis b_star to repb the vectors, but do a 
 * orthogonality transform on it, by matrix U, this will help 
 * to combine the Progressive Sieving and Bucketing technique 
 * tegother, and keeps the input basis quality. a vec v will be
 * mapped to vU. */
void projected_basis_setup(long begin_index, long upper_index){
    //b_local is the original basis
    Mat<RR> b_local;
    b_local.SetDims(n,n);
    for (long i = 0; i < n; i++){
        for (long j = 0; j < i; j++){
            b_local[i][j] = miu_RR[i+begin_index][j+begin_index];
        }
        b_local[i][i] = 1.0;
    }
    for (long i = 0; i < n; i++){
        RR x(sqrt(B_RR[begin_index+i]));
        for (long j = i; j < n; j++){
            b_local[j][i] = b_local[j][i]*x;
        }
    }
    //tansform b_local by U
    U = random_orth_matrix(n);
    Mat<RR> b_;
    b_ = matmul(b_local, U);

    //the length of floating point part of each vec
    vec_length = m8*NUM_COMPONENT;
    if (vec_length % 16 != 0) vec_length += 8; 

    b = new float*[n];
    miu = new double*[n];
    B = (double *) malloc(8*n);
    b_store = (float *) calloc(1, 4*vec_length*n+64);
    miu_store = (double *) calloc(1, 8*n8*n+64);
    float *b_start = (float *) ((((long long)(b_store)-1)/64+1)*64);
    double *miu_start = (double *) ((((long long)(miu_store)-1)/64+1)*64);
    for (long i = 0; i < n; i++){
        b[i] = b_start + i * vec_length;
        miu[i] = miu_start + i * n8;
    }
    for (long i = 0; i < n; i++){
        miu[i][i] = 1.0;
        B[i] = conv<double>(B_RR[i+begin_index]);
        for (long j = 0; j < i; j++){
            miu[i][j] = conv<double>(miu_RR[i+begin_index][j+begin_index]);
        }
    }
    //we store b, miu in an aligned manner so we can use simd instructions
    int a[NUM_COMPONENT];
    for (long i = 0; i < NUM_COMPONENT; i++){
        a[i] = m;
    }
    for (long i = m*NUM_COMPONENT-n-1; i >= 0; i--){
        a[i] -= 1;
    }
    for (long i = 0; i < n; i++){
        long s = 0;
        for (long j = 0; j < NUM_COMPONENT; j++){
            for (long k = 0; k < a[j]; k++){
                b[i][j*m8+k] = conv<float>(b_[i][s]);
                s++;
            }
        }
    }  
    if (SHOW_DETAILS){
        cout << "projected basis computing done!\n";
        //cout << "b_local = " << b_local<<endl;
        //cout << "U = " << U << endl;
        //cout << "b_ = "<< b_ << endl;
        cout << "vec_length is set to be "<< vec_length << ", the vec is sep to "<< NUM_COMPONENT<< " parts, [";
        for (long i = 0; i < NUM_COMPONENT-1; i++) cout<<a[i]<<" ";
        cout << a[NUM_COMPONENT-1]<<"]\n";
        //cout << "b = ";
        //print(b, n, vec_length);
        //cout << "miu = ";
        //print(miu, n, n);
        cout << "B = ";
        print(B, n);
        cout << "\n";
    }
}
void improve_poly(long Psize, float delta){
	float Norm;
	for (long i = 0; i < Psize; i++){
		for (long j = 0; j < Psize; j++){
			float scal = dot_slow(&Polytope[i*m8], &Polytope[j*m8], m8);
			Norm = sqrt(norm_slow(&Polytope[i*m8], m8));
			if (scal<0){
                scal = 0;
            }else{
                scal /= Norm;
            }
			for (int k = 0; k < m8; k++){
                Polytope[i*m8 +k] -= scal * delta * Polytope[j*m8+k];
            }
		}
		Norm = sqrt(norm_slow(&Polytope[i*m8], m8));
        float x = 1/Norm;
		for (int k = 0; k < m8; k++){
            Polytope[i*m8+k] *= x;
        }
	}
}
void ListDecoding_setup(long Psize){
    Polytope_store = new float[m8*Psize+32];
    for (long i = 0; i < m8*Psize+32; i++){
        Polytope_store[i] = 0.0;
    }
    Polytope = (float *) ((((long long)(Polytope_store)-1)/32+1)*32);
    for (long i = 0; i < Psize; i++){
        spherical_sampling(Polytope+i*m8, m);
    }
    for (int i = 0; i < 100; i++){
        improve_poly(Psize, 0.1);
    }
    if (SHOW_DETAILS){
        cout << "ListDecode setup done! Psize = "<< Psize<<", each code is of length "<< m8<<"\n";
        //cout <<"\nPolytope = ";
        //print(Polytope, Psize, m8);
    }
}
void Bucketing_setup(long Psize){
    num_buckets = 1;
	for (int i = 0; i < NUM_COMPONENT; i++){
		num_buckets*=Psize;
	}
    TotalSize = 0;
    Bucket = new uint32_t*[num_buckets];
    for (long i = 0; i < num_buckets; i++){
        Bucket[i] = new uint32_t[BUCKET_INIT_SIZE];
    }
    Bucket_size = new uint16_t[num_buckets];
    Bucket_max_size = new uint16_t[num_buckets];
    for (long i = 0; i < num_buckets; i++){
        Bucket_size[i] = 0;
        Bucket_max_size[i] = BUCKET_INIT_SIZE;
    }
    if (SHOW_DETAILS){
        cout << "Bucketing setup done! num_bucket = "<< num_buckets<< ", Bucket init size = "<<BUCKET_INIT_SIZE << endl;
    }
}
void Simhash_setup(){
    compress_pos = new uint32_t[128*4];
    for (long i = 0; i < 128; i++){
        compress_pos[i*4+0] = i % n;
        compress_pos[i*4+1] = rand() % n;
        compress_pos[i*4+2] = rand() % n;
        compress_pos[i*4+3] = rand() % n;
    }
    int a[NUM_COMPONENT];
    for (long i = 0; i < NUM_COMPONENT; i++){
        a[i] = m;
    }
    for (long i = m*NUM_COMPONENT-n-1; i >= 0; i--){
        a[i] -= 1;
    }
    for (long i = 0; i < 128*4; i++){
        uint32_t sum = 0;
        uint32_t ad = 0;
        for (long j = 0; j < NUM_COMPONENT; j++){
            sum += a[j];
            if (compress_pos[i]>=sum) ad += (m8 - a[j]);
        }
        compress_pos[i] += ad;
    }
    if (SHOW_DETAILS){
        cout << "Simhash setup done!\n";// compress_pos = \n";
        //print(compress_pos, 128, 4);
    }
}
void pool_setup(long max_vec){
    int_bias = ((n+31)/32)*16+16;
    vec_size = vec_length+int_bias;
    vec_store = (float *) calloc (4*vec_size*max_vec+512, 1);
    vec_start = (float *) ((((long long)(vec_store)-1)/64+6)*64);
    empty = (float **) malloc (8*max_vec);
    update = (float **) malloc (8*max_vec);
    num_update = 0;
    num_empty = 0;
    num_used = 0;

    if (SHOW_DETAILS){
        cout << "pool_setup done!\n";
    }
}
void clear_all(){
    delete[] b;
    delete[] miu;
    delete[] Polytope_store;
    for (long i = 0; i < num_buckets; i++){
        delete[] Bucket[i];
    }
    delete[] Bucket;
    delete[] Bucket_size;
    delete[] Bucket_max_size;
    delete[] compress_pos;
    free(vec_store);
    free(empty);
    free(update);
    free(b_store);
    free(miu_store);
    free(B);
}


/*----------------------running----------------------*/
bool sieve_is_over(long current_dim, double& gh){
    long count = 0;
    float *ptr = vec_start;
    float R = 1.3333*gh;
    for (long i = 0; i < num_used; i++){
        ptr = ptr+vec_size;
        if (*((uint64_t *)(&ptr[-4]))==0){
            if (ptr[-1] < R){
                count++;
            }
        }
    }
    if (2.0 * count > 0.5 * pow(1.3333, current_dim/2.0)){
        if (current_dim < n){
            gh = gh_coeff(current_dim+1);
            gh = gh * gh;
            for (long i = 0; i < current_dim+1; i++){
                gh = gh * pow(B[i], 1.0/(current_dim+1));
            }
        }
        return true;
    }
    return false;
}
void ListDecode(long *I, long *In, float *ptr, long Psize, float *vec_tmp, float *InnerProd_tmp, long long *Sort, float delta, float *minsums, float *sums){
    copy(vec_tmp, ptr, vec_length);
    float xx = 1/sqrt(norm(ptr, vec_length)*(n/(float)m));
    mul(vec_tmp, xx, vec_length);

    for (long i = 0; i < NUM_COMPONENT; i++){
        for (long p = 0; p < Psize; p++){
            InnerProd_tmp[i*Psize+p] = dot8(vec_tmp+m8*i, Polytope+m8*p, m8);
        }
    }
    
    float dd = deux28;
    long l = ((NUM_COMPONENT*Psize+15)/16)*16;
    mul(InnerProd_tmp, dd, l);
    for (long i = 0; i < NUM_COMPONENT; i++){
        for (int p = 0; p < Psize; p++){
            Sort[i*Psize + p] = ((long)(InnerProd_tmp[i*Psize + p])) << 32;    
            Sort[i*Psize + p] += p;
        }
        sort(&Sort[i*Psize], &Sort[(i+1)*Psize], greater<long>());
    }
    double x = 1.0/((double) deux28);
    for (int i = 0; i < NUM_COMPONENT; i++){
		for (int p = 0; p < Psize; p++){
			InnerProd_tmp[i*Psize + p] = ((float) (Sort[i*Psize + p] >> 32));
			Sort[i*Psize + p] = (Sort[i*Psize + p] << 32) >> 32 ;
		}
	}
    mul(InnerProd_tmp, x, l);

    long num_rel_buckets = 0;
    if (NUM_COMPONENT==4){
        float *v0 = &InnerProd_tmp[Psize*0];
        float *v1 = &InnerProd_tmp[Psize*1];
        float *v2 = &InnerProd_tmp[Psize*2];
        float *v3 = &InnerProd_tmp[Psize*3];
        for (long i = 0; i < NUM_COMPONENT; i++){
            minsums[i] = delta;
            for (long j = i+1; j < NUM_COMPONENT; j++){
                minsums[i] -= InnerProd_tmp[j*Psize + 0];
            }
        }
        for (long j0 = 0; (j0 < Psize) && (v0[j0] >= minsums[0]); j0++){
            sums[0] = InnerProd_tmp[0*Psize + j0];
            for (long j1 = 0; (j1 < Psize) && (v1[j1] + sums[0] >= minsums[1]); j1++){
                sums[1] = sums[0] + InnerProd_tmp[1*Psize + j1];
                for (long j2 = 0; (j2 < Psize) && (v2[j2] + sums[1] >= minsums[2]); j2++){
                    sums[2] = sums[1] + InnerProd_tmp[2*Psize + j2];
                    for (long j3 = 0; (j3 < Psize) && (v3[j3] + sums[2] >= minsums[3]); j3++){
                        num_rel_buckets++;
                        I[num_rel_buckets] = Psize*(Psize*(Psize*Sort[0*Psize + j0] + Sort[1*Psize + j1]) + Sort[2*Psize + j2]) + Sort[3*Psize + j3];
                    }			
                }
            }
        }
        I[0] = num_rel_buckets;
        if (In != NULL){
            num_rel_buckets = 0;
            for (long i = 0; i < NUM_COMPONENT; i++){
                minsums[i] = -delta;
                for (long j = i+1; j < NUM_COMPONENT; j++){
                    minsums[i] -= InnerProd_tmp[j*Psize + Psize-1];
                }
            }

            for (long j0 = Psize-1; (j0 >= 0) && (v0[j0] <= minsums[0]); j0--){
                sums[0] = InnerProd_tmp[0*Psize + j0];
                for (long j1 = Psize-1; (j1 >= 0) && (v1[j1] + sums[0] <= minsums[1]); j1--){
                    sums[1] = sums[0] + InnerProd_tmp[1*Psize + j1];
                    for (long j2 = Psize-1; (j2 >= 0) && (v2[j2] + sums[1] <= minsums[2]); j2--){
                        sums[2] = sums[1] + InnerProd_tmp[2*Psize + j2];
                        for (long j3 = Psize-1; (j3 >= 0) && (v3[j3] + sums[2] <= minsums[3]); j3--){
                            num_rel_buckets++;
                            In[num_rel_buckets] = Psize*(Psize*(Psize*Sort[0*Psize + j0] + Sort[1*Psize + j1]) + Sort[2*Psize + j2]) + Sort[3*Psize + j3];
                        }			
                    }
                }
            }
            In[0] = num_rel_buckets;
        }
    }
    if (NUM_COMPONENT==3){
        float *v0 = &InnerProd_tmp[Psize*0];
        float *v1 = &InnerProd_tmp[Psize*1];
        float *v2 = &InnerProd_tmp[Psize*2];
        for (long i = 0; i < NUM_COMPONENT; i++){
            minsums[i] = delta;
            for (long j = i+1; j < NUM_COMPONENT; j++){
                minsums[i] -= InnerProd_tmp[j*Psize + 0];
            }
        }

        for (long j0 = 0; (j0 < Psize) && (v0[j0] >= minsums[0]); j0++){
            sums[0] = InnerProd_tmp[0*Psize + j0];
            for (long j1 = 0; (j1 < Psize) && (v1[j1] + sums[0] >= minsums[1]); j1++){
                sums[1] = sums[0] + InnerProd_tmp[1*Psize + j1];
                for (long j2 = 0; (j2 < Psize) && (v2[j2] + sums[1] >= minsums[2]); j2++){
                        num_rel_buckets++;
                        I[num_rel_buckets] = Psize*(Psize*Sort[0*Psize + j0] + Sort[1*Psize + j1])+Sort[2*Psize + j2];
                }
            }
        }
        I[0] = num_rel_buckets;
        if (In != NULL){
            num_rel_buckets = 0;
            for (long i = 0; i < NUM_COMPONENT; i++){
                minsums[i] = -delta;
                for (long j = i+1; j < NUM_COMPONENT; j++){
                    minsums[i] -= InnerProd_tmp[j*Psize + Psize-1];
                }
            }
            for (long j0 = Psize-1; (j0 >= 0) && (v0[j0] <= minsums[0]); j0--){
                sums[0] = InnerProd_tmp[0*Psize + j0];
                for (long j1 = Psize-1; (j1 >= 0) && (v1[j1] + sums[0] <= minsums[1]); j1--){
                    sums[1] = sums[0] + InnerProd_tmp[1*Psize + j1];
                    for (long j2 = Psize-1; (j2 >= 0) && (v2[j2] + sums[1] <= minsums[2]); j2--){
                        num_rel_buckets++;
                        In[num_rel_buckets] = Psize*(Psize*Sort[0*Psize + j0] + Sort[1*Psize + j1])+Sort[2*Psize + j2];  		
                    }
                }
            }
            In[0] = num_rel_buckets;
        }   
    }
}
void Add_To_Bucket(long id, long *Iadd){     
    for (long i = 1; i <= Iadd[0]; i++){
        if (Bucket_size[Iadd[i]] == Bucket_max_size[Iadd[i]]){
            Bucket[Iadd[i]] = (uint32_t *)realloc(Bucket[Iadd[i]], 4*(Bucket_max_size[Iadd[i]]+BUCKET_ENLARGE_SIZE));
            Bucket_max_size[Iadd[i]] += BUCKET_ENLARGE_SIZE;
        }
        Bucket[Iadd[i]][Bucket_size[Iadd[i]]] = id;
        Bucket_size[Iadd[i]]++;
    }
}
void Rem_From_Bucket(long id, long *Irem){     
    for (long i = 1; i <= Irem[0]; i++){
        for (long j = 0; j < Bucket_size[Irem[i]]; j++){
            if (Bucket[Irem[i]][j] == id){				
                Bucket[Irem[i]][j] = Bucket[Irem[i]][Bucket_size[Irem[i]]-1];
                Bucket_size[Irem[i]]--;
                break;
            }
        }
    }
}
void work(long& current_dim, long max_vec, long Psize, bool check_dim_up){
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double next_report = 5.0;

    double gh;
    long last_num_used;
    if (check_dim_up){
        gh = gh_coeff(current_dim);
        gh = gh * gh;
        for (long i = 0; i < current_dim; i++){
            gh = gh * pow(B[i], 1.0/current_dim);
        }
        last_num_used = num_used;
    }
    __attribute__ ((aligned (64))) float tmp_store[vec_size];
    float* tmp = &tmp_store[int_bias];
    float* InnerProd_tmp_store = new float[NUM_COMPONENT*Psize+16];
    float* InnerProd_tmp = (float *) ((((long long)(InnerProd_tmp_store)-1)/64+1)*64);
    long long* Sort = new long long[NUM_COMPONENT*Psize+16];
    float* vec_tmp = new float[vec_length];
    float *sums = new float[NUM_COMPONENT];
	float *minsums = new float[NUM_COMPONENT];
    long *rel_Buckets = new long[num_buckets];
    long *nrel_Buckets = new long[num_buckets];
    long *Irem = new long[num_buckets];
    while ((current_dim <= n) && (num_used < max_vec-1)){
        if (num_update==0){
            gaussian_sampling_on_dim(current_dim, tmp);
            float *res;
            if (num_empty > 0){
                res = empty[num_empty-1];
                num_empty--;
            }else{
                res = vec_start+vec_size*num_used;
                num_used++;
            }
            copy_vec(res, tmp);
            update[num_update] = res;
            num_update++;
        }
        while (num_update > 0){
            gettimeofday(&end, NULL);
            if ((end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0 > next_report){
                next_report+=5;
                cout << "current sieving dimension = "<<current_dim <<"\n";
                cout << "num_try_to_reduce = "<< num_try_to_reduce << ", num_pass_Simhash = "<< num_pass_Simhash << ", num_reduce = "<< num_reduce<< ", current min = "<<sqrt(min_norm) <<"\n"; 
                cout << "shortest vec = ";
                if (min_vec) print_vec(min_vec);
                cout << "TotalSize = " << TotalSize <<", num_used = "<< num_used << ", num_empty = "<<num_empty<<endl<<endl;
            }
            float *ptr = update[num_update-1];
            num_update--;
            bool ptr_is_reduced = false;
            ListDecode(rel_Buckets, nrel_Buckets, ptr, Psize, vec_tmp, InnerProd_tmp, Sort, beta, minsums, sums);
            long num_rel_buckets = rel_Buckets[0];
            long num_nrel_buckets = nrel_Buckets[0];

            for (long i = 1; i <= num_rel_buckets; i++){
                if (ptr_is_reduced) break;
                long ii = rel_Buckets[i];
                for (long j = 0; j < Bucket_size[ii]; j++){
                    float *pptr = vec_start+vec_size*Bucket[ii][j];
                    num_try_to_reduce++;
                    long w = __builtin_popcountl((*((uint64_t *)(&ptr[-8]))) ^ (*((uint64_t *)(&pptr[-8]))));
                    w += __builtin_popcountl((*((uint64_t *)(&ptr[-6]))) ^ (*((uint64_t *)(&pptr[-6]))));
                    if (w < XOR_POPCNT_THRESHOLD || w > (128 - XOR_POPCNT_THRESHOLD)){
                        if (*((uint64_t *)(&pptr[-4]))) continue;
                        num_pass_Simhash++;
                        float x = dot(ptr, pptr, vec_length);                  
                        float f = abs(x+x);
                        if (f > ptr[-1]){
                            tmp[-1] = pptr[-1] + ptr[-1] - f;
                            if (tmp[-1] < 500.0){
                                empty[num_empty] = ptr;
                                num_empty++;
                                ptr_is_reduced = true;
                                break;
                            }
                            num_reduce++;
                            ListDecode(Irem, NULL, pptr, Psize, vec_tmp, InnerProd_tmp, Sort, alpha, minsums, sums);
                            if (x < 0){
                                add_coeff(tmp, pptr, ptr);
                            }else{
                                sub_coeff(tmp, pptr, ptr);
                            }
                            compute_vec(tmp);                            
                            copy_vec(pptr, tmp);
                            uint32_t id = Bucket[ii][j];
                            Rem_From_Bucket(id, Irem);
                            if (pptr[-1] < min_norm) {
                                min_norm = pptr[-1];
                                min_vec = pptr;
                            }
                            TotalSize -= Irem[0];
                            update[num_update] = pptr;
                            num_update++;
                        }else if (f > pptr[-1]){
                            num_reduce++;
                            if (x < 0){
                                add_coeff(tmp, ptr, pptr);
                            }else{
                                sub_coeff(tmp, ptr, pptr);
                            }
                            compute_vec(tmp);
                            copy_vec(ptr, tmp);
                            if (ptr[-1] < min_norm) {
                                min_norm = ptr[-1];
                                min_vec = ptr;
                            }
                            update[num_update] = ptr;
                            num_update++;
                            ptr_is_reduced = true;
                            break;
                        }
                    }
                }
            }
            for (long i = 1; i <= num_nrel_buckets; i++){
                if (ptr_is_reduced) break;
                long ii = nrel_Buckets[i];
                for (long j = 0; j < Bucket_size[ii]; j++){
                    float *pptr = vec_start+vec_size*Bucket[ii][j];
                    num_try_to_reduce++;
                    long w = __builtin_popcountl((*((uint64_t *)(&ptr[-8]))) ^ (*((uint64_t *)(&pptr[-8]))));
                    w += __builtin_popcountl((*((uint64_t *)(&ptr[-6]))) ^ (*((uint64_t *)(&pptr[-6]))));
                    if (w < XOR_POPCNT_THRESHOLD || w > (128 - XOR_POPCNT_THRESHOLD)){
                        if (*((uint64_t *)(&pptr[-4]))) continue;
                        num_pass_Simhash++;
                        float x = dot(ptr, pptr, vec_length);                  
                        float f = abs(x+x);
                        if (f > ptr[-1]){
                            tmp[-1] = pptr[-1] + ptr[-1] - f;
                            if (tmp[-1] < 500.0){
                                empty[num_empty] = ptr;
                                num_empty++;
                                ptr_is_reduced = true;
                                break;
                            }
                            num_reduce++;
                            ListDecode(Irem, NULL, pptr, Psize, vec_tmp, InnerProd_tmp, Sort, alpha, minsums, sums);
                            if (x < 0){
                                add_coeff(tmp, pptr, ptr);
                            }else{
                                sub_coeff(tmp, pptr, ptr);
                            }
                            compute_vec(tmp);
                            copy_vec(pptr, tmp);
                            uint32_t id = Bucket[ii][j];
                            Rem_From_Bucket(id, Irem);
                            if (pptr[-1] < min_norm) {
                                min_norm = pptr[-1];
                                min_vec = pptr;
                            }
                            TotalSize -= Irem[0];
                            update[num_update] = pptr;
                            num_update++;
                        }else if (f > pptr[-1]){
                            num_reduce++;
                            if (x < 0){
                                add_coeff(tmp, ptr, pptr);
                            }else{
                                sub_coeff(tmp, ptr, pptr);
                            }
                            compute_vec(tmp);
                            copy_vec(ptr, tmp);
                            if (ptr[-1] < min_norm) {
                                min_norm = ptr[-1];
                                min_vec = ptr;
                            }
                            update[num_update] = ptr;
                            num_update++;
                            ptr_is_reduced = true;
                            break;
                        }
                    }
                }
            }
            if (!ptr_is_reduced){
                long id = (long)(ptr-vec_start);
                id /= vec_size;
                Add_To_Bucket(id, rel_Buckets);
                *((uint64_t *)(&ptr[-4])) = 0;
                TotalSize+=rel_Buckets[0];
            }
        }
        if (check_dim_up){
            if (last_num_used < num_used - 100){
                last_num_used = num_used;
                if (sieve_is_over(current_dim, gh)){
                    if (current_dim == n){
                        current_dim++;
                    }else{
                        current_dim = min(current_dim+2, n);
                    }
                    cerr << "dim up to "<<current_dim<< ", gh = "<<sqrt(gh)<<endl;
                }
            }
        }  
    }

    gettimeofday(&end, NULL);
    cout << "total time = "<<(end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0<<"s"<<endl;

    delete[] rel_Buckets;
    delete[] nrel_Buckets;
    delete[] Irem;
    delete[] InnerProd_tmp_store;
    delete[] Sort;
    delete[] vec_tmp;
    delete[] sums;
    delete[] minsums;
}


/* Sieving on interval [begin_index, upper_index). */
void LDGaussSieve(Mat<double>& L, long begin_index, long upper_index, long max_vec, long Psize, float alpha_, float beta_, long num_threads){
    //srand(time(NULL));
    n = upper_index-begin_index;
    if (n % NUM_COMPONENT) {
        cerr << "warning! n not divided by NUM_COMPONENT, which is not recommended!\n";
    }
    m = (n-1)/NUM_COMPONENT+1;
    m8 = ((m+7)/8)*8;
    n8 = ((n+7)/8)*8;
    alpha = alpha_;
    beta = beta_;
    if (SHOW_DETAILS){
        cout << "begin to do LDGaussSieve on an "<< L.NumRows() << " dimensional lattice, sieving context = "<< "["<<begin_index<<", "<<upper_index<<")\n";
        cout << "num_component = "<< NUM_COMPONENT << ", m = "<<m<<", m8 = "<<m8<<", alpha = "<< alpha<<", beta = "<< beta <<"\n";
    }

    compute_basis_gso(L);
    projected_basis_setup(begin_index, upper_index);
    ListDecoding_setup(Psize);
    Bucketing_setup(Psize);
    Simhash_setup();
    pool_setup(max_vec);

    long current_dim = 40;
    thread worker[num_threads];
    worker[0] = thread(work, ref(current_dim), max_vec, Psize, true);
    for (long i = 1; i < num_threads; i++){
        worker[i] = thread(work, ref(current_dim), max_vec, Psize, false);
    }
    for (long i = 0; i < num_threads; i++){
        worker[i].join();
    }

    clear_all();
}

int main(){
    ifstream data("dim60sd0-LLL.txt", ios::in);
    Mat<double> L;
    data >> L;
    LDGaussSieve(L, 0, 60, 1000000, 80, 0.44, 0.44, 1);
    return 0;
}




