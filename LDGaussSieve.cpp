#include <iostream>
#include <fstream>
#include <algorithm>
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


double gh_coeff(long n){
    double a = 1.0;
    if (n%2 == 0){
        long m = n/2;
        for (long i = 1; i < m+1; i++){
            a = a * pow(i,1.0/n);
        }
    }else{
        long m = (n-1)/2;
        for (long i = 0; i < m+1; i++){
            a = a * pow(i+0.5,1.0/n);
        }
        a = a * pow(3.14159265357989324,0.5/n);
    }
    a = a / sqrt(3.14159265357989324);
    return a;
}
inline float dot(float *a, float *b, long n){
	__m512 r0 = _mm512_setzero_ps();
	__m512 x0, y0;
	for (long i = 0; i < n/16; i++){
		x0 = _mm512_load_ps(a+16*i);
		y0 = _mm512_load_ps(b+16*i);
		r0 = _mm512_fmadd_ps(x0, y0, r0);
	}

	__m256 r256 = _mm256_add_ps(_mm512_castps512_ps256(r0), _mm512_extractf32x8_ps(r0, 1));
	__m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r256), _mm256_extractf32x4_ps(r256, 1));
	r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
	r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
	return _mm_cvtss_f32(r128);
}
inline void add(float *a, float *b, long n){
	__m512 x0, y0;
	for (long i = 0; i < n/16; i++){
		x0 = _mm512_load_ps(a+i*16);
		y0 = _mm512_load_ps(b+i*16);
		_mm512_store_ps(a+16*i, _mm512_add_ps(x0,y0));
	}
}
inline void sub(float *a, float *b, long n){
	__m512 x0, y0;
	for (long i = 0; i < n/16; i++){
		x0 = _mm512_load_ps(a+i*16);
		y0 = _mm512_load_ps(b+i*16);
		_mm512_store_ps(a+16*i, _mm512_sub_ps(x0,y0));
	}
}
float dot_slow(float *a, float *b, long n){
    float x = 0.0;
    for (long i = 0; i < n; i++){
        x += a[i]*b[i];
    }
    return x;
}
float norm_slow(float *a, long n){
    float x = 0.0;
    for (long i = 0; i < n; i++){
        x += a[i]*a[i];
    }
    return x;
}

void print(double *v, long n){
    cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << v[i] << " ";
    }
    cout << v[n-1] << "]\n";
}
void printsqrt(double *v, long n){
    cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << sqrt(v[i]) << " ";
    }
    cout << sqrt(v[n-1]) << "]\n";
}
void print(float *v, long n){
    cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << v[i] << " ";
    }
    cout << v[n-1] << "]\n";
}
void print(uint32_t *v, long n){
     cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << v[i] << " ";
    }
    cout << v[n-1] << "]\n";
}
void print(int *v, long n){
    cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << v[i] << " ";
    }
    cout << v[n-1] << "]\n";
}
void print(long long *v, long n){
     cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << v[i] << " ";
    }
    cout << v[n-1] << "]\n";
}
void print(double **b, long n, long m){
    cout << "[";
    for (long i = 0; i < n; i++){
        print(b[i], m);
    }
    cout << "]";
}
void print(float *b, long n, long m){
    cout << "[";
    for (long i = 0; i < n; i++){
        print(b+i*m, m);
    }
    cout << "]";
}
void print(uint32_t *b, long n, long m){
    cout << "[";
    for (long i = 0; i < n; i++){
        print(b+i*m, m);
    }
    cout << "]";
}
void print(long long *b, long n, long m){
    cout << "[";
    for (long i = 0; i < n; i++){
        print(b+i*m, m);
    }
    cout << "]";
}
void print(uint16_t *v,long n){
    cout << "[";
    for (long i = 0; i < n-1; i++){
        cout << v[i] << " ";
    }
    cout << v[n-1] << "]\n";
}
void printvec(float **a, long n){
    cout << "[";
    for (long i = 0; i < n-1; i++){
        if (a[i] != NULL)
            cout << sqrt(a[i][-1]) << " ";
        else
            cout << "NULL ";
    }
    if (n > 0){
        if (a[n-1] != NULL)
            cout << sqrt(a[n-1][-1]);
        else
            cout << "NULL";
    }
    cout << "]\n";
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

RR dot(Vec<RR>& u, Vec<RR>& v){
    RR x;
    long n = u.length();
    for (long i = 0; i < n; i++){
        x += u[i]*v[i];
    }
    return x;
}
void red(Vec<RR>& u, Vec<RR>& v, RR& q){
    long n = u.length();
    for (long i = 0; i < n; i++){
        u[i] -= v[i]*q;
    }
    return;
}

long discrete_gaussian(double sigma2){
    long x, B = floor(5*sqrt(sigma2))+1;
    double p,r,y = 0.5/sigma2;
    while(1){
        x = ((long) rand()) % (2*B+1);
        x -= B;
        p = exp(-x*x*y);
        r = ((long) rand() % 100000) / 100000;
        if (r < p) return x;
    }
}
long discrete_gaussian(double t,double sigma2){
  long x,tt = floor(t+.5);
  double p,r,dt = t-tt;
  long B = floor(sqrt(sigma2))+1;
  while(1){
    x = ((long) rand()) % (2*B+1);
    x -= B;
    p = exp(-((x-dt)*(x-dt)) / (2*sigma2));
    r = ((int) rand() % 100000) / 100000;
    if (r<p) return x+tt;
  }
}
void spherical_sampling(float *res, long n){
    long i,j;
    for(i = 0; i < n; i++){
        res[i] = (float) discrete_gaussian(1000000);
    }
    float sqn=norm_slow(res, n);
    float x = 1/sqrt(sqn);
    for(i = n - 1; i >= 0; i--){
        res[i] *= x;
    }

}
void improve_poly(float* Polytope, long m, long Psize, float delta){
	float Norm;
	for (long i = 0; i < Psize; i++){
		for (long j = 0; j < Psize; j++){
			float scal = dot_slow(&Polytope[i*m], &Polytope[j*m], m);
			Norm = sqrt(norm_slow(&Polytope[i*m], m));
			if (scal<0){
                scal = 0;
            }else{
                scal /= Norm;
            }
			for (int k = 0; k < m; k++){
                Polytope[i*m +k] -= scal * delta * Polytope[j*m+k];
            }
		}
		Norm = sqrt(norm_slow(&Polytope[i*m], m));
        float x = 1/Norm;
		for (int k = 0; k < m; k++){
            Polytope[i*m+k] *= x;
        }
	}
}
void compute_Simhash(float *res, uint32_t* compress_pos){       
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

void sampling_on_dim(long current_dim, float *res, long n, double **b, double **miu, uint32_t* compress_pos){
    int coeff[n];
    int c = current_dim/2;
    for (long i = c; i < current_dim; i++){
        coeff[i] = (rand() % 5) -2;
    }
    for (long i = 0; i < n; i++){
        res[i] = 0;
    }
    for (long i = c; i < current_dim; i++){
        for (long j = 0; j <= i; j++){
            res[j] = res[j] + coeff[i]*miu[i][j];
        }
    }
    for (long i = c-1; i >= 0; i--){
        coeff[i] = round(-res[i]);
        for (long j = 0; j <= i; j++){
            res[j] = res[j] + coeff[i] * miu[i][j];
        }
    }
    for (long i = 0; i < n; i++){
        res[i] = 0.0;
    }
    for (long i = 0; i < current_dim; i++){
        for (long j = 0; j < n; j++){
            res[j] += coeff[i]*b[i][j];
        }
    }
    res[-1] = norm_slow(res, n);
    compute_Simhash(res, compress_pos);
}
void gaussian_sampling_on_dim(long current_dim, float *res, long n, double **b, double **miu, uint32_t* compress_pos, double *B){
    int coeff[n];
    double sigma2 = B[current_dim/2];
    for (long i = 0; i < n; i++){
        res[i] = 0;
    }

    for (long i = current_dim - 1; i >= 0; i--){
        coeff[i] = discrete_gaussian(res[i],sigma2/B[i]);
        for(long j = 0; j < i; j++){
            res[j] -= coeff[i]*miu[i][j];
        }
    }
    for (long i = 0; i < n; i++){
        res[i] = 0.0;
    }
    for (long i = 0; i < current_dim; i++){
        for (long j = 0; j < n; j++){
            res[j] += coeff[i]*b[i][j];
        }
    }
    //print(&coeff[0],current_dim);
    res[-1] = norm_slow(res, n);
    compute_Simhash(res, compress_pos);
}


void ListDecode(long *I, float *ptr, long n, long m, long Psize, float *Polytope, float *vec_tmp, float *InnerProd_tmp, long long *Sort, float delta, float *minsums, float *sums){
    for (long i = 0; i < n; i++){
        vec_tmp[i] = ptr[i];
    }
    float xx = 1/sqrt(norm_slow(ptr, n)*NUM_COMPONENT);
    for (long i = 0; i < n; i++){
        vec_tmp[i] *= xx;
    }

    for (long i = 0; i < NUM_COMPONENT; i++){
        for (long p = 0; p < Psize; p++){
            InnerProd_tmp[i*Psize + p] = (float)0.0;
            for (long k = 0; k < m; k++){
                InnerProd_tmp[i*Psize + p] += vec_tmp[m*i + k] * Polytope[m*p + k];
            }
        }
    }
    //print(InnerProd_tmp, NUM_COMPONENT, Psize);
    for (long i = 0; i < NUM_COMPONENT; i++){
        for (int p = 0; p < Psize; p++){
            Sort[i*Psize + p] = ((long) ceil(0.5 + InnerProd_tmp[i*Psize + p] * deux28)) << 32;    //4.4
            Sort[i*Psize + p] += p;
        }
        sort(&Sort[i*Psize], &Sort[(i+1)*Psize], greater<long>());
    }
    for (int i = 0; i < NUM_COMPONENT; i++){
		for (int p = 0; p < Psize; p++){
			InnerProd_tmp[i*Psize + p] = ((double) (Sort[i*Psize + p] >> 32)) / ((double) deux28);
			Sort[i*Psize + p] = (Sort[i*Psize + p] << 32) >> 32 ;
		}
	}
    //print(InnerProd_tmp, NUM_COMPONENT, Psize);
    //print(Sort, NUM_COMPONENT, Psize);
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
                        I[num_rel_buckets] = Psize*Psize*Psize*Sort[0*Psize + j0]
                                                + Psize*Psize*Sort[1*Psize + j1] 
                                                +       Psize*Sort[2*Psize + j2] 
                                                +             Sort[3*Psize + j3];
                    }			
                }
            }
        }
        I[0] = num_rel_buckets;
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
                        I[num_rel_buckets] = Psize*Psize*Sort[0*Psize + j0] 
                                            +       Psize*Sort[1*Psize + j1] 
                                            +             Sort[2*Psize + j2];			
                }
            }
        }
        I[0] = num_rel_buckets;
    }	
    
}
void Add_To_Bucket(long id, long *Iadd, uint32_t **Bucket, uint16_t *Bucket_max_size, uint16_t *Bucket_size){       
    for (long i = 1; i <= Iadd[0]; i++){
        if (Bucket_size[Iadd[i]] == Bucket_max_size[Iadd[i]]){
            Bucket[Iadd[i]] = (uint32_t *)realloc(Bucket[Iadd[i]], 4*(Bucket_max_size[Iadd[i]]+BUCKET_ENLARGE_SIZE));
            Bucket_max_size[Iadd[i]] += BUCKET_ENLARGE_SIZE;
        }
        Bucket[Iadd[i]][Bucket_size[Iadd[i]]] = id;
        Bucket_size[Iadd[i]]++;
    }
}
void Rem_From_Bucket(long id, long *Irem, uint32_t **Bucket, uint16_t *Bucket_max_size, uint16_t *Bucket_size){     
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
bool sieve_is_over(long current_dim, float **list, long num_list, long n, double *B, double& gh){
    long count = 0;
    for (long i = 0; i < num_list; i++){
        if (list[i] != NULL){
            if (list[i][-1] < 1.3333*gh){
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
Vec<double> compute_coeff(float *v, double **b, long n){
    Vec<double> vv;
    vv.SetLength(n);
    for (long i = 0; i < n; i++){
        vv[i] = v[i];
    }
    /*Vec<double> coeff;
    coeff.SetLength(n);
    for (long i = n-1; i >= 0; i--){
        coeff[i] = vv[i]/b[i][i];
        for (long j = i; j >= 0; j--){
            vv[j] -= coeff[i]*b[i][j];
        }
    }
    return coeff;*/
	return vv;
}

/* an implementation of Gausssieve boosted by LSF, SimHash, Progressive Sieve,
 * and avx-512 instructions, which will be future mutithreaded. and I will try
 * to minimize the cost of RAM. the parameters "Psize", "alpha" and "beta" has 
 * the same meaning in the paper https://eprint.iacr.org/2015/1128.pdf by Leo 
 * Ducas et al.*/
void LDGaussSieve(Mat<double>& L, long max_vec, long Psize, float alpha, float beta){
    srand(time(NULL));
    struct timeval start, end;
    double next_report = 5.0;
    gettimeofday(&start, NULL);
    long n = L.NumRows();
    long n8 = ((n+7)/8)*8;
    long n16 = ((n+15)/16)*16;


    /* compute the gso-data in RR. */
    RR::SetPrecision(150);
    Mat<RR> miu_RR;
    Mat<RR> b_star_RR;
    Vec<RR> B_RR;
    miu_RR.SetDims(n,n);
    B_RR.SetLength(n);
	b_star_RR.SetDims(n,L.NumCols());
    for (long i = 0; i < n; i++){
        for (long j = 0; j < L.NumCols(); j++){
            b_star_RR[i][j] = to_RR(L[i][j]);
        }
    }
    B_RR[0] = dot(b_star_RR[0],b_star_RR[0]);
	for (long i = 0; i < n-1; i++){
		for (long j = i+1; j < n; j++){
			miu_RR[j][i] = dot(b_star_RR[j],b_star_RR[i])/B_RR[i];
            red(b_star_RR[j],b_star_RR[i],miu_RR[j][i]);
		}
		B_RR[i+1] = dot(b_star_RR[i+1],b_star_RR[i+1]);
	}

    /* preparing data for fast sampling. */
    double **b = new double*[n];
    double **miu = new double*[n];
    double *b_store = (double *) calloc(1, 8*n8*n+64);
    double *miu_store = (double *) calloc(1, 8*n8*n+64);
    double *B = (double *) malloc(8*n);
    double *b_start = (double *) ((((long long)(b_store)-1)/64+1)*64);
    double *miu_start = (double *) ((((long long)(miu_store)-1)/64+1)*64);
    for (long i = 0; i < n; i++){
        b[i] = b_start + i * n8;
        miu[i] = miu_start + i * n8;
    }
    for (long i = 0; i < n; i++){
        miu[i][i] = 1.0;
       // b[i][i] = 1.0;
        B[i] = conv<double>(B_RR[i]);
        for (long j = 0; j < i; j++){
            miu[i][j] = conv<double>(miu_RR[i][j]);
     //       b[i][j] = miu[i][j];
        }
    }
    /*for (long i = 0; i < n; i++){
        double x = conv<double>(sqrt(B_RR[i]));
        for (long j = i; j < n; j++){
            b[j][i] *= x;
        }
    }*/
    for (long i = 0 ; i < n; i++){
        for (long j = 0; j < n; j++){
            b[i][j] = L[i][j];
        }
    }
    if (SHOW_DETAILS){
        cout << "gso is done! B = \n";
        printsqrt(B, n);
        //print(b,n,n);
    }


    /* ListDecoding setup. */
    if (n%NUM_COMPONENT){
        cerr << "n must be a multiple of num_comp" << endl;
		exit(0);
    }
    long m = n/NUM_COMPONENT;
    float* Polytope = new float[m*Psize];
    float* InnerProd_tmp = new float[NUM_COMPONENT*Psize];
    long long* Sort = new long long[NUM_COMPONENT*Psize];
    float* vec_tmp = new float[n];
    for (long i = 0; i < Psize; i++){
        spherical_sampling(Polytope+i*m, m);
    }
    for (int i = 0; i < 100; i++){
        improve_poly(Polytope, m, Psize, 0.1);
    }
    float *sums = new float[NUM_COMPONENT];
	float *minsums = new float[NUM_COMPONENT];
    if (SHOW_DETAILS){
        cout << "Listdecode setup done! Psize = "<<Psize << ", component length = "<< m << "\n";
    }

    /* Bucketing setup. */
    long num_buckets = 1;
	for (int i = 0; i < NUM_COMPONENT; i++){
		num_buckets*=Psize;
	}
    long TotalSize = 0;
    uint32_t** Bucket = new uint32_t*[num_buckets];
    for (long i = 0; i < num_buckets; i++){
        Bucket[i] = new uint32_t[BUCKET_INIT_SIZE];
    }
    uint16_t* Bucket_size = new uint16_t[num_buckets];
    uint16_t* Bucket_max_size = new uint16_t[num_buckets];
    for (long i = 0; i < num_buckets; i++){
        Bucket_size[i] = 0;
        Bucket_max_size[i] = BUCKET_INIT_SIZE;
    }
    long *rel_Buckets = new long[num_buckets];
    long *Irem = new long[num_buckets];
    long *Iadd = new long[num_buckets];
    long num_rel_buckets = 0;
    if (SHOW_DETAILS){
        cout << "bucketing setup done! number of bucket = " << num_buckets <<endl;
    }

    /* Simhash setup. */
    uint32_t* compress_pos = new uint32_t[128*4];
    for (long i = 0; i < 128; i++){
        compress_pos[i*4+0] = i % n;
        compress_pos[i*4+1] = rand() % n;
        compress_pos[i*4+2] = rand() % n;
        compress_pos[i*4+3] = rand() % n;
    }


    /* pool setup. */
    long vec_size = n16+16;
    long vec_length = n16;
    float *vec_store = (float *) calloc (4*vec_size*max_vec+128, 1);
    float *vec_start = (float *) ((((long long)(vec_store)-1)/64+2)*64);
    float **list = (float **) malloc (8*max_vec);
    float **empty = (float **) malloc (8*max_vec);
    float **update = (float **) malloc (8*max_vec);
    long *null_index = (long *) malloc(8*max_vec);
    long num_null_in_list = 0;
    long num_update = 0;
    long num_empty = 0;
    long num_list = 0;
    long num_used = 0;
    if (SHOW_DETAILS){
        cout << "pool setup done! num_used = "<< num_used << ", num_null_in_list = "<< num_null_in_list << ", num_empty = "<<num_empty<<endl;
    }

    
    /* main loop. */
    float min_norm = 900000000;
    float* min_vec;
    long num_reduce = 0;
    long num_try_to_reduce = 0;
    long num_pass_Simhash = 0;
    long current_dim = max(n-30, 40);
    double gh = gh_coeff(current_dim);
    gh = gh * gh;
    for (long i = 0; i < current_dim; i++){
        gh = gh * pow(B[i], 1.0/(current_dim));
    }

    while ((current_dim <= n) && (num_used < max_vec-1)){
        /* do sampling.*/
        if (num_empty > 0){
            update[num_update] = empty[num_empty-1];
            num_empty--;
            gaussian_sampling_on_dim(current_dim, update[num_update], n, b, miu, compress_pos,B);
            num_update++;
        }else{
            update[num_update] = vec_start+vec_size*num_used;
            num_used++;
            gaussian_sampling_on_dim(current_dim, update[num_update], n, b, miu, compress_pos,B);
            num_update++;            
        }

        /* do reduce. */
        while (num_update > 0){
            gettimeofday(&end, NULL);
            if ((end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0 > next_report){
                next_report+=5;
                cout << "current sieving dimension = "<<current_dim <<"\n";
                cout << "num_try_to_reduce = "<< num_try_to_reduce << ", num_pass_Simhash = "<< num_pass_Simhash << ", num_reduce = "<< num_reduce<< ", current min = "<<sqrt(min_norm) <<"\n"; 
                //cout << "shortest vec = "<< compute_coeff(min_vec, b, n) << endl;
                cout << "min_vec is reduced "<< *((long *)(&min_vec[-10]))<< " times\n";
                cout << "TotalSize = " << TotalSize <<", num_used = "<< num_used << ", num_null_in_list = "<< num_null_in_list << ", num_empty = "<<num_empty<<endl<<endl;
            }
            bool ptr_is_reduced = false;
            float *ptr = update[num_update-1];
            num_update--;


            ListDecode(rel_Buckets, ptr, n, m, Psize, Polytope, vec_tmp, InnerProd_tmp, Sort, beta, minsums, sums);            
            num_rel_buckets = rel_Buckets[0];
            //cerr << num_rel_buckets << " ";
            for (long i = 1; i <= num_rel_buckets; i++){
                if (ptr_is_reduced) break;
                for (long j = 0; j < Bucket_size[rel_Buckets[i]]; j++){     
                    float *pptr = list[Bucket[rel_Buckets[i]][j]];   
                    if ((long)pptr % 64) cerr << "warning ";    
                    if (pptr == NULL) continue;                             
                    num_try_to_reduce++;
                    long w = __builtin_popcountl((*((uint64_t *)(&ptr[-8]))) ^ (*((uint64_t *)(&pptr[-8]))));
                    w += __builtin_popcountl((*((uint64_t *)(&ptr[-6]))) ^ (*((uint64_t *)(&pptr[-6]))));
                    if (w < XOR_POPCNT_THRESHOLD || w > (128 - XOR_POPCNT_THRESHOLD)){
                        num_pass_Simhash++;                                 
                        float x = dot(ptr, pptr, n16);                  
                        float f = abs(x+x);
                        if (f > ptr[-1]){
                            num_reduce++;
                            ListDecode(Irem, pptr, n, m, Psize, Polytope, vec_tmp, InnerProd_tmp, Sort, alpha, minsums, sums);
                            if (x < 0){
                                add(pptr, ptr, n16);
                            }else{
                                sub(pptr, ptr, n16);
                            }
                            ((long *)(&pptr[-10]))[0] += ((long *)(&ptr[-10]))[0]+1;
                            pptr[-1] = dot(pptr, pptr, n16)+100.0;
                            if (pptr[-1] < 200.0){
                                if (x > 0){
                                    for (long i = 0; i < n; i++){
                                        pptr[i] = ptr[i];
                                    }
                                }else{
                                    for (long i = 0; i < n; i++){
                                        pptr[i] = -ptr[i];
                                    }
                                }
                                pptr[-1] = ptr[-1];
                                empty[num_empty] = ptr;
                                num_empty++;
								ptr_is_reduced = true;
                                break; 
                            }
                            if (pptr[-1] < min_norm) {
                                min_norm = pptr[-1];
                                min_vec = pptr;
                            }
                            long id = Bucket[rel_Buckets[i]][j];
                            list[id] = NULL;
                            TotalSize -= Irem[0];
                            Rem_From_Bucket(id, Irem, Bucket, Bucket_max_size, Bucket_size);
                            compute_Simhash(pptr, compress_pos);
                            update[num_update] = pptr;
                            num_update++;
                            null_index[num_null_in_list] = id;
                            num_null_in_list++;
                        }else if (f > pptr[-1]){
                            num_reduce++;
                            if (x < 0){
                                add(ptr, pptr, n16);
                            }else{
                                sub(ptr, pptr, n16);
                            }
                            ((long *)(&ptr[-10]))[0]+=((long *)(&pptr[-10]))[0]+1;
                            ptr[-1] = dot(ptr, ptr, n16)+100.0;
                            if (ptr[-1] < min_norm) {
                                min_norm = ptr[-1];
                                min_vec = ptr;
                            }
                            compute_Simhash(ptr, compress_pos);
                            update[num_update] = ptr;
                            num_update++;
                            ptr_is_reduced = true;
                            break;
                        }
                    }
                }
            }
            if (!ptr_is_reduced){
                for (long i = 0; i < n; i++){
                    ptr[i] = -ptr[i];
                }
                compute_Simhash(ptr, compress_pos);
                ListDecode(rel_Buckets, ptr, n, m, Psize, Polytope, vec_tmp, InnerProd_tmp, Sort, beta, minsums, sums);
                num_rel_buckets = rel_Buckets[0];
                for (long i = 1; i <= num_rel_buckets; i++){                                
                    if (ptr_is_reduced) break;
                    for (long j = 0; j < Bucket_size[rel_Buckets[i]]; j++){
                        float *pptr = list[Bucket[rel_Buckets[i]][j]];
                        if ((long)pptr % 64) cerr << "warning ";                        
                        if (pptr == NULL) continue;                                         
                        num_try_to_reduce++;                                                
                        long w = __builtin_popcountl((*((uint64_t *)(&ptr[-8]))) ^ (*((uint64_t *)(&pptr[-8]))));
                        w += __builtin_popcountl((*((uint64_t *)(&ptr[-6]))) ^ (*((uint64_t *)(&pptr[-6]))));
                        if (w < XOR_POPCNT_THRESHOLD || w > (128 - XOR_POPCNT_THRESHOLD)){
                            num_pass_Simhash++;                                             
                            float x = dot(ptr, pptr, n16);                                  
                            float f = abs(x+x);
                            if (f > ptr[-1]){
                                num_reduce++;
                                ListDecode(Irem, pptr, n, m, Psize, Polytope, vec_tmp, InnerProd_tmp, Sort, alpha, minsums, sums);
                                if (x < 0){
                                    add(pptr, ptr, n16);
                                }else{
                                    sub(pptr, ptr, n16);
                                }
                                ((long *)(&pptr[-10]))[0]+=((long *)(&ptr[-10]))[0]+1;
                                pptr[-1] = dot(pptr, pptr, n16)+100.0;
                                if (pptr[-1] < 200.0){
                                    if (x > 0){
                                        for (long i = 0; i < n; i++){
                                            pptr[i] = ptr[i];
                                        }
                                    }else{
                                        for (long i = 0; i < n; i++){
                                            pptr[i] = -ptr[i];
                                        }
                                    }
                                    pptr[-1] = ptr[-1];
                                    empty[num_empty] = ptr;
                                    num_empty++;
									ptr_is_reduced = true;
                                    break; 
                                }
                                if (pptr[-1] < min_norm) {
                                    min_norm = pptr[-1];
                                    min_vec = pptr;
                                }
                                long id = Bucket[rel_Buckets[i]][j];
                                list[id] = NULL;
                                TotalSize -= Irem[0];
                                Rem_From_Bucket(id, Irem, Bucket, Bucket_max_size, Bucket_size);
                                compute_Simhash(pptr, compress_pos);
                                update[num_update] = pptr;
                                num_update++;
                                null_index[num_null_in_list] = id;
                                num_null_in_list++;
                            }else if (f > pptr[-1]){
                                num_reduce++;
                                if (x < 0){
                                    add(ptr, pptr, n16);
                                }else{
                                    sub(ptr, pptr, n16);
                                }
                                ((long *)(&ptr[-10]))[0]+=((long *)(&pptr[-10]))[0]+1;
                                ptr[-1] = dot(ptr, ptr, n16)+100.0;
                                if (ptr[-1] < min_norm) {
                                    min_norm = ptr[-1];
                                    min_vec = ptr;
                                }
                                compute_Simhash(ptr, compress_pos);
                                update[num_update] = ptr;
                                num_update++;
                                ptr_is_reduced = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (!ptr_is_reduced){
                long id;
                if (num_null_in_list > 0){
                    num_null_in_list--;
                    id = null_index[num_null_in_list];
                }else{
                    id = num_list;
                    num_list++;
                }
                list[id] = NULL;
                Add_To_Bucket(id, rel_Buckets, Bucket, Bucket_max_size, Bucket_size);
                TotalSize+=rel_Buckets[0];
                list[id] = ptr;
            }
        }
        if (sieve_is_over(current_dim, list, num_list, n, B, gh)){
            if (current_dim == n){
                current_dim++;
            }else{
                current_dim = min(current_dim+1, n);
            }


        }
    }
    if (true){
        cout << "shortest vec = "<< compute_coeff(min_vec, b, n) << endl;
        cout << "min_vec is reduced "<< *((long *)(&min_vec[-10]))<< " times\n";
        cout << "num_try_to_reduce = "<< num_try_to_reduce << ", num_pass_Simhash = "<< num_pass_Simhash << ", num_reduce = "<< num_reduce<< ", current min = "<<sqrt(min_norm) <<"\n";
        cout << "TotalSize = " << TotalSize <<", num_used = "<< num_used << ", num_null_in_list = "<< num_null_in_list << ", num_empty = "<<num_empty<<endl<<endl;
        gettimeofday(&end, NULL);
        cout << "total time = "<<(end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0<<"s\n";
    }


   

    delete[] sums;
    delete[] minsums;
    delete[] rel_Buckets;
    delete[] compress_pos;
    delete[] Bucket;
    delete[] Bucket_size;
    delete[] Bucket_max_size;
    delete[] Polytope;
    delete[] InnerProd_tmp;
    delete[] Sort;
    delete[] vec_tmp;
    delete[] b;
    delete[] miu;
    free(vec_store);
    free(list);
    free(empty);
    free(update);
    free(null_index);
    free(b_store);
    free(miu_store);
    free(B);
}




int main(){
    ifstream data("../Leo/LDSieve/challenges/dim69sd0-LLL.txt", ios::in);
    Mat<double> L;
    data >> L;
    LDGaussSieve(L, 1000000, 124, 0.44, 0.44);
    return 0;
}
