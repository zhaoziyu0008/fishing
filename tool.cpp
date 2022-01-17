#include <NTL/LLL.h>
#include "tool.h"


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


/*---------------------prints---------------------*/
void print(float *v, long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << v[i] << " ";
    }
    std::cout << v[nn-1] << "]\n";
}
void print(double *v, long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << v[i] << " ";
    }
    std::cout << v[nn-1] << "]\n";
}
void print(short *v,long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << v[i] << " ";
    }
    std::cout << v[nn-1] << "]\n";
}
void print(int *v,long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << v[i] << " ";
    }
    std::cout << v[nn-1] << "]\n";
}
void print(uint16_t *v,long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << v[i] << " ";
    }
    std::cout << v[nn-1] << "]\n";
}
void print(uint32_t *v, long nn){
     std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << v[i] << " ";
    }
    std::cout << v[nn-1] << "]\n";
}
void printsqrt(double *v, long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << sqrt(v[i]) << " ";
    }
    std::cout << sqrt(v[nn-1]) << "]\n";
}
void printsqrt(float *v, long nn){
    std::cout << "[";
    for (long i = 0; i < nn-1; i++){
        std::cout << sqrt(v[i]) << " ";
    }
    std::cout << sqrt(v[nn-1]) << "]\n";
}
void print(float *b, long nn, long mm){
    std::cout << "[";
    for (long i = 0; i < nn; i++){
        print(b+i*mm, mm);
    }
    std::cout << "]";
}
void print(uint32_t *b, long nn, long mm){
    std::cout << "[";
    for (long i = 0; i < nn; i++){
        print(b+i*mm, mm);
    }
    std::cout << "]";
}
void print(double **b, long nn, long mm){
    std::cout << "[";
    for (long i = 0; i < nn; i++){
        print(b[i], mm);
    }
    std::cout << "]";
}
void print(float **b, long nn, long mm){
    std::cout << "[";
    for (long i = 0; i < nn; i++){
        print(b[i], mm);
    }
    std::cout << "]";
}
void print_vec(float *res, long CSD, long int_bias){
    std::cout << "[";
    for (long i = 0; i < CSD-1; i++){
            std::cout << res[i]<<" ";
    }
    std::cout << res[CSD-1]<<"]\n";
    short *x = (short *)(&res[-int_bias]);
    std::cout << "length = "<<res[-1]<<", uid = "<<*((uint64_t *)(&res[-4])) << ", SimHash = ["<< *((uint64_t *)(&res[-16])) << " ";
    std::cout <<*((uint64_t *)(&res[-14]))<<" "<<*((uint64_t *)(&res[-12]))<<" "<<*((uint64_t *)(&res[-10]))<<"], ";
    std::cout << "coeff = [";
    for (long i = 0; i < CSD-1; i++){
        std::cout << x[i]<<" ";
    }
    std::cout << x[CSD-1]<<"]\n";
}

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

/*---------------------random---------------------*/
uint64_t rand_uint64_t(){
    uint64_t a = rand();
    a = a + a + rand()%2;
    a = a * 2147483648;
    a += rand()*2;
    a += rand()%2;
    return a;
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



