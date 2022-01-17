#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "pool.h"
#include "BoostedSieve.h"

using namespace std;



int main(){
	long num_threads = 1;
 	omp_set_num_threads(num_threads);
	ifstream data ("L_100.txt", ios::in);
    Mat<ZZ> L;
    data >> L;
    Basis basis(L);
    Pool pool(basis);
	long max_pool_size = (long)(pow(4./3., 80*0.5)*3.8)+1;
	pool.set_num_threads(num_threads);
	pool.set_MSD(100);
	pool.set_max_pool_size(max_pool_size);
	pool.set_sieving_context(60,100);

	NVSieve_params p;
	p.alpha = 0.29;
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (long i = 0; i <= 40; i++){
		pool.gaussian_sampling((long)(pow(4./3., pool.CSD*0.5)*3.8));
		pool.NVSieve(p);
		if (i <=39){
			pool.extend_left();	
		}
	}
	gettimeofday(&end, NULL);
	cout << "sieving time = "<<(end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0<<"s\n";






	
	
    return 0;
}