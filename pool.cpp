#include "pool.h"

struct cvec_for_sort{
    __attribute__ ((packed)) long a[6];
};
inline bool cmp_cvec(cvec_for_sort a, cvec_for_sort b){
    return (((float *)&a.a[4])[0] < ((float *)&b.a[4])[0]);
}


//basic operations
Pool::Pool(){
    threadpool.resize(num_threads);
}
Pool::Pool(Basis& bs){
    basis_setup(bs);
    threadpool.resize(num_threads);
}
Pool::~Pool(){
    clear_all();
}
void Pool::clear_pool(){
    if (cvec_store != NULL){
        free(cvec_store);
        cvec_store = NULL;
    }
    if (vec_store != NULL){
        free(vec_store);
        vec_store = NULL;
    }
    num_vec = 0;
}
void Pool::clear_all(){
    if (cvec_store != NULL){
        free(cvec_store);
        cvec_store = NULL;
    }
    if (vec_store != NULL){
        free(vec_store);
        vec_store = NULL;
    }
    if (compress_pos != NULL){
        free(compress_pos);
        compress_pos = NULL;
    }
    if (b_store != NULL){
        free(b_store);
        b_store = NULL;
    }
    if (b_local_store != NULL){
        free(b_local_store);
        b_local_store = NULL;
    }
    if (miu_store != NULL){
        free(miu_store);
        miu_store = NULL;
    }
    if (b != NULL){
        delete[] b;
        b = NULL;
    }
    if (b_local != NULL){
        delete[] b_local;
        b_local = NULL;
    }
    if (miu != NULL){
        delete[] miu;
        miu = NULL;
    }
    if (B != NULL){
        delete[] B;
        B = NULL;
    }
}

//setup
void Pool::Simhash_setup(){
    compress_pos = new uint32_t[256*6];
    for (long i = 0; i < 256; i++){
        compress_pos[i*6+0] = i % MSD;
        compress_pos[i*6+1] = rand() % MSD;
        compress_pos[i*6+2] = rand() % MSD;
        compress_pos[i*6+3] = rand() % MSD;
        compress_pos[i*6+4] = rand() % MSD;
        compress_pos[i*6+5] = rand() % MSD;
    }
}
void Pool::set_num_threads(long n){
    num_threads = n;
    threadpool.resize(num_threads);
}
void Pool::basis_setup(Basis& bs){
    bs.compute_basis_gso();
    dim = bs.b_RR.NumRows();
    long dim8 = ((dim+7)/8)*8;
    
    B = new double[dim];
    b = new double*[dim];
    miu = new double*[dim];
    b_store = (double *) calloc(dim*dim8*8+64, 1);
    miu_store = (double *) calloc(dim*dim8*8+64, 1);
    double *b_start = (double *) ((((long)(b_store)-1)/64+1)*64);
    double *miu_start = (double *) ((((long)(miu_store)-1)/64+1)*64);
    for (long i = 0; i < dim; i++){
        b[i] = b_start + i * dim8;
        miu[i] = miu_start + i * dim8;
    }

    basis = &bs;
    for (long i = 0; i < dim; i++){
        B[i] = conv<double>(bs.B_RR[i]);
        for (long j = 0; j <= i; j++){
            miu[i][j] = conv<double>(bs.miu_RR[i][j]);
            b[i][j] = miu[i][j];
        }
    }

    for (long i = 0; i < dim; i++){
        double x = sqrt(B[i]);
        for (long j = i; j < dim; j++){
            b[j][i] *= x;
        }
    }
}
void Pool::set_basis(Basis& bs){
    clear_all();
    basis_setup(bs);
}
void Pool::set_MSD(long msd){
    MSD = msd;
    vec_length = ((MSD-1)/16+1)*16;
    int_bias = ((MSD-1)/32+1)*16+16;
    vec_size = vec_length+int_bias; 
    Simhash_setup();
}
void Pool::set_max_pool_size(long N){
    max_pool_size = N;
    vec_store = (float *) calloc(4*vec_size*max_pool_size+512, 1);
    vec = (float *) ((((long)(vec_store)-1)/64+8)*64);
    cvec_store = (long *) calloc(8*cvec_size*max_pool_size+16, 1);
    cvec = (long *) ((((long)(cvec_store)-1)/16+1)*16);
}
void Pool::set_sieving_context(long l, long r){
    index_l = l;
    index_r = r;
    CSD = r-l;
    compute_gh2();
    uid.reset_hash_function(CSD);
    update_b_local();
}
void Pool::update_b_local(){
    if (b_local_store != NULL){
        free(b_local_store);
        b_local_store = NULL;
    }
    if (b_local != NULL){
        delete[] b_local;
        b_local = NULL;
    }
    long CSD16 = ((CSD+15)/16)*16;
    b_local = new float*[CSD];
    b_local_store = (float *) calloc(CSD*CSD16*4+64, 1);
    float *b_local_start = (float *) ((((long)(b_local_store)-1)/64+1)*64);
    for (long i = 0; i < CSD; i++){
        b_local[i] = b_local_start + i * CSD16;
    }
    for (long i = 0; i < CSD; i++){
        for (long j = 0; j <= i; j++){
            b_local[i][CSD-1-j] = b[i+index_l][j+index_l];
        }
    }
}
void Pool::compute_gh2(){
    double detn2 = 1.0;
    for (long i = index_l; i < index_r; i++){
        detn2 *= pow(B[i], 1.0/CSD);
    }
    gh2 = detn2 * pow(gh_coeff(CSD), 2.0);
}

//pool operations
void Pool::show_pool_status(){
    std::cout << "current pool status:\n";
    std::cout << "basis dim = "<< dim << ", basis distvec = \n";
    printsqrt(B, dim);
    std::cout << "max sieving dimension = "<< MSD << ", current sieving dimension = "<< CSD<< "\n";
    std::cout << "current sieving context = ["<< index_l << ","<< index_r<< "], gh = "<< sqrt(gh2) << "\n";
    std::cout << "pool_size = "<< num_vec;
    std::cout << ", uid table has "<< uid.size() << " elements\n";
}
void Pool::print_random_vec(long num){
    std::cout << "the following is " << num << " random vector in the pool\n";
    for (long i = 0; i < num; i++){
        long index = rand()%num_vec;
        print_vec(vec+index*vec_size, CSD, int_bias);
    }
}
void Pool::show_vec_length(){
    std::cout << "[";
    for (long i = 0; i < num_vec-1; i++){
        std::cout << sqrt(*((float *)(cvec+i*cvec_size+4))) << " ";
    }
    std::cout << sqrt(*((float *)(cvec+(num_vec-1)*cvec_size+4))) << "]\n";
}
bool Pool::gaussian_sampling(long N){
    if (N > max_pool_size){
        std::cerr << "N is larger than max_pool_size, gaussian sampling failed!\n";
        return 1;
    }
    if (N < num_vec){
        std::cerr << "N is smaller than current pool size, gaussian sampling not done!\n";
        return 1;
    }
    bool success = true;
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++){
        gaussian_sampler R(thread);
        for (long i = num_vec+thread; i < N; i+=num_threads){
            if (!success) continue;
            long count = 0;
            do {
                count++;
                if (count > 30){
                    std::cerr << "sampling always get collision, aborted.\n";
                    success = false;
                    break;
                }
                gaussian_sampling(vec+i*vec_size, cvec+i*cvec_size, R);
            }while(!uid.insert_uid(*((uint64_t *)(&((vec+i*vec_size)[-4])))));
        }
    }
    if (success){
        num_vec = N;
        return 0;
    }
    return 1;
}
void Pool::gaussian_sampling(float *res, long *cres, gaussian_sampler &R){
    set_zero(res-int_bias, vec_size);
    int coeff[CSD];
    double sigma2 = B[(index_l+index_r)/2];
    for (long i = CSD - 1; i >= 0; i--){
        coeff[i] = R.discrete_gaussian(res[i],sigma2/B[i+index_l]);
        for(long j = 0; j < i; j++){
            res[j] -= coeff[i]*miu[i+index_l][j+index_l];
        }
    }
    for (long i = 0; i < CSD; i++){
        ((short*)(&res[-int_bias]))[i] = coeff[CSD-1-i];
    }
    for (long i = CSD; i < MSD; i++){
        ((short*)(&res[-int_bias]))[i] = 0;
    }
    compute_vec(res);
    compute_cvec(res, cres);
}
void Pool::shrink(long N){
    if (N >= num_vec) {
        std::cerr << "N should be smaller than num_vec, shrink aborted.\n";
        return;
    }
    sort_cvec();
    long tindex = N;
    for (long i = 0; i < N; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = (float *) cptr[5];
        if ((ptr-vec)<vec_size*N)continue;
        float *dst;
        long *cdst;
        do{ 
            if (tindex >= num_vec){
                std::cerr << "shrink failed! something must be wrong!\n";
                exit(1);
            }
            cdst = cvec + tindex * cvec_size;
            dst = (float *)(cdst[5]);
            tindex++;
        }while((dst -vec)>=vec_size*N);
        ((float **)cptr)[5] = dst;
        copy_vec(dst, ptr, int_bias,vec_length);
    }
    num_vec = N;
    sorted_index = N;
}
void Pool::extend_left(){
    index_l--;
    CSD++;
    update_b_local();
    compute_gh2();
    uid.reset_hash_function(CSD);

    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        long *cres = cvec + i * cvec_size;
        float *res = (float *)(cres[5]);
        short *x = (short *)(&res[-int_bias]);
        float y = 0.0;
        for (long j = 0; j < CSD-1; j++){
            y += x[CSD-2-j]*b_local[j+1][CSD-1];
        }
        short y_ = round(y/b_local[0][CSD-1]);
        y = y - y_*b_local[0][CSD-1];
        x[CSD-1] = -y_;
        res[CSD-1] = y;
        compute_Simhash(res);
        compute_uid(res);
        res[-1] = norm(res, vec_length);
        *((uint64_t *)(&cres[0])) = *((uint64_t *)(&res[-16]));
        *((uint64_t *)(&cres[1])) = *((uint64_t *)(&res[-14]));
        *((uint64_t *)(&cres[2])) = *((uint64_t *)(&res[-12]));
        *((uint64_t *)(&cres[3])) = *((uint64_t *)(&res[-10]));
        *((float *)(&cres[4])) = res[-1];
        if (!uid.insert_uid(*((uint64_t *)(&(res[-4]))))){
            std::cerr << "uid collision while extend left, something must wrong, aborted.\n";
            int *a = 0x0;
            std::cout << a << std::endl; 
        }
    }
    sorted_index = 0;
}
void Pool::shrink_left(){
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        short *x = (short *) (ptr - int_bias);
        x[CSD -1] = 0;
    }
    index_l++;
    CSD--;
    compute_gh2();
    uid.reset_hash_function(CSD);
    update_b_local();
    sorted_index = 0;

    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        compute_uid(ptr);
        if (uid.insert_uid(*((uint64_t *)(&(ptr[-4]))))) continue;
        if (num_vec > i + 1){
            num_vec--;
            copy((short*)(ptr-int_bias), (short *)(vec + num_vec * vec_size -int_bias), coeff_block);
        }else{
            num_vec--;
            break;
        }
        i--;
    }

    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        long *cptr = cvec + i * cvec_size;
        compute_vec(ptr);
        compute_cvec(ptr, cptr);
    }
}
void Pool::insert(long index){
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    if (index > index_l){
        std::cerr << "insertion index larger than index_l, aborted\n";
        return;
    }
    if (index < 0){
        std::cerr<< "negetive insertion index, aborted\n";
        return;
    }
    long min_index= -1;
    float min_norm = B[index]*0.995;
    pthread_spinlock_t min_lock = 1;


    long ID = index_l - index;
    long FD = index_r - index;
    long ID16 = ((ID+15)/16)*16;
    float **b_insert = new float*[FD];
    float *Bi = new float[ID];
    float *b_insert_store = (float *) calloc(FD*ID16*4+64, 1);
    float *b_insert_start = (float *) ((((long)(b_insert_store)-1)/64+1)*64);
    for (long i = 0; i < FD; i++){
        b_insert[i] = b_insert_start + i * ID16;
    }
    for (long i = 0; i < FD; i++){
        for (long j = 0; j < ID; j++){
            b_insert[i][j] = b[i+index][j+index];
        }
    }
    for (long i = 0; i < ID; i++){
        Bi[i] = 1.0/sqrt(B[index + i]);
    }
    

    //find the best insertion
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++){
        __attribute__ ((aligned (64))) float tmp_store[vec_size];
        float *tmp = &tmp_store[0];        
        for (long i = thread; i < num_vec; i+=num_threads){
            long *cptr = cvec + i * cvec_size;
            float *ptr = (float *) cptr[5];
            short *x = (short *)(ptr - int_bias);
            set_zero(tmp, vec_size);
            for (long j = 0; j < CSD; j++){
                red(tmp, b_insert[j+ID], -x[CSD - 1 - j], ID16);
            }
            for (long j = ID - 1; j >= 0; j--){
                red(tmp, b_insert[j], round(Bi[j]*tmp[j]), ID16); 
            }
            float norm1 = norm(tmp, ID16) + ptr[-1];
            if (norm1 < min_norm){
                pthread_spin_lock(&min_lock);
                if (norm1 < min_norm){
                    bool hasone = false;
                    for (long j = 0; j < CSD; j++){
                        if (x[j] == 1) hasone = true;
                        if (x[j] == -1) hasone = true;
                    }
                    if (hasone){
                        min_norm = norm1;
                        min_index = i;
                    }
                }
                pthread_spin_unlock(&min_lock);
            }
        }
    }
    if (min_index == -1){
        shrink_left();
        return;
    }
    __attribute__ ((aligned (64))) float tmp_store[vec_size];
    float *tmp = &tmp_store[0];
    long *cptr = cvec + min_index * cvec_size;
    float *ptr = (float *)cptr[5];
    short *x = (short *)(ptr - int_bias);
    Vec<RR> v;
    v.SetLength(dim);
    long rm_index;
    for (long i = 0; i < CSD; i++){
        if (abs(x[CSD - 1 - i]) == 1){
            rm_index = CSD - 1 - i;
            break;
        }
        if (i == (CSD - 1)){
            std::cerr << "something must be done while insertion!\n";
        }
    }
    set_zero(tmp, vec_size);
    for (long j = 0; j < CSD; j++){
        RR x1;
        x1 = conv<RR>(-x[CSD - 1 - j]);
        red(v, basis->b_RR[j+index_l], x1);
        red(tmp, b_insert[j+ID], -x[CSD - 1 - j], ID16);
    }
    for (long j = ID - 1; j >= 0; j--){
        long y = round(Bi[j]*tmp[j]);
        RR y1;
        y1 = conv<RR>(y);
        red(v, basis->b_RR[j + index_l - ID], y1);
        red(tmp, b_insert[j], y, ID16);
    }


    //compute the new coeff
    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        if (i == min_index) continue;
        long *cptr1 = cvec + i * cvec_size;
        float *ptr1 = (float *)cptr1[5];
        short *x1 = (short *)(ptr1-int_bias);
        if (x[rm_index] > 0){
            red(x1, x, x1[rm_index], coeff_block);
        }else{
            red(x1, x, -x1[rm_index], coeff_block);
        }
        for (long j = rm_index; j < CSD - 1; j++){
            x1[j] = x1[j+1];
        }
        x1[CSD-1] = 0;
    }
    red(x,x,1,coeff_block);


    index_l++;
    CSD--;    
    for (long i = CSD - rm_index+index_l-1; i > index; i--){
        basis->b_RR[i] = basis->b_RR[i-1];
    }
    basis->b_RR[index] = v;
    basis->size_reduce(index);
    basis->compute_basis_gso();
    for (long i = 0; i < dim; i++){
        B[i] = conv<double>(basis->B_RR[i]);
        for (long j = 0; j <= i; j++){
            miu[i][j] = conv<double>(basis->miu_RR[i][j]);
            b[i][j] = miu[i][j];
        }
    }
    for (long i = 0; i < dim; i++){
        double x = sqrt(B[i]);
        for (long j = i; j < dim; j++){
            b[j][i] *= x;
        }
    }

    compute_gh2();
    uid.reset_hash_function(CSD);
    update_b_local();

    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        compute_uid(ptr);
        if (uid.insert_uid(*((uint64_t *)(&(ptr[-4]))))) continue;
        if (num_vec > i + 1){
            num_vec--;
            copy((short*)(ptr-int_bias), (short *)(vec + num_vec * vec_size -int_bias), coeff_block);
        }else{
            num_vec--;
            break;
        }
        i--;
    }
    #pragma omp parallel for
    for (long i = 0; i < num_vec; i++){
        float *ptr = vec + i * vec_size;
        long *cptr = cvec + i * cvec_size;
        compute_vec(ptr);
        compute_cvec(ptr, cptr);
    }
    sorted_index = 0;
}
void Pool::LLL_ZZ(double delta){
    Mat<ZZ> b_ZZ;
    b_ZZ.SetDims(dim, dim);
    for (long i = 0; i < dim; i++){
        for (long j = 0; j < dim; j++){
            b_ZZ[i][j] = conv<ZZ>(floor(basis->b_RR[i][j]+0.5));
        }
    }
    LLL_QP(b_ZZ, delta, 0, 0, 0);
    for (long i = 0; i < dim; i++){
        for (long j = 0; j < dim; j++){
            basis->b_RR[i][j] = conv<RR>(b_ZZ[i][j]);
        }
    }
    basis->compute_basis_gso();
    for (long i = 0; i < dim; i++){
        B[i] = conv<double>(basis->B_RR[i]);
        for (long j = 0; j <= i; j++){
            miu[i][j] = conv<double>(basis->miu_RR[i][j]);
            b[i][j] = miu[i][j];
        }
    }
    for (long i = 0; i < dim; i++){
        double x = sqrt(B[i]);
        for (long j = i; j < dim; j++){
            b[j][i] *= x;
        }
    }
    num_vec = 0;
    sorted_index = 0;
}
void Pool::sort_cvec(){
    cvec_for_sort *start = (cvec_for_sort *)cvec;
    cvec_for_sort *middle = (cvec_for_sort *)(cvec + sorted_index*cvec_size);
    cvec_for_sort *end = (cvec_for_sort *)(cvec + num_vec*cvec_size);
    if (sorted_index == num_vec) return;
    if (sorted_index > num_vec/4){
        parallel_algorithms::sort(middle, end, cmp_cvec,threadpool);
        cvec_for_sort *tmp = new cvec_for_sort[num_vec];
        parallel_algorithms::merge(start, middle, middle, end, tmp, cmp_cvec, threadpool);
        parallel_algorithms::copy(tmp, tmp+num_vec, start, threadpool);
        delete[] tmp;
    }else{
        parallel_algorithms::sort(start, end, cmp_cvec, threadpool);
    }
    sorted_index = num_vec;   
}
bool Pool::sieve_is_over(double saturation_radius, double saturation_ratio){
    float goal = gh2 * saturation_radius;
    long goal_num = saturation_ratio * 0.5 * pow(saturation_radius, CSD/2.0);
    long num_in_sorted_part = 0;            //this name is a little confusing
    long up = sorted_index;
    long down = 0;
    while (up > down + 1){
        long mid = (up+down+1)/2;
        if (((float *)(cvec+mid*cvec_size+4))[0] < goal){
            down = mid;
        }else{
            up = mid;
        }
    }
    num_in_sorted_part = up;
    if (num_in_sorted_part >= goal_num) return true;
    for (long i = sorted_index; i < num_vec; i++){
        if (i % 100 == 0){
            if (num_in_sorted_part + num_vec-i < goal_num) return false;
            if (num_in_sorted_part >= goal_num) return true;
        }
        if (((float *)(cvec+i*cvec_size+4))[0] < goal) num_in_sorted_part++;
    }
    if (num_in_sorted_part >= goal_num) return true;
    return false;
}
bool Pool::check_pool_status(){
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + cvec_size * i;
        float *ptr = (float *)cptr[5];
        uint64_t u = *((uint64_t *)(&ptr[-4]));
        bool s0 = !(*((uint64_t *)(&cptr[0])) == *((uint64_t *)(&ptr[-16])));
        bool s1 = !(*((uint64_t *)(&cptr[1])) == *((uint64_t *)(&ptr[-14])));
        bool s2 = !(*((uint64_t *)(&cptr[2])) == *((uint64_t *)(&ptr[-12])));
        bool s3 = !(*((uint64_t *)(&cptr[3])) == *((uint64_t *)(&ptr[-10])));
        if (s0 || s1 || s2 || s3){
            std::cerr << "the "<< i << "-th vector, Simhash do not match!\n";
        }
        compute_vec(ptr);
        s0 = !(*((uint64_t *)(&cptr[0])) == *((uint64_t *)(&ptr[-16])));
        s1 = !(*((uint64_t *)(&cptr[1])) == *((uint64_t *)(&ptr[-14])));
        s2 = !(*((uint64_t *)(&cptr[2])) == *((uint64_t *)(&ptr[-12])));
        s3 = !(*((uint64_t *)(&cptr[3])) == *((uint64_t *)(&ptr[-10])));
        if (s0 || s1 || s2 || s3){
            std::cerr << "the "<< i << "-th vector, cvec Simhash wrong!\n";
        }
        if (u != *((uint64_t *)(&ptr[-4]))){
            std::cerr << "the "<< i << "-th vector, uid wrong!\n";
        }
        if (!uid.check_uid(*((uint64_t *)(&ptr[-4])))){
            std::cerr << "the "<< i << "-th vector, uid not in hashtable!\n";
        }
    }
}
void Pool::store(const char *file_name){
    std::ofstream hout (file_name, std::ios::out); 
    //hout << (basis->b_RR) << std::endl;
    hout << MSD << std::endl;
    hout << index_l << std::endl;
    hout << index_r << std::endl;
    hout << max_pool_size << std::endl;
    hout << "[";
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = (float *)cptr[5];
        ptr -= int_bias;
        short *x = (short *)ptr;
        hout << "[";
        for (long j = CSD-1; j > 0; j--){
            hout << x[j] << " ";
        }
        hout << x[0]<<"]\n";
    }
    hout << "]";
}
void Pool::store_vec(const char *file_name){
    /*if (index_l != 0){
        std::cerr << "index_l not equal to 0, aborted\n";
        return;
    }*/
    sort_cvec();
    std::ofstream hout (file_name, std::ios::out);
    hout << num_vec << std::endl;
    hout << CSD << std::endl;
    hout << basis->b_RR << std::endl;
    long dim16 = ((dim + 15)/16)*16;
    float **b_int = new float*[CSD];
    float *b_int_store = (float *) malloc(dim16 * CSD * 4+ 64);
    float *b_int_start = (float *) ((((long)(b_int_store)-1)/64+1)*64);
    for (long i = 0; i < CSD; i++){
        b_int[i] = b_int_start + dim16 * i;
    }
    for (long i = 0; i < CSD; i++){
        for (long j = 0; j < dim; j++){
            b_int[i][j] = conv<double>(basis->b_RR[i][j]);
        }
    }
    __attribute__ ((aligned (64))) float tmp_store[dim16];
    float *tmp = &tmp_store[0];
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = (float *)cptr[5];
        ptr -= int_bias;
        short *x = (short *)ptr;
        set_zero(tmp, dim16);
        for (long j = 0; j < CSD; j++){
            red(tmp, b_int[j], -x[CSD - 1 - j], dim16);
        }
        hout << "[";
        for (long j = 0; j < dim-1; j++){
            hout << tmp[j]<< " ";
        }
        hout << tmp[dim-1]<< "]\n";
    }
}
void Pool::load(const char *file_name){
    if (cvec_store != NULL){
        free(cvec_store);
        cvec_store = NULL;
    }
    if (vec_store != NULL){
        free(vec_store);
        vec_store = NULL;
    }
    if (b_local_store != NULL){
        free(b_local_store);
        b_local_store = NULL;
    }
    if (b_local != NULL){
        delete[] b_local;
        b_local = NULL;
    }
    if (compress_pos != NULL){
        free(compress_pos);
        compress_pos = NULL;
    }
    std::ifstream data (file_name, std::ios::in);
    data >> MSD;
    data >> index_l;
    data >> index_r;
    data >> max_pool_size;
    CSD = index_r-index_l;
    update_b_local();
    compute_gh2();
    uid.reset_hash_function(CSD);
    set_MSD(MSD);
    set_max_pool_size(max_pool_size);
    Mat<short> coeff;
    data >> coeff;
    num_vec = coeff.NumRows();
    for (long i = 0; i < num_vec; i++){
        long *cptr = cvec + i * cvec_size;
        float *ptr = vec + i * vec_size;
        short *x = (short *)(ptr-int_bias);
        for (long j = 0; j < CSD; j++){
            x[j] = coeff[i][CSD-1-j];
        }
        compute_vec(ptr);
        compute_cvec(ptr, cptr);
        uid.insert_uid(*((uint64_t *)(&ptr[-4])));
    }
}


coeff_buffer::coeff_buffer(long coeffsize, long maxsize){
    max_size = maxsize;
    coeff_size = coeffsize;
    buffer_store = (short *)malloc(2 * max_size * coeff_size + 64);
    buffer = (short *) ((((long)(buffer_store)-1)/64+1)*64);
}
coeff_buffer::~coeff_buffer(){
    if (buffer_store != NULL){
        free(buffer_store);
        buffer_store = NULL;
    }
}
void coeff_buffer::buffer_setup(long coeffsize, long maxsize){
    if (buffer_store == NULL){
        max_size = maxsize;
        coeff_size = coeffsize;
        buffer_store = (short *)malloc(2 * max_size * coeff_size + 64);
        buffer = (short *) ((((long)(buffer_store)-1)/64+1)*64);
    }else{
        printf("buffer already set up, nothing done!\n");
    }
}