#include "pool.h"

#define SHOW_3SIEVE_DETAILS false
#define XPC_TS_THRESHOLD 129
#define XPC_TS_BUCKET_THRESHOLD 129

void Pool::three_Sieve(three_Sieve_params params){
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    if (true){
        std::cerr << "begin 3Sieve on context ["<< index_l<<", "<<index_r << "], gh = "<< sqrt(gh2);
        std::cerr << ", pool size = "<< num_vec<<", "<<num_threads<<" threads will be used\n";
    }
    if (SHOW_3SIEVE_DETAILS){
        std::cerr << "begin NVSieve on context ["<< index_l<<", "<<index_r << "], gh = "<< sqrt(gh2);
        std::cerr << ", pool size = "<< num_vec<<", "<<num_threads<<" threads will be used\n";
    }
    sort_cvec();
    if (SHOW_3SIEVE_DETAILS){
        std::cerr << "initial sorting done!\n";
    }

    long count = -1;
    while (!sieve_is_over(params.saturation_radius, params.saturation_ratio)){
        count++;
        long goal_index = (long)(params.improve_ratio*num_vec);
        float goal_norm = ((float *)(cvec+goal_index*cvec_size+4))[0];
        if (SHOW_3SIEVE_DETAILS){
            std::cerr << "begin the "<< count << "-th epoch, goal norm = "<<sqrt(goal_norm) << std::endl;
        }


        //prepare the buffer
        coeff_buffer local_buffer[num_threads];
        coeff_buffer main_buffer(coeff_size, num_vec);
        for (long i = 0; i < num_threads; i++){
            local_buffer[i].buffer_setup(coeff_size, num_vec/num_threads);
        }
        if (SHOW_3SIEVE_DETAILS){
            std::cerr << "buffer prepared, begin to collect relations...\n";
        }


        //collect solutions
        bool rel_collection_stop = false;
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++){
            long ccount = 0;
            long avg_bucket_size = 0;
            long already_in = 0;
            long found = 0;
            long already_in3 = 0;
            long found3 = 0;
            while(!rel_collection_stop){
                ccount++;
                //centering
                long index = rand()%(num_vec);
                long *cptr = cvec+index*cvec_size;
                float *ptr = (float *) cptr[5];
                long **bucket = new long*[num_vec];
                float *dot1 = new float[num_vec];
                long num_element = 1;

                //bucketing
                long *cpptr = cvec;
                float ap2 = params.alpha*params.alpha*ptr[-1];
                for (long i = 0; i < num_vec; i++){
                    long w = __builtin_popcountl((*((uint64_t *)(&cptr[0]))) ^ (*((uint64_t *)(&cpptr[0]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[1]))) ^ (*((uint64_t *)(&cpptr[1]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[2]))) ^ (*((uint64_t *)(&cpptr[2]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[3]))) ^ (*((uint64_t *)(&cpptr[3]))));
                    if (w < XPC_TS_BUCKET_THRESHOLD || w > (256 - XPC_TS_BUCKET_THRESHOLD)){
                        float *pptr = (float *)cpptr[5];
                        float x = dot(pptr, ptr, vec_length);
                        float y = ap2*pptr[-1];
                        if (x*x>y){
                            if (x > pptr[-1]*0.95){
                                bucket[0] = cpptr;
                                dot1[0] = x;
                                continue;
                            }
                            bucket[num_element] = cpptr;
                            dot1[num_element] = x;
                            num_element++;
                        }
                    }
                    cpptr += cvec_size;
                }
                avg_bucket_size += num_element;
                if (SHOW_3SIEVE_DETAILS){
                    //printf("[thread %ld] the %ld-th bucketing done, centered at vector %ld, bucket size = %ld\n", thread, ccount, index, num_element);
                }
                long old_num_sol = local_buffer[thread].size;
                //search the reductions
                for (long i = 0; i < 1; i++){
                    if (local_buffer[thread].size>local_buffer[thread].max_size*0.9) break;
                    long *icptr = bucket[i];
                    float *iptr = (float *)icptr[5];
                    float ib = goal_norm-iptr[-1];
                    for (long j = i+1; j<num_element; j++){
                        long *jcptr = bucket[j];
                        long w = __builtin_popcountl((*((uint64_t *)(&icptr[0]))) ^ (*((uint64_t *)(&jcptr[0]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[1]))) ^ (*((uint64_t *)(&jcptr[1]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[2]))) ^ (*((uint64_t *)(&jcptr[2]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[3]))) ^ (*((uint64_t *)(&jcptr[3]))));
                        if (w < XPC_THRESHOLD || w > (256 - XPC_THRESHOLD)){
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(jptr, iptr, vec_length);
                            float y = abs(x);
                            float z = jptr[-1]-ib;
                            if (z < (y+y)){
                                found++;
                                uint64_t u;
                                if (x < 0.0){
                                    u = (*((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4])));
                                }else{
                                    u = (*((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4])));
                                }
                                if (uid.check_uid(u)) {
                                    already_in++;
                                    continue;
                                }
                                if (!uid.insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                if (x < 0.0){
                                    add(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_block);
                                }else{
                                    sub(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_block);
                                }
                                local_buffer[thread].size++;
                            }
                        }
                    }
                }
                for (long i = 1; i < num_element; i++){
                    if (local_buffer[thread].size>local_buffer[thread].max_size*0.9) break;
                    long *icptr = bucket[i];
                    float *iptr = (float *)icptr[5];
                    float ib = goal_norm-iptr[-1];
                    float it = (ptr[-1] + iptr[-1])*0.5;
                    float is = (goal_norm - ptr[-1] - iptr[-1])*0.5;
                    float di = dot1[i];
                    for (long j = i+1; j<num_element; j++){
                        long *jcptr = bucket[j];
                        long w = __builtin_popcountl((*((uint64_t *)(&icptr[0]))) ^ (*((uint64_t *)(&jcptr[0]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[1]))) ^ (*((uint64_t *)(&jcptr[1]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[2]))) ^ (*((uint64_t *)(&jcptr[2]))));
                        w += __builtin_popcountl((*((uint64_t *)(&icptr[3]))) ^ (*((uint64_t *)(&jcptr[3]))));
                        if (w < XPC_TS_THRESHOLD || w > (256 - XPC_TS_THRESHOLD)){
                            float *jptr = (float *)(jcptr[5]);
                            float x = dot(jptr, iptr, vec_length);
                            float y = abs(x);
                            float z = jptr[-1]-ib;
                            float w = jptr[-1]*0.5;
                            float dj = dot1[j];
                            if (x + abs(di-dj) > w + it){
                                if (count == 14){
                                    if (CSD == 70){
                                        //std::cerr<< "["<<index << " " << ((bucket[i]-cvec)/cvec_size) << " "<< ((bucket[j]-cvec)/cvec_size) << "] ";
                                    }
                                }
                                found3++;
                                uint64_t u;
                                if (di>dj){
                                    u = -*((uint64_t *)(&ptr[-4])) + *((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4]));
                                }else{
                                    u = +*((uint64_t *)(&ptr[-4])) + *((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4]));
                                }
                                if (uid.check_uid(u)) {
                                    already_in3++;
                                    continue;
                                }
                                if (!uid.insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                sub(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_block);
                                if (di>dj){
                                    sub(dst, (short*)(ptr-int_bias), coeff_block);
                                }else{
                                    add(dst, (short*)(ptr-int_bias), coeff_block);
                                }
                                local_buffer[thread].size++;
                            }
                            if (x + w < is + abs(di+dj)){
                                if (count == 14){
                                    if (CSD == 70){
                                        //std::cerr<< "["<<index << " " << ((bucket[i]-cvec)/cvec_size) << " "<< ((bucket[j]-cvec)/cvec_size) << "] ";
                                    }
                                }
                                found3++;
                                uint64_t u;
                                if (di>-dj){
                                    u = -*((uint64_t *)(&ptr[-4])) + *((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4]));
                                }else{
                                    u = +*((uint64_t *)(&ptr[-4])) + *((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4]));
                                }
                                if (uid.check_uid(u)) {
                                    already_in3++;
                                    continue;
                                }
                                if (!uid.insert_uid(u)) continue;
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                add(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_block);
                                if (di>-dj){
                                    sub(dst, (short*)(ptr-int_bias), coeff_block);
                                }else{
                                    add(dst, (short*)(ptr-int_bias), coeff_block);
                                }
                                local_buffer[thread].size++;
                            }
                            if (z < (y+y)){
                                if (count == 14){
                                    if (CSD == 70){
                                        //std::cerr<< "[" << ((bucket[i]-cvec)/cvec_size) << " "<< ((bucket[j]-cvec)/cvec_size) << " "<<(((iptr[-1]+jptr[-1]-2*x)<iptr[-1])||((iptr[-1]+jptr[-1]-2*x)<jptr[-1]))<<"] ";
                                        //std::cerr<< (((iptr[-1]+jptr[-1]-2*x)<iptr[-1])||((iptr[-1]+jptr[-1]-2*x)<jptr[-1]))<<" ";
                                    }
                                }
                                found++;
                                uint64_t u;
                                if (x < 0.0){
                                    u = (*((uint64_t *)(&iptr[-4]))+*((uint64_t *)(&jptr[-4])));
                                }else{
                                    u = (*((uint64_t *)(&iptr[-4]))-*((uint64_t *)(&jptr[-4])));
                                }
                                if (uid.check_uid(u)) {
                                    already_in++;
                                    continue;
                                }
                                if (!uid.insert_uid(u)) continue;
                                if (count == 14){
                                    if (CSD == 70){
                                        //std::cerr<< "[" << ((bucket[i]-cvec)/cvec_size) << " "<< ((bucket[j]-cvec)/cvec_size) << " "<<(((iptr[-1]+jptr[-1]-2*x)<iptr[-1])||((iptr[-1]+jptr[-1]-2*x)<jptr[-1]))<<"] ";
                                        //std::cerr<< (((iptr[-1]+jptr[-1]-2*x)<iptr[-1])||((iptr[-1]+jptr[-1]-2*x)<jptr[-1]))<<" ";
                                    }
                                }
                                short *dst = local_buffer[thread].buffer + coeff_size * local_buffer[thread].size;
                                if (x < 0.0){
                                    add(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_block);
                                }else{
                                    sub(dst, (short*)(iptr-int_bias), (short *)(jptr-int_bias), coeff_block);
                                }
                                local_buffer[thread].size++;
                            }

                        }
                    }
                }
                delete[] bucket;
                delete[] dot1;

                long num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    num_total_sol += local_buffer[i].size;
                }
                if (num_total_sol > params.one_epoch_ratio * num_vec){
                    rel_collection_stop = true;
                }
                if (SHOW_3SIEVE_DETAILS){
                    //printf("[thread %ld] the %ld-th bucket-search done, %ld new solutions found, num_total_sol = %ld\n", thread, ccount, local_buffer[thread].size-old_num_sol, num_total_sol);
                }
            }
            printf("the %ld-th epooch, goal = %f, avg bucket size = %ld, %ld solutions found in %ld buckets, already in ratio = %f, already in3 ratio = %f, found = %ld, found3 = %ld\n", count, sqrt(goal_norm),avg_bucket_size/ccount, local_buffer[thread].size,ccount, (already_in+.0)/found, (already_in3+.0)/found3, found, found3);
        }

        //put to the main buffer
        long num_total_sol = 0;
        for (long i = 0; i < num_threads; i++){
            num_total_sol += local_buffer[i].size;
        }
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++){
            long begin_index = 0;
            for (long j = 0; j < thread; j++){
                begin_index += local_buffer[j].size;
            }
            short *dst = main_buffer.buffer + begin_index * coeff_size;
            short *src = local_buffer[thread].buffer;
            for (long i = 0; i < local_buffer[thread].size; i++){
                copy(dst + i * coeff_size, src + i * coeff_size, coeff_block);
            }
        }
        if (SHOW_3SIEVE_DETAILS){
            std::cerr << "sol collection done, "<<num_total_sol<<" solutions found in total, begin insertion...\n";
        }
        

        //insert to the pool
        #pragma omp parallel for
        for (long i = 0; i < num_total_sol; i++){
            long *cdst = cvec + cvec_size * (sorted_index-i-1);
            float *dst = (float *)cdst[5];
            short *src = main_buffer.buffer + i * coeff_size;
            if (!uid.erase_uid(*((uint64_t *)(&(dst[-4]))))){
                std::cerr << "something must be wrong with the UidHashTable, warning!\n";
            }
            copy((short *)(dst-int_bias), src, coeff_block);
            compute_vec(dst);
            if (dst[-1] > goal_norm*1.00005){
                std::cerr << "warning!";
            }
            if (!uid.check_uid(*((uint64_t *)(&(dst[-4]))))){
                std::cerr << "ssomething must be wrong with the UidHashTable, warning!\n";
            }
            

            *((uint64_t *)(&cdst[0])) = *((uint64_t *)(&dst[-16]));
            *((uint64_t *)(&cdst[1])) = *((uint64_t *)(&dst[-14]));
            *((uint64_t *)(&cdst[2])) = *((uint64_t *)(&dst[-12]));
            *((uint64_t *)(&cdst[3])) = *((uint64_t *)(&dst[-10]));
            *((float *)(&cdst[4])) = dst[-1];
        }
        if (SHOW_3SIEVE_DETAILS){
            std::cerr << "insertion done!\n";
        }
        
        
        sorted_index = sorted_index - num_total_sol;
        if (params.resort_ratio * num_vec > sorted_index){
            sort_cvec();
            if (SHOW_3SIEVE_DETAILS){
                std::cerr<<"pool resorted\n";
            }
        }
    }
}