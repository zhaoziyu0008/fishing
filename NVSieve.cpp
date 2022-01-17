#include "pool.h"

#define SHOW_NVSIEVE_DETAILS false


void Pool::NVSieve(NVSieve_params params){
    long coeff_size = (int_bias-16)*2;
    long coeff_block = coeff_size/32;
    if (true){
        std::cerr << "begin NVSieve on context ["<< index_l<<", "<<index_r << "], gh = "<< sqrt(gh2);
        std::cerr << ", pool size = "<< num_vec<<", "<<num_threads<<" threads will be used\n";
    }
    sort_cvec();

    long count = -1;
    while (!sieve_is_over(params.saturation_radius, params.saturation_ratio)){
        count++;
        long goal_index = (long)(params.improve_ratio*num_vec);
        float goal_norm = ((float *)(cvec+goal_index*cvec_size+4))[0];


        //prepare the buffer
        coeff_buffer local_buffer[num_threads];
        coeff_buffer main_buffer(coeff_size, num_vec);
        for (long i = 0; i < num_threads; i++){
            local_buffer[i].buffer_setup(coeff_size, num_vec/num_threads);
        }


        //collect solutions
        bool rel_collection_stop = false;
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++){
            long ccount = 0;
            long avg_bucket_size = 0;
            long already_in = 0;
            long found = 0;
            while(!rel_collection_stop){
                ccount++;
                //centering
                long index = rand()%(num_vec);
                long *cptr = cvec+index*cvec_size;
                float *ptr = (float *)cptr[5];
                long **bucket = new long*[num_vec];
                long num_element = 0;

                //bucketing
                long *cpptr = cvec;
                float ap2 = params.alpha*params.alpha*ptr[-1];
                for (long i = 0; i < num_vec; i++){
                    long w = __builtin_popcountl((*((uint64_t *)(&cptr[0]))) ^ (*((uint64_t *)(&cpptr[0]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[1]))) ^ (*((uint64_t *)(&cpptr[1]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[2]))) ^ (*((uint64_t *)(&cpptr[2]))));
                    w += __builtin_popcountl((*((uint64_t *)(&cptr[3]))) ^ (*((uint64_t *)(&cpptr[3]))));
                    if (w < XPC_BUCKET_THRESHOLD || w > (256 - XPC_BUCKET_THRESHOLD)){
                        float *pptr = (float *)cpptr[5];
                        float x = dot(pptr, ptr, vec_length);
                        float y = ap2*pptr[-1];
                        if (x*x>y){
                            bucket[num_element] = cpptr;
                            num_element++;
                        }
                    }
                    cpptr += cvec_size;
                }
                avg_bucket_size += num_element;
                if (SHOW_NVSIEVE_DETAILS){
                    //printf("[thread %ld] the %ld-th bucketing done, centered at vector %ld, bucket size = %ld\n", thread, ccount, index, num_element);
                }
                long old_num_sol = local_buffer[thread].size;
                //search the reductions
                for (long i = 0; i < num_element; i++){
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
                delete[] bucket;

                long num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    num_total_sol += local_buffer[i].size;
                }
                if (num_total_sol > params.one_epoch_ratio * num_vec){
                    rel_collection_stop = true;
                }
                if (SHOW_NVSIEVE_DETAILS){
                    //printf("[thread %ld] the %ld-th bucket-search done, %ld new solutions found, num_total_sol = %ld\n", thread, ccount, local_buffer[thread].size-old_num_sol, num_total_sol);
                }
            }
            if (CSD > 40){
                printf("the %ld-th epooch, goal = %f, avg bucket size = %ld, %ld solutions found in %ld buckets, already in ratio = %f\n", count, sqrt(goal_norm),avg_bucket_size/ccount, local_buffer[thread].size,ccount, (already_in+.0)/found);
            }
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
            if (!uid.check_uid(*((uint64_t *)(&(dst[-4]))))){
                std::cerr << "ssomething must be wrong with the UidHashTable, warning!\n";
            }
            
            if (dst[-1]>goal_norm){
                if (dst[-1] > goal_norm*1.0002){
                    std::cerr << "warning! "<<sqrt(dst[-1])<< " ";
                }

            }

            *((uint64_t *)(&cdst[0])) = *((uint64_t *)(&dst[-16]));
            *((uint64_t *)(&cdst[1])) = *((uint64_t *)(&dst[-14]));
            *((uint64_t *)(&cdst[2])) = *((uint64_t *)(&dst[-12]));
            *((uint64_t *)(&cdst[3])) = *((uint64_t *)(&dst[-10]));
            *((float *)(&cdst[4])) = dst[-1];
        }
        
        
        sorted_index = sorted_index - num_total_sol;
        if (params.resort_ratio * num_vec > sorted_index){
            sort_cvec();
        }
    }
}