#ifndef UIDHASHTABLE_H
#define UIDHASHTABLE_H

#include <pthread.h>
#include <array>
#include <unordered_set>
#include <type_traits>
#include <limits>
#include "tool.h"

using UidType = uint64_t;
struct padded_unordered_set { __attribute__ ((aligned(128))) std::unordered_set<UidType> a; };
struct padded_spinlock { __attribute__ ((aligned(128))) pthread_spinlock_t a = 1; };


class UidHashTable {
    public:
        uint64_t *uid_coeffs;
        UidHashTable();
        ~UidHashTable();
        inline bool insert_uid(UidType uid);
        inline bool erase_uid(UidType uid);
        inline bool check_uid(UidType uid);
        inline void reset_hash_function(long CSD);
        long size();

    private:
        inline void normalize_uid(UidType &uid);
        static constexpr unsigned NUM_UID_LOCK = 8191;
        std::array<padded_spinlock, NUM_UID_LOCK> uid_lock;
        std::array<padded_unordered_set, NUM_UID_LOCK> uid_table;
        long n;                         //current dimension   
};


inline void UidHashTable::normalize_uid(UidType &uid){
    if (uid > std::numeric_limits<UidType>::max()/2  + 1) uid = -uid;
}
inline void UidHashTable::reset_hash_function(long CSD){
    delete[] uid_coeffs;
    uid_coeffs = new uint64_t[CSD+16];
    for (long i = 0; i < CSD; i++){
        uid_coeffs[i] = rand_uint64_t();
    }
    #pragma omp parallel for
    for (long i = 0; i < NUM_UID_LOCK; i++){
        uid_table[i].a.clear();
    }
    insert_uid(0);
}
inline bool UidHashTable::insert_uid(UidType uid){
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = uid_table[uid % NUM_UID_LOCK].a.insert(uid).second;
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
inline bool UidHashTable::erase_uid(UidType uid){
    if (uid == 0) return false;
    normalize_uid(uid);
    pthread_spin_lock(&uid_lock[uid % NUM_UID_LOCK].a);
    bool success = (uid_table[uid % NUM_UID_LOCK].a.erase(uid) != 0);
    pthread_spin_unlock(&uid_lock[uid % NUM_UID_LOCK].a);
    return success;
}
inline bool UidHashTable::check_uid(UidType uid){
    normalize_uid(uid);
    if (uid_table[uid % NUM_UID_LOCK].a.count(uid) != 0) return true;
    return false;
}


#endif