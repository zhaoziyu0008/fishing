#include "UidHashTable.h"

UidHashTable::UidHashTable(){
    n = 0;
    insert_uid(0);
    uid_coeffs = NULL;
}
UidHashTable::~UidHashTable(){
    if (uid_coeffs != NULL){
        delete[] uid_coeffs;
        uid_coeffs = NULL;
    }
}
long UidHashTable::size(){
    long sum = 0;
    for (long i = 0; i < NUM_UID_LOCK; i++){
        sum += uid_table[i].a.size();
    }
    return sum;
}