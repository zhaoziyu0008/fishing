#ifndef BASIS_H
#define BASIS_H

#include <NTL/LLL.h>
#include "tool.h"

class Basis {
    public:
        Mat<RR> b_RR;
        Mat<RR> miu_RR;
        Vec<RR> B_RR;
        Basis(Mat<double>& L);
        Basis(Mat<ZZ>& L);
        ~Basis(){};
        void size_reduce(long index);
        void compute_basis_gso();
        void show_dist_vec();
};




#endif