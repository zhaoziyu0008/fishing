#include "basis.h"


Basis::Basis(Mat<double>& L){
    long n = L.NumRows();
    long m = L.NumCols();
    b_RR.SetDims(n, m);
    miu_RR.SetDims(n, n);
    B_RR.SetLength(n);
    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            b_RR[i][j] = to_RR(L[i][j]);
        }
    }
}
Basis::Basis(Mat<ZZ>& L){
    long n = L.NumRows();
    long m = L.NumCols();
    b_RR.SetDims(n, m);
    miu_RR.SetDims(n, n);
    B_RR.SetLength(n);
    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            b_RR[i][j] = conv<RR>(L[i][j]);
        }
    }
}
void Basis::show_dist_vec(){
    Vec<double> dist_vec;
    dist_vec.SetLength(B_RR.length());
    for (long i = 0; i < dist_vec.length(); i++){
        dist_vec[i] = conv<double>(sqrt(B_RR[i]));
    }
    std::cout << dist_vec << std::endl;
}
void Basis::compute_basis_gso(){
    long n = b_RR.NumRows();
    long m = b_RR.NumCols();
    Mat<RR> b_star_RR = b_RR;
    B_RR[0] = dot(b_star_RR[0],b_star_RR[0]);
	for (long i = 0; i < n-1; i++){
		for (long j = i+1; j < n; j++){
			miu_RR[j][i] = dot(b_star_RR[j],b_star_RR[i])/B_RR[i];
            red(b_star_RR[j],b_star_RR[i],miu_RR[j][i]);
		}
		B_RR[i+1] = dot(b_star_RR[i+1],b_star_RR[i+1]);
	}
    for (long i = 0; i < n; i++){
        miu_RR[i][i] = 1.0;
    }
}
void Basis::size_reduce(long index){
    long n = b_RR.NumRows();
    long m = b_RR.NumCols();
    Mat<RR> b_star_RR = b_RR;
    B_RR[0] = dot(b_star_RR[0],b_star_RR[0]);
	for (long i = 0; i < n-1; i++){
		for (long j = i+1; j < n; j++){
			miu_RR[j][i] = dot(b_star_RR[j],b_star_RR[i])/B_RR[i];
            red(b_star_RR[j],b_star_RR[i],miu_RR[j][i]);
		}
		B_RR[i+1] = dot(b_star_RR[i+1],b_star_RR[i+1]);
	}
    for (long i = 0; i < n; i++){
        miu_RR[i][i] = 1.0;
    }
    for (long i = index-1; i >= 0; i--){
        RR q;
        q = floor(miu_RR[index][i]+0.5);
        for (long j = i; j >=0; j--){
            miu_RR[index][j] -= q * miu_RR[i][j];
        }
        red(b_RR[index], b_RR[i], q);
    }

}