#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <iostream>
#ifdef USE_USPMV
#include "../Ultimate-SpMV/code/interface.hpp"
#endif
#include "mmio.h"

template <typename VT>
struct CRSMtxData
{
    // CRSMtxData(int _n_rows, int _n_cols, int _nnz, int *_row_ptr, int *_col, double *_val) 
    //     : n_rows(_n_rows), n_cols(_n_cols), nnz(_nnz), row_ptr(_row_ptr), col(_col), val(_val) {};

    // CRSMtxData() 
    //     : n_rows(0), n_cols(0), nnz(0), row_ptr(nullptr), col(nullptr), val(nullptr) {};

    int n_rows{};
    int n_cols{};
    int nnz{};

    // TODO
    // Interface* ce;
    int *row_ptr, *col;
    VT *val;

    // TODO
    // int *rcmPerm, *rcmInvPerm;
    // bool readFile(char* filename);

    // void makeDiagFirst(double missingDiag_value=0.0, bool rewriteAllDiag_with_maxRowSum=false);
    // bool isSymmetric();
    // void doRCM();
    // void doRCMPermute();

    void permute(int* perm, int* invPerm, bool RACEalloc=false);

    void print(void)
    {
        std::cout << "n_rows = " << n_rows << std::endl;
        std::cout << "n_cols = " << n_cols << std::endl;
        std::cout << "nnz = " << nnz << std::endl;

        std::cout << "row_ptr = [";
        for(int i = 0; i < n_rows+1; ++i){
            std::cout << row_ptr[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "col = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << col[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "values = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << val[i] << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

template <typename VT>
struct COOMtxData
{
    int n_rows{};
    int n_cols{};
    int nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<int> I;
    std::vector<int> J;
    std::vector<VT> values;

    // bool operator==(COOMtxData &rhs);

    // void operator^(COOMtxData &rhs);

    // void print(void);

    // void convert_to_crs(CRSMtxData *rhs);

    void write_to_mtx_file(int my_rank, std::string file_out_name)
    {
        std::string file_name = file_out_name + "_rank_" + std::to_string(my_rank) + ".mtx"; 

        int elem_num = 0;
        for(int nz_idx = 0; nz_idx < nnz; ++nz_idx){
            ++I[nz_idx];
            ++J[nz_idx];
        }

        mm_write_mtx_crd(
            &file_name[0], 
            n_rows, 
            n_cols, 
            nnz, 
            &(I)[0], 
            &(J)[0], 
            &(values)[0], 
            "MCRG" // TODO: <- make more general, i.e. flexible based on the matrix. Read from original mtx?
        );
    }

bool operator==(COOMtxData &rhs)
    {
    return (
        (n_rows == rhs.n_rows) &&
        (n_cols == rhs.n_cols) &&
        (nnz == rhs.nnz) &&
        (is_sorted == rhs.is_sorted) &&
        (is_symmetric == rhs.is_symmetric) &&
        (I == rhs.I) &&
        (J == rhs.J) &&
        (values == rhs.values)
    );
    }
void operator^(COOMtxData &rhs)
        {
        if (n_rows != rhs.n_rows){
            std::cout << "n_rows != rhs.n_rows" << std::endl;
        }
        if (n_cols != rhs.n_cols){
            std::cout << "n_cols != rhs.n_cols" << std::endl;
        }
        if (nnz != rhs.nnz){
            std::cout << "nnz != rhs.nnz" << std::endl;
        }
        if (is_sorted != rhs.is_sorted){
            std::cout << "is_sorted == rhs.is_sorted" << std::endl;
        }
        if (is_symmetric != rhs.is_symmetric){
            std::cout << "is_symmetric != rhs.is_symmetric" << std::endl;
        }
        if(I != rhs.I){
            std::cout << "I != rhs.I" << std::endl;
        }
        if(I.size() != rhs.I.size()){
            std::cout << "I.size() " << I.size() << " != rhs.I.size() " << rhs.I.size() << std::endl;
        }
        if(J != rhs.J){
            std::cout << "J != rhs.J" << std::endl;
        }
        if(J.size() != rhs.J.size()){
            std::cout << "J.size() " << J.size() << " != rhs.J.size() " << rhs.I.size() << std::endl;
        }
        if(values != rhs.values){
            std::cout << "values != rhs.values" << std::endl;
        }
        if(values.size() != rhs.values.size()){
            std::cout << "values.size() " << values.size() << " != rhs.values.size() " << rhs.values.size() << std::endl;
        }
    }

void print(void)
{
        std::cout << "n_rows = " << n_rows << std::endl;
        std::cout << "n_cols = " << n_cols << std::endl;
        std::cout << "nnz = " << nnz << std::endl;
        std::cout << "is_sorted = " << is_sorted << std::endl;
        std::cout << "is_symmetric = " << is_symmetric << std::endl;

        std::cout << "I = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << I[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "J = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << J[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "values = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << values[i] << ", ";
        }
        std::cout << "]" << std::endl;
}



    // COOMtxData operator+(COOMtxData &rhs)
    // {
    //     COOMtxData sum_coo_mtx;
    //     // Dimension stays constant
    //     sum_coo_mtx.n_rows = n_rows;
    //     sum_coo_mtx.n_cols = n_cols;

    //     // add nnz
    //     sum_coo_mtx.nnz = nnz + rhs.nnz;

    //     // NOTE: not sorted anymore?
    //     sum_coo_mtx.is_sorted = false;
    //     sum_coo_mtx.is_symmetric = false;

    //     // NOTE: how to best arrange? Concatenate for now
    //     sum_coo_mtx.I.reserve( I.size() + rhs.I.size() ); // preallocate memory
    //     sum_coo_mtx.I.insert( sum_coo_mtx.I.end(), I.begin(), I.end() );
    //     sum_coo_mtx.I.insert( sum_coo_mtx.I.end(), rhs.I.begin(), rhs.I.end() );

    //     sum_coo_mtx.J.reserve( J.size() + rhs.J.size() ); // preallocate memory
    //     sum_coo_mtx.J.insert( sum_coo_mtx.J.end(), J.begin(), J.end() );
    //     sum_coo_mtx.J.insert( sum_coo_mtx.J.end(), rhs.J.begin(), rhs.J.end() );

    //     sum_coo_mtx.values.reserve( values.size() + rhs.values.size() ); // preallocate memory
    //     sum_coo_mtx.values.insert( sum_coo_mtx.values.end(), values.begin(), values.end() );
    //     sum_coo_mtx.values.insert( sum_coo_mtx.values.end(), rhs.values.begin(), rhs.values.end() );

    //     return sum_coo_mtx;
    // }
};


// TODO
//necessary for GS like kernels
// void CRSMtxData::makeDiagFirst(double missingDiag_value, bool rewriteAllDiag_with_maxRowSum)
// {
//     double maxRowSum=0.0;
//     if(!diagFirst || rewriteAllDiag_with_maxRowSum)
//     {
//         //check whether a new allocation is necessary
//         int extra_nnz=0;
//         std::vector<double>* val_with_diag = new std::vector<double>();
//         std::vector<int>* col_with_diag = new std::vector<int>();
//         std::vector<int>* rowPtr_with_diag = new std::vector<int>(rowPtr, rowPtr+nrows+1);

//         for(int row=0; row<nrows; ++row)
//         {
//             bool diagHit = false;
//             double rowSum=0;
//             for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
//             {
//                 val_with_diag->push_back(val[idx]);
//                 col_with_diag->push_back(col[idx]);
//                 rowSum += val[idx];

//                 if(col[idx] == row)
//                 {
//                     diagHit = true;
//                 }
//             }
//             if(!diagHit)
//             {
//                 val_with_diag->push_back(missingDiag_value);
//                 col_with_diag->push_back(row);
//                 ++extra_nnz;
//                 rowSum += missingDiag_value;
//             }
//             maxRowSum = std::max(maxRowSum, std::abs(rowSum));
//             rowPtr_with_diag->at(row+1) = rowPtr_with_diag->at(row+1) + extra_nnz;
//         }

//         //allocate new matrix if necessary
//         if(extra_nnz)
//         {
//             delete[] val;
//             delete[] col;
//             delete[] rowPtr;

//             nnz += extra_nnz;
//             val = new double[nnz];
//             col = new int[nnz];
//             rowPtr = new int[nrows+1];

//             rowPtr[0] = rowPtr_with_diag->at(0);
// #pragma omp parallel for schedule(static)
//             for(int row=0; row<nrows; ++row)
//             {
//                 rowPtr[row+1] = rowPtr_with_diag->at(row+1);
//                 for(int idx=rowPtr_with_diag->at(row); idx<rowPtr_with_diag->at(row+1); ++idx)
//                 {
//                     val[idx] = val_with_diag->at(idx);
//                     col[idx] = col_with_diag->at(idx);
//                 }
//             }
//             printf("Explicit 0 in diagonal entries added\n");
//         }

//         delete val_with_diag;
//         delete col_with_diag;
//         delete rowPtr_with_diag;

// #pragma omp parallel for schedule(static)
//         for(int row=0; row<nrows; ++row)
//         {
//             bool diag_hit = false;

//             double* newVal = new double[rowPtr[row+1]-rowPtr[row]];
//             int* newCol = new int[rowPtr[row+1]-rowPtr[row]];
//             for(int idx=rowPtr[row], locIdx=0; idx<rowPtr[row+1]; ++idx, ++locIdx)
//             {
//                 //shift all elements+1 until diag entry
//                 if(col[idx] == row)
//                 {
//                     if(rewriteAllDiag_with_maxRowSum)
//                     {
//                         newVal[0] = maxRowSum;
//                     }
//                     else
//                     {
//                         newVal[0] = val[idx];
//                     }
//                     newCol[0] = col[idx];
//                     diag_hit = true;
//                 }
//                 else if(!diag_hit)
//                 {
//                     newVal[locIdx+1] = val[idx];
//                     newCol[locIdx+1] = col[idx];
//                 }
//                 else
//                 {
//                     newVal[locIdx] = val[idx];
//                     newCol[locIdx] = col[idx];
//                 }
//             }
//             //assign new Val
//             for(int idx = rowPtr[row], locIdx=0; idx<rowPtr[row+1]; ++idx, ++locIdx)
//             {
//                 val[idx] = newVal[locIdx];
//                 col[idx] = newCol[locIdx];
//             }

//             delete[] newVal;
//             delete[] newCol;
//         }
//         diagFirst = true;
//     }
// }

// #ifdef USE_USPMV
// typedef struct {
//     ScsData    *scs_mat,
//     ScsData    *scs_L,
//     ScsData    *scs_U
// } scsArgType;
// #endif

// typedef struct {
//     CRSMtxData *crs_mat,
//     CRSMtxData    *crs_L,
//     CRSMtxData    *crs_U
// } crsArgType;

template <typename VT>
struct SparseMtxFormat{
#ifdef USE_USPMV
    ScsData<VT, int> *scs_mat;
#ifdef USE_AP
    ScsData<double, int> *scs_mat_hp;
    ScsData<float, int> *scs_mat_lp;
#endif
    ScsData<VT, int> *scs_L;
    ScsData<VT, int> *scs_U;
#endif
    CRSMtxData<VT> *crs_mat;
    CRSMtxData<VT> *crs_L;
    CRSMtxData<VT> *crs_U;
};

#endif