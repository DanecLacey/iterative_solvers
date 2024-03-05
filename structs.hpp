#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>
#include <iostream>

// NOTE: Is not having default bools really bad?
struct Flags
{
    bool print_iters;
    bool print_summary;
    bool print_residuals;
    bool convergence_flag;
    bool export_errors;
};

// TODO: Make user-enetered option available
struct LoopParams
{
    int iter_count;
    int residual_count;
    int residual_check_len;
    int max_iters;
    double stopping_criteria;
    double tol;
};

struct CRSMtxData
{
    int n_rows{};
    int n_cols{};
    int nnz{};

    // TODO
    // Interface* ce;
    int *row_ptr, *col;
    double *val;

    // TODO
    // int *rcmPerm, *rcmInvPerm;
    // bool readFile(char* filename);

    // void makeDiagFirst(double missingDiag_value=0.0, bool rewriteAllDiag_with_maxRowSum=false);
    // bool isSymmetric();
    // void doRCM();
    // void doRCMPermute();

    void permute(int* perm, int* invPerm, bool RACEalloc=false);
};

// TODO: Make class with member funcitons
struct COOMtxData
{
    int n_rows{};
    int n_cols{};
    int nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<int> I;
    std::vector<int> J;
    std::vector<double> values;

    // bool operator==(COOMtxData &rhs);

    // void operator^(COOMtxData &rhs);

    // void print(void);

    // void convert_to_crs(CRSMtxData *rhs);

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

void convert_to_crs(CRSMtxData *rhs)
{
    rhs->n_rows = this->n_rows;
    rhs->n_cols = this->n_cols;
    rhs->nnz = this->nnz;

    rhs->row_ptr = new int[rhs->n_rows+1];
    int *nnzPerRow = new int[rhs->n_rows];

    rhs->col = new int[rhs->nnz];
    rhs->val = new double[rhs->nnz];

    for(int idx = 0; idx < rhs->nnz; ++idx)
    {
        rhs->col[idx] = this->J[idx];
        rhs->val[idx] = this->values[idx];
    }

    for(int i = 0; i < rhs->n_rows; ++i)
    { 
        nnzPerRow[i] = 0;
    }

    //count nnz per row
    for(int i=0; i < rhs->nnz; ++i)
    {
        ++nnzPerRow[this->I[i]];
    }

    rhs->row_ptr[0] = 0;
    for(int i=0; i < rhs->n_rows; ++i)
    {
        rhs->row_ptr[i+1] = rhs->row_ptr[i]+nnzPerRow[i];
    }

    if(rhs->row_ptr[rhs->n_rows] != rhs->nnz)
    {
        printf("ERROR: converting to CRS.\n");
        exit(1);
    }

    delete[] nnzPerRow;
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
#endif /*STRUCTS_H*/