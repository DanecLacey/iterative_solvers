#include "kernels.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>

void mtx_spmv_coo(
    std::vector<double> *y,
    const COOMtxData *mtx,
    const std::vector<double> *x
    )
{
    int nnz = mtx->nnz;
    if (mtx->I.size() != mtx->J.size() || mtx->I.size() != mtx->values.size()) {
        printf("ERROR: mtx_spmv_coo : sizes of rows, cols, and values differ.\n");
        exit(1);
    }

    // parallel for candidate
    for (int i = 0; i < nnz; ++i) {
        (*y)[mtx->I[i]] += mtx->values[i] * (*x)[mtx->J[i]];
        // std::cout << mtx->values[i] << " * " << (*x)[mtx->J[i]] << " = " << (*y)[mtx->I[i]] << " at idx: " << mtx->I[i] << std::endl; 
    }
}

void vec_spmv_coo(
    std::vector<double> *mult_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
    )
{
    int nrows = vec1->size();

    if(vec1->size() != vec2->size()){
        printf("ERROR: vec_spmv_coo: mismatch in vector sizes.\n");
        exit(1);
    }

    // parallel for candidate
    for (int i = 0; i < nrows; ++i) {
        (*mult_vec)[i] += (*vec1)[i] * (*vec2)[i];
        // std::cout << (*vec)[i] << " * " << (*x)[i]<< " = " << (*y)[i] << " at idx: " << i << std::endl; 
    }
}

void sum_vectors(
    std::vector<double> *result_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
){
#ifdef DEBUG_MODE
    if(vec1->size() != vec2->size()){
        printf("ERROR: sum_vectors: mismatch in vector sizes.\n");
        exit(1);
    }
    if(vec1->size() == 0){
        printf("ERROR: sum_vectors: zero size vectors.\n");
        exit(1);
    }
#endif

    #pragma omp parallel for
    for (int i=0; i<vec1->size(); i++){
        (*result_vec)[i] = (*vec1)[i] + (*vec2)[i];
    }
}

void subtract_vectors(
    std::vector<double> *result_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
){
#ifdef DEBUG_MODE
    if(vec1->size() != vec2->size()){
        printf("ERROR: sum_vectors: mismatch in vector sizes.\n");
        exit(1);
    }
    if(vec1->size() == 0){
        printf("ERROR: sum_vectors: zero size vectors.\n");
        exit(1);
    }
#endif
    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for
    for (int i=0; i<vec1->size(); i++){
        (*result_vec)[i] = (*vec1)[i] - (*vec2)[i];
    }
}

void sum_matrices(
    COOMtxData *sum_mtx,
    const COOMtxData *mtx1,
    const COOMtxData *mtx2
){
    // NOTE: Matrices are assumed are to be sorted by row index

    // Traverse through all nz_idx in mtx1 and mtx2, and insert the
    // element with smaller row (and col?) value into the sum_mtx

    // In the case that an element has the same row and column value, 
    // add their values, and inster into the sum_mtx

    // if matrices don't have same dimensions
    if (mtx1->n_rows != mtx2->n_rows || mtx1->n_cols != mtx2->n_cols)
    {
        printf("ERROR: sum_matrices: Matrix dimensions do not match, cannot add.\n");
        exit(1);
    }

    sum_mtx->n_rows = mtx1->n_rows;
    sum_mtx->n_cols = mtx2->n_cols;

    sum_mtx->is_sorted = true;
    sum_mtx->is_symmetric = false; // TODO: How to actually detect this?

    // Just allocate for worst case scenario, in terms of storage
    sum_mtx->I.reserve(mtx1->nnz + mtx2->nnz);
    sum_mtx->J.reserve(mtx1->nnz + mtx2->nnz);
    sum_mtx->values.reserve(mtx1->nnz + mtx2->nnz);

    int nz_idx1 = 0, nz_idx2 = 0; //, nz_idx_sum = 0;
    while (nz_idx1 < mtx1->nnz && nz_idx2 < mtx2->nnz){
        // if mtx2's row and col is smaller (or tiebreaker)
        if (mtx1->I[nz_idx1] > mtx2->I[nz_idx2] ||
            ( mtx1->I[nz_idx1] == mtx2->I[nz_idx2] && mtx1->J[nz_idx1] > mtx2->J[nz_idx2])){

                // insert this smaller value into sum_mtx
                // sum_mtx->I[nz_idx_sum] = mtx2->I[nz_idx2];
                // sum_mtx->J[nz_idx_sum] = mtx2->J[nz_idx2];
                // sum_mtx->values[nz_idx_sum] = mtx2->values[nz_idx2]; 
                sum_mtx->I.push_back(mtx2->I[nz_idx2]);
                sum_mtx->J.push_back(mtx2->J[nz_idx2]);
                sum_mtx->values.push_back(mtx2->values[nz_idx2]);        

                // inc the nz counter for mtx2 and sum_mtx
                ++nz_idx2;        
                // ++nz_idx_sum;
        }
        // else if mtx1's row and col is smaller (or tiebreaker)
        else if (mtx1->I[nz_idx1] < mtx2->I[nz_idx2] ||
        (mtx1->I[nz_idx1] == mtx2->I[nz_idx2] && mtx1->J[nz_idx1] < mtx2->J[nz_idx2])){
            
                // insert this smaller value into sum_mtx
                // sum_mtx->I[nz_idx_sum] = mtx2->I[nz_idx2];
                // sum_mtx->J[nz_idx_sum] = mtx2->J[nz_idx2];
                // sum_mtx->values[nz_idx_sum] = mtx2->values[nz_idx2];  
                sum_mtx->I.push_back(mtx1->I[nz_idx1]);
                sum_mtx->J.push_back(mtx1->J[nz_idx1]);
                sum_mtx->values.push_back(mtx1->values[nz_idx1]);   

                // inc the nz counter for mtx1 and sum_mtx
                ++nz_idx1;     
                // ++nz_idx_sum;
        }
        // else, we actually sum the elements from mtx1 and mtx2!
        else{
            double sum_val = mtx1->values[nz_idx1] + mtx2->values[nz_idx2];
            if(sum_val > 1e-16){ // TODO: Maybe adjust this, just to avoid very small numbers
                // sum_mtx->I[nz_idx_sum] = mtx1->I[nz_idx1]; // Could either take row and col from mtx1 or mtx2
                // sum_mtx->J[nz_idx_sum] = mtx1->J[nz_idx1];
                // sum_mtx->values[nz_idx_sum] = sum_val;     
                sum_mtx->I.push_back(mtx1->I[nz_idx1]); // Could either take row and col from mtx1 or mtx2
                sum_mtx->J.push_back(mtx1->J[nz_idx1]);
                sum_mtx->values.push_back(sum_val);    
            }

            // increment all counters
            ++nz_idx1;
            ++nz_idx2;     
            // ++nz_idx_sum;
        }
    }

        // clean up remaining elements
    while(nz_idx1 < mtx1->nnz){
            sum_mtx->I.push_back(mtx1->I[nz_idx1]);
            sum_mtx->J.push_back(mtx1->J[nz_idx1]);
            sum_mtx->values.push_back(mtx1->values[nz_idx1]);   

        // ++nz_idx_sum;
        ++nz_idx1;
    }

    while(nz_idx2 < mtx2->nnz){
            sum_mtx->I.push_back(mtx2->I[nz_idx2]);
            sum_mtx->J.push_back(mtx2->J[nz_idx2]);
            sum_mtx->values.push_back(mtx2->values[nz_idx2]);   

        // ++nz_idx_sum;
        ++nz_idx2;
    }

    // Sanity checks
    if(nz_idx1 != mtx1->nnz){
        printf("ERROR: sum_matrices: Not all elements from mtx1 accounted for.\n");
        exit(1);
    }
    if(nz_idx2 != mtx2->nnz){
        printf("ERROR: sum_matrices: Not all elements from mtx2 accounted for.\n");
        exit(1);
    }

    sum_mtx->nnz = sum_mtx->values.size();

    // if(sum_mtx->nnz > (nz_idx1 + nz_idx2)){
    //     printf("ERROR: sum_matrices: sum_mtx contains too many nonzeros.\n");
    // }

}

void spmv_crs(
    std::vector<double> *y,
    const CRSMtxData *crs_mat,
    const std::vector<double> *x
    )
{
    double tmp;

    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for schedule (static)
    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
        tmp = 0.0;
        for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            tmp += crs_mat->val[nz_idx] * (*x)[crs_mat->col[nz_idx]];
#ifdef DEBUG_MODE
            std::cout << crs_mat->val[nz_idx] << " * " << (*x)[crs_mat->col[nz_idx]] << " = " << crs_mat->val[nz_idx] * (*x)[crs_mat->col[nz_idx]] << " at idx: " << row_idx << std::endl; 
#endif
        }
        (*y)[row_idx] = tmp;
    }
}

void jacobi_normalize_x(
    std::vector<double> *x_new,
    const std::vector<double> *x_old,
    const std::vector<double> *D,
    const std::vector<double> *b,
    int n_rows
){
    double adjusted_x;

    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for schedule (static)
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        adjusted_x = (*x_new)[row_idx] - (*D)[row_idx] * (*x_old)[row_idx];
        (*x_new)[row_idx] = ((*b)[row_idx] - adjusted_x)/ (*D)[row_idx];
#ifdef DEBUG_MODE
            std::cout << (*b)[row_idx] << " - " << adjusted_x << " / " << (*D)[row_idx] << " = " << (*x_new)[row_idx] << " at idx: " << row_idx << std::endl; 
#endif
    }
}

void spltsv_crs(
    const CRSMtxData *crs_L,
    std::vector<double> *x,
    const std::vector<double> *D,
    const std::vector<double> *b_Ux
)
{
    double sum;

    for(int row_idx = 0; row_idx < crs_L->n_rows; ++row_idx){
        sum = 0.0;
        for(int nz_idx = crs_L->row_ptr[row_idx]; nz_idx < crs_L->row_ptr[row_idx+1]; ++nz_idx){
            sum += crs_L->val[nz_idx] * (*x)[crs_L->col[nz_idx]];
#ifdef DEBUG_MODE
            std::cout << crs_L->val[nz_idx] << " * " << (*x)[crs_L->col[nz_idx]] << " = " << crs_L->val[nz_idx] * (*x)[crs_L->col[nz_idx]] << " at idx: " << row_idx << std::endl; 
#endif
        }
        (*x)[row_idx] = ((*b_Ux)[row_idx] - sum)/(*D)[row_idx];
    }
}