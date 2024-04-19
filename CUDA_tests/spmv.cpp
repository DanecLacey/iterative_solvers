#include "helpers/io_funcs.hpp"
#include "helpers/mmio.h"
#include "helpers/structs.hpp"
#include "helpers/utilities.hpp"
#include <string>
#ifndef __CUDACC__
    #include <omp.h>
#endif

void spmv_crs_cpu(
    const double *val,
    const int *col,
    const int *row_ptr,
    double *y,
    const double *x,
    const int n_rows
    )
{
    double tmp;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        tmp = 0.0;
        for(int nz_idx = row_ptr[row_idx]; nz_idx < row_ptr[row_idx+1]; ++nz_idx){
            tmp += val[nz_idx] * x[col[nz_idx]];
        }
        y[row_idx] = tmp;
    }
}

// TODO: how do
#ifdef __CUDACC__
__global__
void spmv_crs_gpu(
//     std::vector<double> *y,
//     const CRSMtxData *crs_mat,
//     const std::vector<double> *x
    )
{
//     double tmp;

//     // Orphaned directive: Assumed already called within a parallel region
//     #pragma omp for schedule (static)
//     for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
//         tmp = 0.0;
//         for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
//             tmp += crs_mat->val[nz_idx] * (*x)[crs_mat->col[nz_idx]];
//         }
//         (*y)[row_idx] = tmp;
//     }
}
#endif

int main(int argc, char *argv[]) {
    // Read .mtx file
    std::string matrix_file_name = argv[1];
    COOMtxData *coo_mat = new COOMtxData;
    CRSMtxData *crs_mat = new CRSMtxData;
    read_mtx(coo_mat, matrix_file_name);

    // Convert to CRS 
    convert_to_crs(coo_mat, crs_mat);

    // Initialize x and y
    double *x = new double[crs_mat->n_cols];
    double *y = new double[crs_mat->n_cols];
    generate_x_and_y(x, y, crs_mat->n_cols, crs_mat->nnz, false, crs_mat->val, 1.1);

#ifdef __CUDACC__
    // Move data to device

#endif

    // Do Spmv
#ifdef __CUDACC__
//     spmv_crs_gpu<<< , >>>();
#else
    spmv_crs_cpu(
        crs_mat->val,
        crs_mat->col,
        crs_mat->row_ptr,
        y,
        x,
        crs_mat->n_rows
    );
#endif

#ifdef __CUDACC__
    // Move data back to host

#endif

    // Collect error by comparing against MKL
    validate_dp_result(crs_mat, x, y);

    delete coo_mat;
    delete crs_mat;
    delete x;
    delete y;
}