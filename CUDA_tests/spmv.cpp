#include "helpers/io_funcs.hpp"
#include "helpers/mmio.h"
#include "helpers/structs.hpp"
#include "helpers/utilities.hpp"
#include <string>
#include <omp.h>

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
    const double *val,
    const int *col,
    const int *row_ptr,
    double *y,
    const double *x,
    const int n_rows
    )
{
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int i = block_offset + thread_idx_in_block;

    if(i < n_rows){
        double tmp = 0.0;

        for(int nz_idx = row_ptr[i]; nz_idx < row_ptr[i+1]; ++nz_idx){
            tmp += val[nz_idx] * x[col[nz_idx]];
        }

        y[i] = tmp;
    }
}
#endif

int main(int argc, char *argv[]) {
    // Read .mtx file
    if(argc != 2){
        printf("ERROR: Unexpected input.\n");
        exit(1);
    }
    std::string matrix_file_name = argv[1];
    COOMtxData *coo_mat = new COOMtxData;
    CRSMtxData *crs_mat = new CRSMtxData;
    printf("Reading matrix...\n");
    read_mtx(coo_mat, matrix_file_name);
    int vec_size = coo_mat->n_cols;

    // Convert to CRS 
    printf("Converting to CRS...\n");
    convert_to_crs(coo_mat, crs_mat);

    // Initialize x and y
    double *x = new double[vec_size];
    double *y = new double[vec_size];
    generate_x_and_y(x, y, vec_size, crs_mat->nnz, false, crs_mat->val, 1.1);

#ifdef __CUDACC__
    printf("Moving data to device...\n");
    // Move data to device
    double *d_x = new double[vec_size];
    double *d_y = new double[vec_size]; <- dont need to allocate all this space on the host?????
    double *d_val = new double[crs_mat->nnz];
    int *d_col = new int[crs_mat->nnz];
    int *d_row_ptr = new int[crs_mat->n_rows + 1];

    cudaMalloc(&d_x, vec_size*sizeof(double));
    cudaMalloc(&d_y, vec_size*sizeof(double));
    cudaMalloc(&d_val, crs_mat->nnz*sizeof(double));
    cudaMalloc(&d_col, crs_mat->nnz*sizeof(int));
    cudaMalloc(&d_row_ptr, (crs_mat->n_rows+1)*sizeof(int));

    cudaMemcpy(d_x, x, vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, crs_mat->val, crs_mat->nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, crs_mat->col, crs_mat->nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, crs_mat->row_ptr, (crs_mat->n_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
#endif
    printf("Performing SpMVs...\n");
    // Do Spmv (multiple times, to see easier on nsys report)
    for(int i = 0; i < 100; ++i){
#ifdef __CUDACC__
        int threads_per_block = 256;
        int num_blocks = crs_mat->n_rows / threads_per_block;

        // Allocate one extra block to catch remaining rows
        spmv_crs_gpu<<< num_blocks+1, threads_per_block >>>(
            d_val,
            d_col,
            d_row_ptr,
            d_y,
            d_x,
            vec_size
        );
#else
        spmv_crs_cpu(
            crs_mat->val,
            crs_mat->col,
            crs_mat->row_ptr,
            y,
            x,
            vec_size
        );
#endif
}
#ifdef __CUDACC__
    printf("Copying data back to host...\n");
    // Move result data back to host
    cudaMemcpy(y, d_y, vec_size*sizeof(double), cudaMemcpyDeviceToHost);
#endif

    printf("Validating results...\n");
    // Collect error by comparing against MKL
    validate_dp_result(crs_mat, x, y);

        cudaFree(d_x);
    cudaFree(d_y);
    Helloooo?????

    delete coo_mat;
    delete crs_mat;
    delete x;
    delete y;
}