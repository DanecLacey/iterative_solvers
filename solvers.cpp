#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"


void jacobi_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new // treat like y
){
    double diag_elem = 0;
    double sum = 0;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < sparse_mat->crs_mat->n_rows; ++row_idx){
        sum = 0;
        for(int nz_idx = sparse_mat->crs_mat->row_ptr[row_idx]; nz_idx < sparse_mat->crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == sparse_mat->crs_mat->col[nz_idx]){
                diag_elem = sparse_mat->crs_mat->val[nz_idx];
            }
            else{
                sum += sparse_mat->crs_mat->val[nz_idx] * x_old[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        x_new[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}


/*
    I would think this would allow the easiest library integration, since the SpMV kernel is the same.
    Except here, you would need some way to avoid opening and closing the two parallel regions.
*/
void jacobi_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new,
    int N
){
    int n_rows = N; // <- make more flexible
    #pragma omp parallel
    {
#ifdef USE_USPMV
        uspmv_omp_scs_cpu<double, int>(
            sparse_mat->scs_mat->C,
            sparse_mat->scs_mat->n_chunks,
            &(sparse_mat->scs_mat->chunk_ptrs)[0],
            &(sparse_mat->scs_mat->chunk_lengths)[0],
            &(sparse_mat->scs_mat->col_idxs)[0],
            &(sparse_mat->scs_mat->values)[0],
            x_old,
            x_new);

        n_rows = sparse_mat->scs_mat->n_rows;
        // TODO: not sure which is correct
        // n_rows = sparse_mat->scs_mat->n_rows_padded;
#else
        spmv_crs_cpu(x_new, sparse_mat->crs_mat, x_old);
#endif
        // account for diagonal element in sum, RHS, and division 
        jacobi_normalize_x_cpu(x_new, x_old, D, b, N);
    }
}

void gs_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x
){
    double diag_elem;
    double sum;

    for(int row_idx = 0; row_idx < sparse_mat->crs_mat->n_rows; ++row_idx){
        sum = 0;
        for(int nz_idx = sparse_mat->crs_mat->row_ptr[row_idx]; nz_idx < sparse_mat->crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == sparse_mat->crs_mat->col[nz_idx]){
                diag_elem = sparse_mat->crs_mat->val[nz_idx];
            }
            else{
                sum += sparse_mat->crs_mat->val[nz_idx] * x[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        x[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}


void gs_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x,
    int N
){
    #pragma omp parallel
    {
        // spmv on strictly upper triangular portion of A to compute tmp <- Ux_{k-1}
// #ifdef USE_USPMV
//          TODO: Bug with upper triangular SpMV in USpMV library
//         spmv_omp_scs<double, int>(
//             sparse_mat->scs_U->C,
//             sparse_mat->scs_U->n_chunks,
//             &(sparse_mat->scs_U->chunk_ptrs)[0],
//             &(sparse_mat->scs_U->chunk_lengths)[0],
//             &(sparse_mat->scs_U->col_idxs)[0],
//             &(sparse_mat->scs_U->values)[0],
//             &(*x)[0],
//             &(*x)[0]);
// #else
        // trspmv_crs(tmp, crs_U, x); // <- TODO: Could you benefit from a triangular spmv?
        spmv_crs_cpu(tmp, sparse_mat->crs_U, x);
// #endif

#ifdef DEBUG_MODE_FINE
        printf("Ux = [");
        for(int i = 0; i < N; ++i){
            std::cout << tmp[i] << ",";
        }
        printf("]\n");
#endif

        // subtract b to compute tmp <- b-Ux_{k-1}
        subtract_vectors_cpu(tmp, b, tmp, N);
#ifdef DEBUG_MODE_FINE
        printf("b-Ux = [");
        for(int i = 0; i < N; ++i){
            std::cout << tmp[i] << ",";
        }
        printf("]\n");
#endif

        // performs the lower triangular solve (L+D)x_k=b-Ux_{k-1}
        #pragma omp master
        {
            spltsv_crs(sparse_mat->crs_L, x, D, tmp);
        }
#ifdef DEBUG_MODE_FINE
        printf("(D+L)^{-1}(b-Ux) = [");
        for(int i = 0; i < N; ++i){
            std::cout << x[i] << ",";
        }
        printf("]\n");
#endif
    
    }
}

void solve_cpu(
    argType *args
){
    // Unpack relevant args
    double *x = args->x_old; // GS
    double *x_new = args->x_new; // Jacobi
    double *x_old = args->x_old; // Jacobi
    double *tmp = args->tmp;
    double *D = args->D;
    double *r = args->r;
    double *b = args->b;
    SparseMtxFormat *sparse_mat = args->sparse_mat;
    Flags *flags = args->flags;

    double residual_norm;

    if(args->flags->print_iters){
        iter_output(x, args->vec_size, args->loop_params->iter_count);
        printf("\n");
    }

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << x[i] << std::endl;
    }
#endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    do{
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            // jacobi_iteration_ref_cpu(sparse_mat, D, b, x_old, x_new);
            jacobi_iteration_sep_cpu(sparse_mat, D, b, x_old, x_new, args->vec_size);
        }
        else if(args->solver_type == "gauss-seidel"){
            // For a reference solution, not meant for use with USpMV library
            // gs_iteration_ref_cpu(sparse_mat, tmp, D, b, x);
            gs_iteration_sep_cpu(sparse_mat, tmp, D, b, x, args->vec_size);
        }
        
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            if(args->solver_type == "jacobi"){
                
                calc_residual_cpu(sparse_mat, x_new, b, r, tmp, args->vec_size);
            }
            else if(args->solver_type == "gauss-seidel"){
                calc_residual_cpu(sparse_mat, x, b, r, tmp, args->vec_size);
            }
            
            residual_norm = infty_vec_norm_cpu(r, args->vec_size);
            args->normed_residuals[args->loop_params->residual_count] = residual_norm;
            ++args->loop_params->residual_count;

            if(flags->print_iters){
                if(args->solver_type == "jacobi"){
                    iter_output(x_new, args->vec_size, args->loop_params->iter_count);
                }
                else if(args->solver_type == "gauss-seidel"){
                    iter_output(x, args->vec_size, args->loop_params->iter_count);
                }
            }
        }

#ifdef DEBUG_MODE
        std::cout << "[";
        if(args->solver_type == "jacobi"){
            for(int i = 0; i < args->vec_size; ++i){
                std::cout << x_new[i] << ", ";
            }
        }
        else if (args->solver_type == "gauss-seidel"){
            for(int i = 0; i < args->vec_size; ++i){
                std::cout << x[i] << ", ";
            }
        }
        std::cout << "]" << std::endl;
  
        std::cout << "residual norm: " << infty_vec_norm_cpu(r, args->vec_size) << std::endl;
        std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
#endif  

        if(args->solver_type == "jacobi"){
            std::swap(x_new, x_old);
        }
    
        ++args->loop_params->iter_count;

    } while(residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    if(args->solver_type == "jacobi"){
        std::swap(x_old, args->x_star);
    }
    else if (args->solver_type == "gauss-seidel"){
        std::swap(*x, *(args->x_star));
    }    
    
    // Record final residual with approximated solution vector x
    calc_residual_cpu(sparse_mat, args->x_star, b, r, tmp, args->vec_size);
    args->normed_residuals[args->loop_params->residual_count] = infty_vec_norm_cpu(r, args->vec_size);

#ifdef USE_USPMV
    // Bring final result vector out of permuted space
    double *x_star_perm = new double[args->vec_size];
    apply_permutation(x_star_perm, args->x_star, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);

    // Deep copy, so you can free memory
    // TODO: wrap in func
    for(int i = 0; i < args->vec_size; ++i){
        args->x_star[i] = x_star_perm[i];
    }

    delete x_star_perm;
#endif

    // End timer
    args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);
}


#ifdef __CUDACC__
__global__
void jacobi_iteration_ref_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new,
    int n_rows
){
    double diag_elem = 0;
    double sum = 0;

    const unsigned int thread_idx_in_block = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * blockDim.x;
    const unsigned int thread_idx = block_offset + thread_idx_in_block;
    const unsigned int stride = gridDim.x * blockDim.x; // <- equiv. to total num threads
    unsigned int offset = 0;
    unsigned int row_idx;

    while (thread_idx + offset < n_rows){
        row_idx = thread_idx + offset;
        sum = 0;
        for(int nz_idx = d_row_ptr[row_idx]; nz_idx < d_row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == d_col[nz_idx]){
                diag_elem = d_val[nz_idx];
            }
            else{
                sum += d_val[nz_idx] * d_x_old[d_col[nz_idx]];
#ifdef DEBUG_MODE_FINE
                printf("%f * %f = %f at index %i\n", d_val[nz_idx], d_x_old[d_col[nz_idx]], d_val[nz_idx] * d_x_old[d_col[nz_idx]], row_idx); 
#endif
            }
        }

        d_x_new[row_idx] = (d_b[row_idx] - sum) / diag_elem;

        offset += stride;        
    }
}


/*
    I would think this would allow the easiest library integration, since the SpMV kernel is the same.
    Except here, you would need some way to avoid opening and closing the two parallel regions.
*/
void jacobi_iteration_sep_gpu(
    int d_n_rows,
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new
){
    int n_rows = d_n_rows;
#ifdef USE_USPMV
    if(CHUNK_SIZE > 1 || SIGMA > 1){
        printf("ERROR: SCS not yet supported in jacobi_iteration_sep_gpu.\n");
    }
    uspmv_csr_gpu<double, int><<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        d_n_rows,
        d_row_ptr,
        d_col,
        d_val,
        d_x_old,
        d_x_new);
#else
    spmv_crs_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_n_rows, d_row_ptr, d_col, d_val, d_x_old, d_x_new);
#endif
    // account for diagonal element in sum, RHS, and division 
    jacobi_normalize_x_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_x_new, d_x_old, d_D, d_b, d_n_rows);
}

void solve_gpu(
    argType *args
){

    // NOTE: Only for convenience. Will change to UM later.
    double *h_residual_norm = new double;

    // TODO: Why does this get messed up? 
    args->loop_params->residual_count = 0;

    // // Unpack relevant args
    // double *d_x = args->d_x_old; // GS
    // double *d_x_new = args->d_x_new; // Jacobi
    // double *d_x_old = args->d_x_old; // Jacobi
    // int d_n_rows = args->coo_mat->n_rows;

    // // TODO: collect into a struct
    // int *d_row_ptr = args->d_row_ptr;
    // int *d_col = args->d_col;
    // double *d_val = args->d_val; 

    // double *d_tmp = args->d_tmp;
    // double *d_D = args->d_D;
    // double *d_r = args->d_r;
    // double *d_b = args->d_b;

    double *d_residual_norm;
    cudaMalloc(&d_residual_norm, sizeof(double));
    cudaMemset(d_residual_norm, 0.0, sizeof(double));

    Flags *flags = args->flags; 

    double residual_norm;

    // TODO: Adapt for GPUs
    // if(args->flags->print_iters){
    //     iter_output(d_x, args->loop_params->iter_count);
    //     printf("\n");
    // }

    // TODO: Adapt for GPUs
// #ifdef DEBUG_MODE
//     std::cout << "x vector:" << std::endl;
//     for(int i = 0; i < args->vec_size; ++i){
//         std::cout << d_x[i] << std::endl;
//     }
// #endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    do{
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            // jacobi_iteration_ref_gpu<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new, args->vec_size);
            jacobi_iteration_sep_gpu(args->vec_size, args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new);
        }
        else if(args->solver_type == "gauss-seidel"){
            // TODO: Adapt for GPUs
            printf("GS_solve still under development for GPUs.\n");
            exit(1);
            // For a reference solution, not meant for use with USpMV library
            // gs_iteration_ref_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
            // gs_iteration_sep_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
        }
        
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            if(args->solver_type == "jacobi"){
                calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_new, args->d_b, args->d_r, args->d_tmp, args->vec_size);
            }
            else if(args->solver_type == "gauss-seidel"){
                // TODO: Adapt for GPUs
                printf("GS_solve still under development for GPUs.\n");
                exit(1);
                // calc_residual_gpu(sparse_mat, x, b, r, tmp);
            }
            
///////////////////////////////////// Grrr DEBUG! //////////////////////////////////////////
            // For now, have to do this on the CPU. Giving up on GPU implementation
            // cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
            // *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);
            // TODO: Correct grid + block size?
            // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
            // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
            infty_vec_norm_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(double)>>>(args->d_r, d_residual_norm, args->vec_size);
///////////////////////////////////// DEBUG! //////////////////////////////////////////

            // TODO: Put residual_norm in unified memory to avoid this transfer
            // NOTE: need to convert *double to *void
            cudaMemcpy(&(*h_residual_norm), &(*d_residual_norm), sizeof(double), cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize(); // <- not necessary
            // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

            // std::cout << "the first h_residual_norm = " << *h_residual_norm << std::endl;
            // exit(0);
            
            args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;
            ++args->loop_params->residual_count;

// TODO: Adapt for GPUs
//             if(flags->print_iters){
//                 if(args->solver_type == "jacobi"){
//                     iter_output(x_new, args->loop_params->iter_count);
//                 }
//                 else if(args->solver_type == "gauss-seidel"){
//                     iter_output(x, args->loop_params->iter_count);
//                 }
//             }
        }

// TODO: Adapt for GPUs
// #ifdef DEBUG_MODE
//         std::cout << "[";
//         if(args->solver_type == "jacobi"){
//             for(int i = 0; i < x_new->size(); ++i){
//                 std::cout << (*x_new)[i] << ", ";
//             }
//         }
//         else if (args->solver_type == "gauss-seidel"){
//             for(int i = 0; i < x->size(); ++i){
//                 std::cout << (*x)[i] << ", ";
//             }
//         }
//         std::cout << "]" << std::endl;
  
//         std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
//         std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
// #endif  

// TODO: Adapt for GPUs???
        cudaDeviceSynchronize();
        if(args->solver_type == "jacobi"){
            // NOTE: Might work, might not..
            // TODO: huh?? Causes seg fault
            // std::cout << "d_x_new pointer: " << d_x_new << std::endl;
            // std::cout << "d_x_old pointer: " << d_x_old << std::endl;
            std::swap(args->d_x_new, args->d_x_old);
            // std::cout << "d_x_new pointer after swap: " << d_x_new << std::endl;
            // std::cout << "d_x_old pointer after swap: " << d_x_old << std::endl;
        }
    
        ++args->loop_params->iter_count;

    // TODO: Put residual_norm in unified memory to avoid this transfer
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

    // Do check on host for now, easiest
        // std::cout << *h_residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
        // exit(0);
    } while(*h_residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (*h_residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    cudaDeviceSynchronize();
    if(args->solver_type == "jacobi"){
        // TODO: huh?? Causes seg fault
        std::swap(args->d_x_old, args->d_x_star);
    }
    else if (args->solver_type == "gauss-seidel"){
        // TODO: Adapt for GPUs
        printf("GS_solve still under development for GPUs.\n");
        exit(1);
        // std::swap(*x, *(args->x_star));
    }

    // Record final residual with approximated solution vector x
///////////////////////////////////// DEBUG! //////////////////////////////////////////
    // TODO: Giving up on GPU for this for now
    cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
    *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);

    // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
    // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
///////////////////////////////////////////////////////////////////////////////////////

    // TODO: Put residual_norm in unified memory to avoid this transfer
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

    args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;

// TODO: Adapt for GPUs
// #ifdef USE_USPMV
//     // Bring final result vector out of permuted space
//     std::vector<double> x_star_perm(args->vec_size, 0);
//     apply_permutation(&(x_star_perm)[0], &(*args->x_star)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
//     std::swap(x_star_perm, (*args->x_star));
// #endif

    // End timer
    args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);

    // Why are you freeing this here?
    cudaFree(d_residual_norm);
    delete h_residual_norm;
}
#endif

void solve(
    argType *args
){
#ifndef __CUDACC__
    solve_cpu(args);
#else
    solve_gpu(args);
#endif
}