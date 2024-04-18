#include "kernels.hpp"
#ifdef USE_USPMV
    #include "../Ultimate-SpMV/code/interface.hpp"
#endif
#include "utility_funcs.hpp"
#include "io_funcs.hpp"


void jacobi_iteration_ref(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new // treat like y
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
                sum += sparse_mat->crs_mat->val[nz_idx] * (*x_old)[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        (*x_new)[row_idx] = ((*b)[row_idx] - sum) / diag_elem;
    }
}


/*
    I would think this would allow the easiest library integration, since the SpMV kernel is the same.
    Except here, you would need some way to avoid opening and closing the two parallel regions.
*/
void jacobi_iteration_sep(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new // treat like y
){
    int n_rows;
    #pragma omp parallel
    {
#ifdef USE_USPMV
        spmv_omp_scs<double, int>(
            sparse_mat->scs_mat->C,
            sparse_mat->scs_mat->n_chunks,
            &(sparse_mat->scs_mat->chunk_ptrs)[0],
            &(sparse_mat->scs_mat->chunk_lengths)[0],
            &(sparse_mat->scs_mat->col_idxs)[0],
            &(sparse_mat->scs_mat->values)[0],
            &(*x_old)[0],
            &(*x_new)[0]);

        n_rows = sparse_mat->scs_mat->n_rows;
        // TODO: not sure which is correct
        // n_rows = sparse_mat->scs_mat->n_rows_padded;
#else
        spmv_crs(x_new, sparse_mat->crs_mat, x_old);
        n_rows = sparse_mat->crs_mat->n_rows;
#endif

        // account for diagonal element in sum, RHS, and division 
        jacobi_normalize_x(x_new, x_old, D, b, n_rows);
    }
}

void gs_iteration_ref(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x
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
                sum += sparse_mat->crs_mat->val[nz_idx] * (*x)[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        (*x)[row_idx] = ((*b)[row_idx] - sum) / diag_elem;
    }
}


void gs_iteration_sep(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x
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
        spmv_crs(tmp, sparse_mat->crs_U, x);
// #endif

#ifdef DEBUG_MODE_FINE
        printf("Ux = [");
        for(int i = 0; i < tmp->size(); ++i){
            std::cout << (*tmp)[i] << ",";
        }
        printf("]\n");
#endif

        // subtract b to compute tmp <- b-Ux_{k-1}
        subtract_vectors(tmp, b, tmp);
#ifdef DEBUG_MODE_FINE
        printf("b-Ux = [");
        for(int i = 0; i < tmp->size(); ++i){
            std::cout << (*tmp)[i] << ",";
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
        for(int i = 0; i < x->size(); ++i){
            std::cout << (*x)[i] << ",";
        }
        printf("]\n");
#endif
    
    }
}

void solve(
    argType *args
){
    // Unpack relevant args
    std::vector<double> *x = args->x_old; // GS
    std::vector<double> *x_new = args->x_new; // Jacobi
    std::vector<double> *x_old = args->x_old; // Jacobi
    std::vector<double> *tmp = args->tmp;
    std::vector<double> *D = args->D;
    std::vector<double> *r = args->r;
    std::vector<double> *b = args->b;
    SparseMtxFormat *sparse_mat = args->sparse_mat;
    Flags *flags = args->flags;

    double residual_norm;

    if(args->flags->print_iters){
        iter_output(x, args->loop_params->iter_count);
        printf("\n");
    }

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << (*x)[i] << std::endl;
    }
#endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    do{
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            // jacobi_iteration_ref(sparse_mat, D, b, x_old, x_new);
            jacobi_iteration_sep(sparse_mat, D, b, x_old, x_new);
        }
        else if(args->solver_type == "gauss-seidel"){
            // For a reference solution, not meant for use with USpMV library
            // gs_iteration_ref(sparse_mat, tmp, D, b, x);
            gs_iteration_sep(sparse_mat, tmp, D, b, x);
        }
        
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            if(args->solver_type == "jacobi"){
                calc_residual(sparse_mat, x_new, b, r, tmp);
            }
            else if(args->solver_type == "gauss-seidel"){
                calc_residual(sparse_mat, x, b, r, tmp);
            }
            
            residual_norm = infty_vec_norm(r);
            (*(args->normed_residuals))[args->loop_params->residual_count] = residual_norm;
            ++args->loop_params->residual_count;

            if(flags->print_iters){
                if(args->solver_type == "jacobi"){
                    iter_output(x_new, args->loop_params->iter_count);
                }
                else if(args->solver_type == "gauss-seidel"){
                    iter_output(x, args->loop_params->iter_count);
                }
            }
        }

#ifdef DEBUG_MODE
        std::cout << "[";
        if(args->solver_type == "jacobi"){
            for(int i = 0; i < x_new->size(); ++i){
                std::cout << (*x_new)[i] << ", ";
            }
        }
        else if (args->solver_type == "gauss-seidel"){
            for(int i = 0; i < x->size(); ++i){
                std::cout << (*x)[i] << ", ";
            }
        }
        std::cout << "]" << std::endl;
  
        std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
        std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
#endif  

        if(args->solver_type == "jacobi"){
            std::swap(*x_new, *x_old);
        }
    
        ++args->loop_params->iter_count;

    } while(residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    if(args->solver_type == "jacobi"){
        std::swap(*x_old, *(args->x_star));
    }
    else if (args->solver_type == "gauss-seidel"){
        std::swap(*x, *(args->x_star));
    }
    
    
    // Record final residual with approximated solution vector x
    calc_residual(sparse_mat, args->x_star, b, r, tmp);
    (*(args->normed_residuals))[args->loop_params->residual_count] = infty_vec_norm(r);

#ifdef USE_USPMV
    // Bring final result vector out of permuted space
    std::vector<double> x_star_perm(args->vec_size, 0);
    apply_permutation(&(x_star_perm)[0], &(*args->x_star)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
    std::swap(x_star_perm, (*args->x_star));
#endif

    // End timer
    args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);
}