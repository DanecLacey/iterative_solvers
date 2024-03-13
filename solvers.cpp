#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

void jacobi_iteration(
    CRSMtxData *crs_mat,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
){
    double diag_elem;
    double sum;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
        sum = 0;
        for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == crs_mat->col[nz_idx]){
                diag_elem = crs_mat->val[nz_idx];
            }
            else{
                sum += crs_mat->val[nz_idx] * (*x_old)[crs_mat->col[nz_idx]];
            }
        }
        (*x_new)[row_idx] = ((*b)[row_idx] - sum) / diag_elem;
    }
}

void jacobi_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *A_x_tmp,
    CRSMtxData *crs_mat,
    std::vector<double> *normed_residuals,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
)
{
    int ncols = crs_mat->n_cols;
    double residual_norm;

    if(flags->print_iters){
        iter_output(x_old, loop_params->iter_count);
        printf("\n");
    }

#ifdef DEBUG_MODE
    std::cout << "x_old vector:" << std::endl;
    for(int i = 0; i < crs_mat->n_cols; ++i){
        std::cout << (*x_old)[i] << std::endl;
    }
#endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    // Perform Jacobi iterations until error cond. satisfied
    // Using relation: x_new = -D^{-1}*(L + U)*x_old + D^{-1}*b
    // After Jacobi iteration loop, x_new ~ A^{-1}b
    // NOTE: Tasking candidate
    do{
        jacobi_iteration(crs_mat, b, x_old, x_new);
        
        if (loop_params->iter_count % loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            calc_residual(crs_mat, x_new, b, r, A_x_tmp);
            residual_norm = infty_vec_norm(r);
            (*normed_residuals)[loop_params->residual_count] = residual_norm;
            ++loop_params->residual_count;

            if(flags->print_iters){
                iter_output(x_new, loop_params->iter_count);
            }  
        }

#ifdef DEBUG_MODE
        std::cout << "[";
        for(int i = 0; i < x_new->size(); ++i){
            std::cout << (*x_new)[i] << ", ";
        }
        std::cout << "]" << std::endl;
  
        std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
        std::cout << "stopping_criteria: " << loop_params->stopping_criteria << std::endl; 
#endif  
    
        std::swap(*x_new, *x_old);

        ++loop_params->iter_count;

    } while(residual_norm > loop_params->stopping_criteria && loop_params->iter_count < loop_params->max_iters);

    flags->convergence_flag = (residual_norm <= loop_params->stopping_criteria) ? true : false;

    std::swap(*x_old, *x_star);

    // Record final residual with approximated solution vector x
    calc_residual(crs_mat, x_star, b, r, A_x_tmp);
    (*normed_residuals)[loop_params->residual_count] = infty_vec_norm(r);

    // End timer
    (*calc_time_elapsed) = end_time(&calc_time_start, &calc_time_end);
}

void gs_iteration(
    CRSMtxData *crs_mat,
    std::vector<double> *b,
    std::vector<double> *x
){
    double diag_elem;
    double sum;

    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
        sum = 0;
        for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == crs_mat->col[nz_idx]){
                diag_elem = crs_mat->val[nz_idx];
            }
            else{
                sum += crs_mat->val[nz_idx] * (*x)[crs_mat->col[nz_idx]];
            }
        }
        (*x)[row_idx] = ((*b)[row_idx] - sum) / diag_elem;
    }
}

void gs_solve(
    std::vector<double> *x,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *A_x_tmp,
    CRSMtxData *crs_mat,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){
    int ncols = crs_mat->n_cols;
    double residual_norm;

    if(flags->print_iters){
        iter_output(x, loop_params->iter_count);
        printf("\n");
    }

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < ncols; ++i){
        std::cout << (*x)[i] << std::endl;
    }
#endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    // Perform Jacobi iterations until error cond. satisfied
    // Using relation: x_new = -D^{-1}*(L + U)*x_old + D^{-1}*b
    // After Jacobi iteration loop, x_new ~ A^{-1}b
    // NOTE: Tasking candidate
    do{
        gs_iteration(crs_mat, b, x);
        
        if (loop_params->iter_count % loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            calc_residual(crs_mat, x, b, r, A_x_tmp);
            residual_norm = infty_vec_norm(r);
            (*residuals_vec)[loop_params->residual_count] = residual_norm;
            ++loop_params->residual_count;

            if(flags->print_iters){
                iter_output(x, loop_params->iter_count);
            }  
        }

#ifdef DEBUG_MODE
        std::cout << "[";
        for(int i = 0; i < x->size(); ++i){
            std::cout << (*x)[i] << ", ";
        }
        std::cout << "]" << std::endl;
  
        std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
        std::cout << "stopping_criteria: " << loop_params->stopping_criteria << std::endl; 
#endif  
    
        ++loop_params->iter_count;

    } while(residual_norm > loop_params->stopping_criteria && loop_params->iter_count < loop_params->max_iters);

    flags->convergence_flag = (residual_norm <= loop_params->stopping_criteria) ? true : false;

    std::swap(*x, *x_star);

    // Record final residual with approximated solution vector x
    calc_residual(crs_mat, x, b, r, A_x_tmp);
    (*residuals_vec)[loop_params->residual_count] = infty_vec_norm(r);

    // End timer
    (*calc_time_elapsed) = end_time(&calc_time_start, &calc_time_end);
}