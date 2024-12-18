#ifndef SOLVERS_H
#define SOLVERS_H
#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"
#include "solver.hpp"
#include "structs.hpp"
#include "methods/jacobi.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/gmres.hpp"

#include <cmath>

template <typename VT>
void solve_cpu(
    argType<VT> *args,
    Solver<VT> *solver
){
    std::cout << "Entering Solver Harness" << std::endl;

    VT residual_norm;

    if(args->flags->print_iters){
        iter_output(solver->x_old, args->vec_size, args->loop_params->iter_count);
        printf("\n");
    }

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << static_cast<double>(solver->x[i]) << std::endl;
    }
#endif

#ifdef USE_LIKWID
    register_likwid_markers();
#endif


    do{
        args->timers->solver_wtime->start_stopwatch();
        //////////////////// Main Iteration //////////////////////
        solver->iterate(
            args->sparse_mat,
            args->timers,
            args->vec_size,
            args->coo_mat->n_rows,
            args->loop_params->iter_count,
            &residual_norm
        );
        /////////////////////////////////////////////////////////
        args->timers->solver_wtime->end_stopwatch();


        // Record residual every "residual_check_len" iterations
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            record_residual_norm<VT>(
                args, 
                args->flags, 
                args->sparse_mat, 
                solver->r,
                solver->x, 
                solver->b, 
                solver->x_new, 
                solver->tmp,
                solver->tmp_perm,
#ifdef USE_AP
                solver->x_dp, 
                solver->x_new_dp, 
                solver->tmp_dp,
                solver->tmp_perm_dp,
                solver->x_sp, 
                solver->x_new_sp, 
                solver->tmp_sp,
                solver->tmp_perm_sp,
#ifdef HAVE_HALF_MATH
                solver->x_hp, 
                solver->x_new_hp, 
                solver->tmp_hp,
                solver->tmp_perm_hp,
#endif
#endif
                &residual_norm
            );
        }

        if(args->flags->print_iters)
            solver->print_x(args->vec_size, args->coo_mat->n_cols, args->loop_params->iter_count);  

        solver->exchange_arrays(args->vec_size);

        if(
            args->solver_type == "gmres" &&
            (residual_norm > args->loop_params->stopping_criteria && 
            args->loop_params->iter_count < args->loop_params->max_iters &&
            (args->loop_params->iter_count+1) % solver->gmres_args->restart_length == 0)
        )
            solver->restart_gmres(
                args->timers,
                args->sparse_mat,
                args->vec_size,
                args->coo_mat->n_cols,
                args->loop_params->iter_count
            );

        ++args->loop_params->iter_count;

#ifdef DEBUG_MODE_FINE
        if(args->loop_params->iter_count == 2){
            exit(1);
        }
#endif

    } while(residual_norm > args->loop_params->stopping_criteria && \
    args->loop_params->iter_count < args->loop_params->max_iters && \
    !isinf(static_cast<double>(residual_norm))); // TODO: <- isinf check not working

    args->flags->convergence_flag = ((residual_norm <= args->loop_params->stopping_criteria) && !isinf(static_cast<double>(residual_norm))) ? true : false;

    solver->save_x_star(args->timers, args->vec_size, args->loop_params->iter_count);

    record_residual_norm<VT>(
        args, 
        args->flags, 
        args->sparse_mat, 
        solver->r, 
        solver->x_star, 
        solver->b, 
        solver->x_star, 
        solver->tmp,
        solver->tmp_perm,
#ifdef USE_AP
        solver->x_star_dp, 
        solver->x_star_dp, 
        solver->tmp_dp,
        solver->tmp_perm_dp,
        solver->x_star_sp, 
        solver->x_star_sp, 
        solver->tmp_sp,
        solver->tmp_perm_sp,
#ifdef HAVE_HALF_MATH
        solver->x_star_hp, 
        solver->x_star_hp, 
        solver->tmp_hp,
        solver->tmp_perm_hp,
#endif
#endif
        &residual_norm
    );
}

#ifdef __CUDACC__
// void solve_gpu(
//     argType *args
// ){

//     // NOTE: Only for convenience. Will change to UM later.
//     double *h_residual_norm = new double;

//     // TODO: Why does this get messed up? 
//     args->loop_params->residual_count = 0;

//     // // Unpack relevant args
//     // double *d_x = args->d_x_old; // GS
//     // double *d_x_new = args->d_x_new; // Jacobi
//     // double *d_x_old = args->d_x_old; // Jacobi
//     // int d_n_rows = args->coo_mat->n_rows;

//     // // TODO: collect into a struct
//     // int *d_row_ptr = args->d_row_ptr;
//     // int *d_col = args->d_col;
//     // double *d_val = args->d_val; 

//     // double *d_tmp = args->d_tmp;
//     // double *d_D = args->d_D;
//     // double *d_r = args->d_r;
//     // double *d_b = args->d_b;

//     double *d_residual_norm;
//     cudaMalloc(&d_residual_norm, sizeof(double));
//     cudaMemset(d_residual_norm, 0.0, sizeof(double));

//     Flags *flags = args->flags; 

//     double residual_norm;

//     // TODO: Adapt for GPUs
//     // if(args->flags->print_iters){
//     //     iter_output(d_x, args->loop_params->iter_count);
//     //     printf("\n");
//     // }

//     // TODO: Adapt for GPUs
// // #ifdef DEBUG_MODE
// //     std::cout << "x vector:" << std::endl;
// //     for(int i = 0; i < args->vec_size; ++i){
// //         std::cout << d_x[i] << std::endl;
// //     }
// // #endif

//     // Begin timer
//     struct timeval calc_time_start, calc_time_end;
//     start_time(&calc_time_start);

//     do{
//         if(args->solver_type == "jacobi"){
//             // For a reference solution, not meant for use with USpMV library
//             // jacobi_iteration_ref_gpu<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new, args->vec_size);
//             jacobi_iteration_sep_gpu(args->vec_size, args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new);
//         }
//         else if(args->solver_type == "gauss-seidel"){
//             // TODO: Adapt for GPUs
//             printf("GS_solve still under development for GPUs.\n");
//             exit(1);
//             // For a reference solution, not meant for use with USpMV library
//             // gs_iteration_ref_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
//             // gs_iteration_sep_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
//         }
        
//         if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            
//             // Record residual every "residual_check_len" iterations
//             if(args->solver_type == "jacobi"){
//                 calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_new, args->d_b, args->d_r, args->d_tmp, args->vec_size);
//             }
//             else if(args->solver_type == "gauss-seidel"){
//                 // TODO: Adapt for GPUs
//                 printf("GS_solve still under development for GPUs.\n");
//                 exit(1);
//                 // calc_residual_gpu(sparse_mat, x, b, r, tmp);
//             }
            
// ///////////////////////////////////// Grrr DEBUG! //////////////////////////////////////////
//             // For now, have to do this on the CPU. Giving up on GPU implementation
//             // cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
//             // *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);
//             // TODO: Correct grid + block size?
//             // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
//             // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
//             infty_vec_norm_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(double)>>>(args->d_r, d_residual_norm, args->vec_size);
// ///////////////////////////////////// DEBUG! //////////////////////////////////////////

//             // TODO: Put residual_norm in unified memory to avoid this transfer
//             // NOTE: need to convert *double to *void
//             cudaMemcpy(&(*h_residual_norm), &(*d_residual_norm), sizeof(double), cudaMemcpyDeviceToHost);
//             // cudaDeviceSynchronize(); // <- not necessary
//             // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

//             // std::cout << "the first h_residual_norm = " << *h_residual_norm << std::endl;
//             // exit(0);
            
//             args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;
//             ++args->loop_params->residual_count;

// // TODO: Adapt for GPUs
// //             if(flags->print_iters){
// //                 if(args->solver_type == "jacobi"){
// //                     iter_output(x_new, args->loop_params->iter_count);
// //                 }
// //                 else if(args->solver_type == "gauss-seidel"){
// //                     iter_output(x, args->loop_params->iter_count);
// //                 }
// //             }
//         }

// // TODO: Adapt for GPUs
// // #ifdef DEBUG_MODE
// //         std::cout << "[";
// //         if(args->solver_type == "jacobi"){
// //             for(int i = 0; i < x_new->size(); ++i){
// //                 std::cout << (*x_new)[i] << ", ";
// //             }
// //         }
// //         else if (args->solver_type == "gauss-seidel"){
// //             for(int i = 0; i < x->size(); ++i){
// //                 std::cout << (*x)[i] << ", ";
// //             }
// //         }
// //         std::cout << "]" << std::endl;
  
// //         std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
// //         std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
// // #endif  

// // TODO: Adapt for GPUs???
//         cudaDeviceSynchronize();
//         if(args->solver_type == "jacobi"){
//             // NOTE: Might work, might not..
//             // TODO: huh?? Causes seg fault
//             // std::cout << "d_x_new pointer: " << d_x_new << std::endl;
//             // std::cout << "d_x_old pointer: " << d_x_old << std::endl;
//             std::swap(args->d_x_new, args->d_x_old);
//             // std::cout << "d_x_new pointer after swap: " << d_x_new << std::endl;
//             // std::cout << "d_x_old pointer after swap: " << d_x_old << std::endl;
//         }
    
//         ++args->loop_params->iter_count;

//     // TODO: Put residual_norm in unified memory to avoid this transfer
//     // cudaDeviceSynchronize();
//     // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

//     // Do check on host for now, easiest
//         // std::cout << *h_residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
//         // exit(0);
//     } while(*h_residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters);

//     args->flags->convergence_flag = (*h_residual_norm <= args->loop_params->stopping_criteria) ? true : false;

//     cudaDeviceSynchronize();
//     if(args->solver_type == "jacobi"){
//         // TODO: huh?? Causes seg fault
//         std::swap(args->d_x_old, args->d_x_star);
//     }
//     else if (args->solver_type == "gauss-seidel"){
//         // TODO: Adapt for GPUs
//         printf("GS_solve still under development for GPUs.\n");
//         exit(1);
//         // std::swap(*x, *(args->x_star));
//     }

//     // Record final residual with approximated solution vector x
// ///////////////////////////////////// DEBUG! //////////////////////////////////////////
//     // TODO: Giving up on GPU for this for now
//     cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
//     *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);

//     // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
//     // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
// ///////////////////////////////////////////////////////////////////////////////////////

//     // TODO: Put residual_norm in unified memory to avoid this transfer
//     // cudaDeviceSynchronize();
//     // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

//     args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;

// // TODO: Adapt for GPUs
// // #ifdef USE_USPMV
// //     // Bring final result vector out of permuted space
// //     std::vector<double> x_star_perm(args->vec_size, 0);
// //     apply_permutation(&(x_star_perm)[0], &(*args->x_star)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
// //     std::swap(x_star_perm, (*args->x_star));
// // #endif

//     // End timer
//     args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);

//     // Why are you freeing this here?
//     cudaFree(d_residual_norm);
//     delete h_residual_norm;
// }
#endif

template <typename VT>
void solve(
    argType<VT> *args,
    Solver<VT> *solver
){
    timeval *solver_harness_time_start = new timeval;
    timeval *solver_harness_time_end = new timeval;
    Stopwatch *solver_harness_wtime = new Stopwatch(solver_harness_time_start, solver_harness_time_end);
    args->timers->solver_harness_wtime = solver_harness_wtime ;
    args->timers->solver_harness_wtime->start_stopwatch();

    timeval *solver_time_start = new timeval;
    timeval *solver_time_end = new timeval;
    Stopwatch *solver_wtime = new Stopwatch(solver_time_start, solver_time_end);
    args->timers->solver_wtime = solver_wtime ;

#ifndef __CUDACC__
    solve_cpu(args, solver);
#else
    // TODO: adapt to refactoring
    // solve_gpu(args);
#endif

    args->timers->solver_harness_wtime->end_stopwatch();
}
#endif /*SOLVERS_H*/