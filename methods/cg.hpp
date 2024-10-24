#ifndef CG_H
#define CG_H

#include <iomanip>
#include <cmath>

#include "../stopwatch.hpp"
#include "../kernels.hpp"
#include "../utility_funcs.hpp"
#include "../sparse_matrix.hpp"

#ifdef USE_USPMV
#include "../../Ultimate-SpMV/code/interface.hpp"
#endif

template <typename VT>
struct cgArgs
{
    VT alpha;
    VT beta;

    VT *p_old;
    VT *p_new;
    VT *r_old;
    VT *r_new;
#ifdef USE_USPMV
#ifdef USE_AP
    double *p_old_dp;
    double *p_new_dp;
    float *p_old_sp;
    float *p_new_sp;
#ifdef HAVE_HALF_MATH
    _Float16 *p_old_hp;
    _Float16 *p_new_hp;
#endif
#endif
#endif
};

template <typename VT>
void cg_iteration_ref_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    Timers *timers,
    std::string preconditioner_type,
    VT *alpha,
    VT *beta,
    VT *r,
    VT *x_old,
    VT *x_new,
    VT *tmp,
    VT *tmp_perm,
    VT *p_old,
    VT *p_new,
    VT *r_old,
    VT *r_new,
#ifdef USE_USPMV
#ifdef USE_AP
    double *tmp_dp,
    double *tmp_perm_dp,
    double *p_old_dp,
    double *p_new_dp,
    float *tmp_sp,
    float *tmp_perm_sp,
    float *p_old_sp,
    float *p_new_sp,
#ifdef HAVE_HALF_MATH
    _Float16 *tmp_hp,
    _Float16 *tmp_perm_hp,
    _Float16 *p_old_hp,
    _Float16 *p_new_hp,
#endif
#endif
#endif
    int n_rows
){
    VT num_tmp_dot;
    VT den_tmp_dot;
    // Compute SpMV: tmp <- Ap_old
    timers->cg_spmv_wtime->start_stopwatch();

#ifdef USE_USPMV
    execute_uspmv<VT, int>(
        &(sparse_mat->scs_mat->C),
        &(sparse_mat->scs_mat->n_chunks),
        sparse_mat->scs_mat->chunk_ptrs.data(),
        sparse_mat->scs_mat->chunk_lengths.data(),
        sparse_mat->scs_mat->col_idxs.data(),
        sparse_mat->scs_mat->values.data(),
        p_old, //input
        tmp_perm, //output
#ifdef USE_AP
        &(sparse_mat->scs_mat_dp->C),
        &(sparse_mat->scs_mat_dp->n_chunks),
        sparse_mat->scs_mat_dp->chunk_ptrs.data(),
        sparse_mat->scs_mat_dp->chunk_lengths.data(),
        sparse_mat->scs_mat_dp->col_idxs.data(),
        sparse_mat->scs_mat_dp->values.data(),
        p_old_dp, //input
        tmp_perm_dp, //output
        &(sparse_mat->scs_mat_sp->C),
        &(sparse_mat->scs_mat_sp->n_chunks),
        sparse_mat->scs_mat_sp->chunk_ptrs.data(),
        sparse_mat->scs_mat_sp->chunk_lengths.data(),
        sparse_mat->scs_mat_sp->col_idxs.data(),
        sparse_mat->scs_mat_sp->values.data(),
        p_old_sp, //input
        tmp_perm_sp, //output
#ifdef HAVE_HALF_MATH
        &(sparse_mat->scs_mat_hp->C),
        &(sparse_mat->scs_mat_hp->n_chunks),
        sparse_mat->scs_mat_hp->chunk_ptrs.data(),
        sparse_mat->scs_mat_hp->chunk_lengths.data(),
        sparse_mat->scs_mat_hp->col_idxs.data(),
        sparse_mat->scs_mat_hp->values.data(),
        p_old_hp, //input
        tmp_perm_hp, //output
#endif
#endif
        AP_VALUE_TYPE
    );

    timers->cg_spmv_wtime->end_stopwatch();

    // Permute rows back
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        apply_permutation<double, int>(tmp_dp, tmp_perm_dp, &(sparse_mat->scs_mat_dp->old_to_new_idx)[0], n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        apply_permutation<float, int>(tmp_sp, tmp_perm_sp, &(sparse_mat->scs_mat_sp->old_to_new_idx)[0], n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        apply_permutation<_Float16, int>(tmp_hp, tmp_perm_hp, &(sparse_mat->scs_mat_hp->old_to_new_idx)[0], n_rows);
#endif
    }
#else
    apply_permutation<VT, int>(tmp, tmp_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], n_rows);
#endif
#else
    // If you're not using USPMV, then just perform this spmv
    spmv_crs_cpu<VT, VT>(tmp, sparse_mat->crs_mat, p_old);
#endif
    
#ifdef DEBUG_MODE
    std::cout << "p_old = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << static_cast<double>(p_old[i]) << ", ";
        }
    std::cout << "]" << std::endl;

    std::cout << "tmp = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << static_cast<double>(tmp[i]) << ", ";
        }
    std::cout << "]" << std::endl;

    std::cout << "r_old = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << static_cast<double>(r_old[i]) << ", ";
        }
    std::cout << "]" << std::endl;
#endif

    // alpha <- (r_old, r_old) / (Ap_old, p_old)
    timers->cg_dot1_wtime->start_stopwatch();
    // NOTE: Could explicitly fuse dots if necessary
    dot<VT, VT>(r_old, r_old, &num_tmp_dot, n_rows);
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        dot<double, VT>(tmp_dp, p_old_dp, &den_tmp_dot, n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        dot<float, VT>(tmp_sp, p_old_sp, &den_tmp_dot, n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        dot<_Float16, VT>(tmp_hp, p_old_hp, &den_tmp_dot, n_rows);
#endif
    }
#else
    dot<VT, VT>(tmp, p_old, &den_tmp_dot, n_rows);
#endif
    // exit(1);
    *alpha = num_tmp_dot / den_tmp_dot;
    timers->cg_dot1_wtime->end_stopwatch();

    timers->cg_sum1_wtime->start_stopwatch();

    // x <- x + alpha*p_old
    // r_new <- r_old - alpha*Ap_old
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double"){
        sum_vectors_cpu<VT, VT, double, VT>(x_new, x_old, p_old_dp, n_rows, *alpha);
        subtract_vectors_cpu<VT, VT, double, VT>(r_new, r_old, tmp_dp, n_rows, *alpha);
    }
    else if(xstr(WORKING_PRECISION) == "float"){
        sum_vectors_cpu<VT, VT, float, VT>(x_new, x_old, p_old_sp, n_rows, *alpha);
        subtract_vectors_cpu<VT, VT, float, VT>(r_new, r_old, tmp_sp, n_rows, *alpha);
    }
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        sum_vectors_cpu<VT, VT, _Float16, VT>(x_new, x_old, p_old_hp, n_rows, *alpha);
        subtract_vectors_cpu<VT, VT, _Float16, VT>(r_new, r_old, tmp_hp, n_rows, *alpha);
#endif
    }
#else
    sum_vectors_cpu<VT,VT,VT, VT>(x_new, x_old, p_old, n_rows, *alpha);
    subtract_vectors_cpu<VT, VT, VT, VT>(r_new, r_old, tmp, n_rows, *alpha);
#endif
    timers->cg_sum1_wtime->end_stopwatch();

    // beta <- (r_new,r_new) / (r_old, r_old)
    timers->cg_dot2_wtime->start_stopwatch();
    // NOTE: Could explicitly fuse dots if necessary
    dot<VT, VT>(r_new, r_new, &num_tmp_dot, n_rows);
    dot<VT, VT>(r_old, r_old, &den_tmp_dot, n_rows);
    *beta = num_tmp_dot / den_tmp_dot;
    timers->cg_dot2_wtime->end_stopwatch();

    // p_new <- r_new + beta*p_old
    timers->cg_sum2_wtime->start_stopwatch();
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        sum_vectors_cpu<double, VT, double, VT>(p_new_dp, r_new, p_old_dp, n_rows, *beta);
    else if(xstr(WORKING_PRECISION) == "float")
        sum_vectors_cpu<float, VT, float, VT>(p_new_sp, r_new, p_old_sp, n_rows, *beta);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        sum_vectors_cpu<_Float16, VT, _Float16, VT>(p_new_hp, r_new, p_old_hp, n_rows, *beta);
#endif
    }
#else
    sum_vectors_cpu<VT, VT, VT, VT>(p_new, r_new, p_old, n_rows, *beta);
#endif
    timers->cg_sum2_wtime->end_stopwatch();

#ifdef DEBUG_MODE
    std::cout << "alpha = " << static_cast<double>(*alpha) << std::endl;
    std::cout << "beta = " << static_cast<double>(*beta) << std::endl;
    printf("p_new = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(p_new[i]) << ", ";
    }
    printf("]\n");

    printf("p_old = [");
#ifdef USE_AP
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(p_old_dp[i]) << ", ";
    }
#else
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(p_old[i]) << ", ";
    }
#endif
    printf("]\n");

    printf("r_new = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(r_new[i]) << ", ";
    }
    printf("]\n");

    printf("r_old = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(r_old[i]) << ", ";
    }
    printf("]\n");

    printf("x_new = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(x_new[i]) << ", ";
    }
    printf("]\n");

    printf("x_old = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(x_old[i]) << ", ";
    }
    printf("]\n");
    printf("\n");
#endif
// exit(1);
}

template <typename VT>
void allocate_cg_structs(
    cgArgs<VT> *cg_args,
    int n_rows
){
    std::cout << "Allocating space for CG structs" << std::endl;
    VT *p_old = new VT[n_rows];
    VT *p_new = new VT[n_rows];
    VT *r_old = new VT[n_rows];
    VT *r_new = new VT[n_rows];

#ifdef USE_USPMV
#ifdef USE_AP
    double *p_old_dp = new double[n_rows];
    double *p_new_dp = new double[n_rows];
    float *p_old_sp = new float[n_rows];
    float *p_new_sp = new float[n_rows];
#ifdef HAVE_HALF_MATH
    _Float16 *p_old_hp = new _Float16[n_rows];
    _Float16 *p_new_hp = new _Float16[n_rows];
#endif
#endif
#endif

    cg_args->p_old = p_old;
    cg_args->p_new = p_new;
    cg_args->r_old = r_old;
    cg_args->r_new = r_new;

#ifdef USE_USPMV
#ifdef USE_AP
    cg_args->p_old_dp = p_old_dp;
    cg_args->p_new_dp = p_new_dp;
    cg_args->p_old_sp = p_old_sp;
    cg_args->p_new_sp = p_new_sp;
#ifdef HAVE_HALF_MATH
    cg_args->p_old_hp = p_old_hp;
    cg_args->p_new_hp = p_new_hp;
#endif
#endif
#endif
}

template <typename VT>
void init_cg_structs(
    cgArgs<VT> *cg_args,
    VT *r,
    int n_rows
){
    // Copy residual to initial p
    for(int i = 0; i < n_rows; ++i){
        cg_args->p_old[i] = r[i];
        cg_args->p_new[i] = (VT)0.0;
        cg_args->r_old[i] = r[i];
        cg_args->r_new[i] = (VT)0.0;
#ifdef USE_USPMV
#ifdef USE_AP
        cg_args->p_old_dp[i] = static_cast<double>(r[i]);
        cg_args->p_new_dp[i] = 0.0;
        cg_args->p_old_sp[i] = static_cast<float>(r[i]);
        cg_args->p_new_sp[i] = 0.0f;
#ifdef HAVE_HALF_MATH
        cg_args->p_old_hp[i] = static_cast<_Float16>(r[i]);
        cg_args->p_new_hp[i] = 0.0f16;
#endif
#endif
#endif
    }

}

void init_cg_timers(Timers *timers){
    timeval *cg_spmv_start = new timeval;
    timeval *cg_spmv_end = new timeval;
    Stopwatch *cg_spmv_wtime = new Stopwatch(cg_spmv_start, cg_spmv_end);
    timers->cg_spmv_wtime = cg_spmv_wtime;

    timeval *cg_dot1_start = new timeval;
    timeval *cg_dot1_end = new timeval;
    Stopwatch *cg_dot1_wtime = new Stopwatch(cg_dot1_start, cg_dot1_end);
    timers->cg_dot1_wtime = cg_dot1_wtime;

    timeval *cg_dot2_start = new timeval;
    timeval *cg_dot2_end = new timeval;
    Stopwatch *cg_dot2_wtime = new Stopwatch(cg_dot2_start, cg_dot2_end);
    timers->cg_dot2_wtime = cg_dot2_wtime;

    timeval *cg_sum1_start = new timeval;
    timeval *cg_sum1_end = new timeval;
    Stopwatch *cg_sum1_wtime = new Stopwatch(cg_sum1_start, cg_sum1_end);
    timers->cg_sum1_wtime = cg_sum1_wtime;

    timeval *cg_sum2_start = new timeval;
    timeval *cg_sum2_end = new timeval;
    Stopwatch *cg_sum2_wtime = new Stopwatch(cg_sum2_start, cg_sum2_end);
    timers->cg_sum2_wtime = cg_sum2_wtime;

    // timeval *cg_apply_preconditioner_start = new timeval;
    // timeval *cg_apply_preconditioner_end = new timeval;
    // Stopwatch *cg_apply_preconditioner_wtime = new Stopwatch(cg_apply_preconditioner_start, cg_apply_preconditioner_end);
    // timers->cg_apply_preconditioner_wtime = cg_apply_preconditioner_wtime;
}
#endif