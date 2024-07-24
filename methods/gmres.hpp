#include "../structs.hpp"

void gmres_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    Timers *timers,
    double *V,
    double *H,
    double *H_tmp,
    double *J,
    double *Q,
    double *Q_copy,
    double *w,
    double *R,
    double *g,
    double *g_copy,
    double *b,
    double *x,
    double beta,
    int n_rows,
    int restart_count,
    int iter_count,
    double *residual_norm,
    int max_gmres_iters
);