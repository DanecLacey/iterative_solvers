#ifndef GMRES_H
#define GMRES_H

#include "../stopwatch.hpp"
#include "../sparse_matrix.hpp"

struct gmresArgs
{
    double beta;
    double *init_v;
    double *V;
    double *Vy;
    double *H;
    double *H_tmp;
    double *J;
    double *R;
    double *Q;
    double *Q_copy;
    double *g;
    double *g_copy;
    int restart_count;
    int restart_length;
};


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
    int restart_length
);

void allocate_gmres_structs(
    gmresArgs *gmres_args,
    int vec_size
);

void init_gmres_structs(
    gmresArgs *gmres_args,
    double *r,
    int n_rows
);

void init_gmres_timers(Timers *timers);

void gmres_get_x(
    double *R,
    double *g,
    double *x,
    double *x_0,
    double *V,
    double *Vy,
    int n_rows,
    int restart_count,
    int iter_count,
    int restart_len
);
#endif