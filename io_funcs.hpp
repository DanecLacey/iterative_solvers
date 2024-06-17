#ifndef IO_FUNCS_H
#define IO_FUNCS_H

inline void sort_perm(int *arr, int *perm, int len, bool rev=false);

void assign_cli_inputs(
    int argc,
    char *argv[],
    std::string *matrix_file_name,
    std::string *solver_type
);

void read_mtx(
    const std::string matrix_file_name,
    COOMtxData *coo_mat
);

void summary_output(
    double *residuals_vec,
    std::string *solver_type,
    LoopParams loop_params,
    Flags flags,
    double total_time_elapsed,
    double calc_time_elapsed
);

void iter_output(
    const double *x_approx,
    int N,
    int iter_count
);

void residuals_output(
    bool print_residuals,
    double* residuals_vec,
    LoopParams loop_params
);

void write_residuals_to_file(std::vector<double> *residuals_vec);

void write_comparison_to_file(
    std::vector<double> *x_star,
    double iterative_final_residual,
    std::vector<double> *x_direct,
    double direct_final_residual
);

void postprocessing(
    argType *args
);
#endif /*IO_FUNCS_H*/