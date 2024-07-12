#ifndef IO_FUNCS_H
#define IO_FUNCS_H

inline void sort_perm(int *arr, int *perm, int len, bool rev=false);

void assign_cli_inputs(
    argType *args, 
    int argc,
    char *argv[],
    std::string *matrix_file_name
);

void read_mtx(
    const std::string matrix_file_name,
    COOMtxData *coo_mat
);

void summary_output(
    double *residuals_vec,
    std::string *solver_type,
    LoopParams loop_params,
    Flags flags
);

void print_timers(Timers *timers);

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