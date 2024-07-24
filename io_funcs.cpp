#include "mmio.h"
#include "utility_funcs.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

inline void sort_perm(int *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

void assign_cli_inputs(
    argType *args, 
    int argc,
    char *argv[],
    std::string *matrix_file_name
    )
{
    if(argc != 3){
        printf("ERROR: assign_cli_inputs: Please only select a .mtx file name and solver type [-j (Jacobi) / -gs (Gauss-Seidel) / -cg (Conjugate Gradient)].\n");
        exit(1);
    }

#ifdef USE_SCAMAC
    args->scamac_args = argv[1];
#else
    *matrix_file_name = argv[1];
    // if(fn.substr(matrix_file_name->find_last_of(".") + 1) == "mtx")
    //     printf("ERROR: assign_cli_inputs: Verify you are using an .mtx file. \n");
    // exit(1);
#endif

    std::string st = argv[2];

    if(st == "-j"){
        args->solver_type = "jacobi"; 
    }
    else if(st == "-gs"){
        args->solver_type = "gauss-seidel";
    }
    else if(st == "-cg"){
        args->solver_type = "conjugate-gradient";
        printf("ERROR: assign_cli_inputs: Conjugate Gradient [-cg] is still under development.\n");
        exit(1);
    }
    else if(st == "-gm"){
        args->solver_type = "gmres";
    }
    else{
        printf("ERROR: assign_cli_inputs: Please choose an available solver type [-j (Jacobi) / -gs (Gauss-Seidel) / -cg (Conjugate Gradient) / -gm (GMRES)].\n");
        exit(1);
    }

}

void read_mtx(
    const std::string matrix_file_name,
    COOMtxData *coo_mat
    )
{
    char* filename = const_cast<char*>(matrix_file_name.c_str());
    int nrows, ncols, nnz;
    double *val_ptr;
    int *I_ptr;
    int *J_ptr;

    MM_typecode matcode;
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL) {printf("Unable to open file");}

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", filename);
        // return -1;
    }

    fclose(f);

    // bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode)||mm_is_integer(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool symm_flag = mm_is_symmetric(matcode);
    bool pattern_flag = mm_is_pattern(matcode);

    if(!compatible_flag)
    {
        printf("The matrix market file provided is not supported.\n Reason :\n");
        if(!mm_is_sparse(matcode))
        {
            printf(" * matrix has to be sparse\n");
        }

        if(!mm_is_real(matcode) && !(mm_is_pattern(matcode)))
        {
            printf(" * matrix has to be real or pattern\n");
        }

        if(!mm_is_symmetric(matcode) && !mm_is_general(matcode))
        {
            printf(" * matrix has to be einther general or symmetric\n");
        }

        exit(0);
    }

    //int ncols;
    int *row_unsorted;
    int *col_unsorted;
    double *val_unsorted;

    if(mm_read_unsymmetric_sparse<double, int>(filename, &nrows, &ncols, &nnz, &val_unsorted, &row_unsorted, &col_unsorted) < 0)
    {
        printf("Error in file reading\n");
        exit(1);
    }
    if(nrows != ncols)
    {
        printf("Matrix not square. Currently only square matrices are supported\n");
        exit(1);
    }

    //If matrix market file is symmetric; create a general one out of it
    if(symm_flag)
    {
        // printf("Creating a general matrix out of a symmetric one\n");

        int ctr = 0;

        //this is needed since diagonals might be missing in some cases
        for(int idx=0; idx<nnz; ++idx)
        {
            ++ctr;
            if(row_unsorted[idx]!=col_unsorted[idx])
            {
                ++ctr;
            }
        }

        int new_nnz = ctr;

        int *row_general = new int[new_nnz];
        int *col_general = new int[new_nnz];
        double *val_general = new double[new_nnz];

        int idx_gen=0;

        for(int idx=0; idx<nnz; ++idx)
        {
            row_general[idx_gen] = row_unsorted[idx];
            col_general[idx_gen] = col_unsorted[idx];
            val_general[idx_gen] = val_unsorted[idx];
            ++idx_gen;

            if(row_unsorted[idx] != col_unsorted[idx])
            {
                row_general[idx_gen] = col_unsorted[idx];
                col_general[idx_gen] = row_unsorted[idx];
                val_general[idx_gen] = val_unsorted[idx];
                ++idx_gen;
            }
        }

        free(row_unsorted);
        free(col_unsorted);
        free(val_unsorted);

        nnz = new_nnz;

        //assign right pointers for further proccesing
        row_unsorted = row_general;
        col_unsorted = col_general;
        val_unsorted = val_general;

        // delete[] row_general;
        // delete[] col_general;
        // delete[] val_general;
    }

    //permute the col and val according to row
    int *perm = new int[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(row_unsorted, perm, nnz);

    int *col = new int[nnz];
    int *row = new int[nnz];
    double *val = new double[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        col[idx] = col_unsorted[perm[idx]];
        val[idx] = val_unsorted[perm[idx]];
        row[idx] = row_unsorted[perm[idx]];
    }

    delete[] perm;
    delete[] col_unsorted;
    delete[] val_unsorted;
    delete[] row_unsorted;

    coo_mat->values = std::vector<double>(val, val + nnz);
    coo_mat->I = std::vector<int>(row, row + nnz);
    coo_mat->J = std::vector<int>(col, col + nnz);
    coo_mat->n_rows = nrows;
    coo_mat->n_cols = ncols;
    coo_mat->nnz = nnz;
    coo_mat->is_sorted = 1; // TODO: not sure
    coo_mat->is_symmetric = 0; // TODO: not sure

    delete[] val;
    delete[] row;
    delete[] col;
}

void residuals_output(
    bool print_residuals,
    double* residuals_vec,
    LoopParams loop_params
){
    for(int i = 0; i < loop_params.residual_count; ++i){
        std::cout << "||A*x_" << i*loop_params.residual_check_len << " - b||_infty = " << std::setprecision(16) << residuals_vec[i] << std::endl;
    }
}

void summary_output(
    argType *args,
    double *residuals_vec, // Here is the problem. Why would you send the entire residuals vec??
    std::string *solver_type,
    LoopParams loop_params,
    Flags flags
){
    if(flags.print_residuals){
        residuals_output(flags.print_residuals, residuals_vec, loop_params);
    }

    if(flags.convergence_flag){
        // x_new ~ A^{-1}b
        std::cout << "\n" << *solver_type << " solver converged in: " << loop_params.iter_count << " iterations." << std::endl;
    }
    else{
        // x_new !~ A^{-1}b
        std::cout << "\n" << *solver_type << " solver did not converge after " << loop_params.max_iters << " iterations." << std::endl;
    }
    std::cout << "With the stopping criteria \"tol * || b-A*x_0 ||_infty\" is: " << loop_params.stopping_criteria << std::endl;
#ifdef USE_AP
    std::cout << "and AP splits: " << 100*args->lp_percent << "\% lp elements and " << 100*args->hp_percent << "\% hp elements" <<  std::endl;
#endif
    std::cout << "The residual of the final iteration is: ||A*x_star - b||_infty = " <<
    std::scientific << residuals_vec[loop_params.residual_count] << ".\n";
}

void write_residuals_to_file(std::vector<double> *residuals_vec){
    std::fstream out_file;
    out_file.open("residuals.txt", std::fstream::in | std::fstream::out | std::fstream::app);

    for(int i = 0; i < residuals_vec->size(); ++i){
        out_file << i << " " << (*residuals_vec)[i] << "\n";
    }
    out_file.close();
}

void write_comparison_to_file(
    std::vector<double> *x_star,
    double iterative_final_residual,
    std::vector<double> *x_direct,
    double direct_final_residual
){
    std::fstream out_file;
    out_file.open("sparse_to_direct.txt", std::fstream::in | std::fstream::out | std::fstream::app);

    out_file << "Iterative Method Final Residual: " << iterative_final_residual << ", Direct Method Final Residual: " << direct_final_residual << std::endl;

    for(int i = 0; i < x_star->size(); ++i){
        out_file << "idx: " << i << ", " << (*x_star)[i] << " - " <<  (*x_direct)[i] << " = " << (*x_star)[i] - (*x_direct)[i] << "\n";
    }
    out_file.close();
}

void print_timers(argType *args){
    long double total_wtime = args->timers->total_wtime->get_wtime();
    long double preprocessing_wtime = args->timers->preprocessing_wtime->get_wtime();
    long double solver_harness_wtime = args->timers->solver_harness_wtime->get_wtime();
    long double solver_wtime = args->timers->solver_wtime->get_wtime();
    long double spmv_wtime;
    long double orthog_wtime; 
    long double mgs_wtime;
    long double mgs_dot_wtime;
    long double mgs_sub_wtime;
    long double leastsq_wtime;
    long double compute_H_tmp_wtime;
    long double compute_Q_wtime;
    long double compute_R_wtime;
    long double get_x_wtime;

    if(args->solver_type == "gmres"){
        spmv_wtime = args->timers->gmres_spmv_wtime->get_wtime();
        orthog_wtime = args->timers->gmres_orthog_wtime->get_wtime();
        mgs_wtime = args->timers->gmres_mgs_wtime->get_wtime();
#ifdef FINE_TIMERS
        mgs_dot_wtime = args->timers->gmres_mgs_dot_wtime->get_wtime();
        mgs_sub_wtime = args->timers->gmres_mgs_sub_wtime->get_wtime();
#endif
        leastsq_wtime = args->timers->gmres_leastsq_wtime->get_wtime();
#ifdef FINE_TIMERS
        compute_H_tmp_wtime = args->timers->gmres_compute_H_tmp_wtime->get_wtime();
        compute_Q_wtime = args->timers->gmres_compute_Q_wtime->get_wtime();
        compute_R_wtime = args->timers->gmres_compute_R_wtime->get_wtime();
#endif
        get_x_wtime = args->timers->gmres_get_x_wtime->get_wtime();
    }

    int right_flush_width = 30;
    int left_flush_width = 25;

    std::cout << "+---------------------------------------------------------+" << std::endl;

    std::cout << std::left << std::setw(left_flush_width) << "Total elapsed time: " << std::right << std::setw(right_flush_width) << total_wtime  << "[s]" << std::endl;
    std::cout << std::left << std::setw(left_flush_width) << "| Preprocessing time: " << std::right << std::setw(right_flush_width) << preprocessing_wtime  << "[s]" << std::endl;
    std::cout << std::left << std::setw(left_flush_width) << "| Solver Harness time: " << std::right << std::setw(right_flush_width) << solver_harness_wtime  << "[s]" << std::endl;

    if(args->solver_type == "jacobi"){
        std::cout << std::left << std::setw(left_flush_width) << "| | Jacobi Solver time: " << std::right << std::setw(right_flush_width) << solver_wtime << "[s]" << std::endl;
    }
    else if(args->solver_type == "gauss-seidel"){
        std::cout << std::left << std::setw(left_flush_width) << "| | GS Solver time: " << std::right << std::setw(right_flush_width) << solver_wtime << "[s]" << std::endl;
    }

    if(args->solver_type == "gmres"){
        std::cout << std::left << std::setw(left_flush_width) << "| | GMRES Solver time: " << std::right << std::setw(right_flush_width) << solver_wtime << "[s]" << std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| | | SpMV: "               << std::right << std::setw(right_flush_width) << spmv_wtime  << "[s]" <<std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| | | Orthogonalization: "  << std::right << std::setw(right_flush_width) << orthog_wtime  << "[s]" <<std::endl;
#ifdef FINE_TIMERS 
        std::cout << std::left << std::setw(left_flush_width) << "| | | | MGS: "              << std::right << std::setw(right_flush_width) << mgs_wtime  << "[s]" <<std::endl;       
        std::cout << std::left << std::setw(left_flush_width) << "| | | | | Dot: "              << std::right << std::setw(right_flush_width) << mgs_dot_wtime  << "[s]" <<std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| | | | | Sub: "              << std::right << std::setw(right_flush_width) << mgs_sub_wtime  << "[s]" <<std::endl;
        long double orthog_tmp = mgs_wtime;
        std::cout << std::left << std::setw(left_flush_width) << "| | | | Other: "            << std::right << std::setw(right_flush_width) << orthog_wtime - orthog_tmp  << "[s]" <<std::endl;
#endif
        std::cout << std::left << std::setw(left_flush_width) << "| | | Givens Rotations: "   << std::right << std::setw(right_flush_width) << leastsq_wtime  << "[s]" <<std::endl;
#ifdef FINE_TIMERS
        std::cout << std::left << std::setw(left_flush_width) << "| | | | Compute H_tmp: "    << std::right << std::setw(right_flush_width) << compute_H_tmp_wtime  << "[s]" <<std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| | | | Compute Q: "        << std::right << std::setw(right_flush_width) << compute_Q_wtime  << "[s]" <<std::endl;
        std::cout << std::left << std::setw(left_flush_width) << "| | | | Compute R: "        << std::right << std::setw(right_flush_width) << compute_R_wtime  << "[s]" <<std::endl;
        long double leastsq_tmp = compute_H_tmp_wtime + compute_Q_wtime + compute_R_wtime; 
        std::cout << std::left << std::setw(left_flush_width) << "| | | | Other: "            << std::right << std::setw(right_flush_width) << leastsq_wtime - leastsq_tmp << "[s]" << std::endl;
#endif
        std::cout << std::left << std::setw(left_flush_width) << "| | | Get x: "               << std::right << std::setw(right_flush_width) << get_x_wtime  << "[s]" <<std::endl;

        solver_wtime += get_x_wtime;

    }
    std::cout << std::left << std::setw(left_flush_width) << "| | Other: " << std::right << std::setw(right_flush_width) << solver_harness_wtime - solver_wtime << "[s]" << std::endl;

    std::cout << "+---------------------------------------------------------+" << std::endl;

    // Validate timers
    // long double timer_sum = 0.0;
    // timer_sum += preprocessing_wtime;
}

void postprocessing(
    argType *args
){
    std::vector<double> *x_star;
    // double *normed_residuals;

#ifdef __CUDACC__
// Something weird is going on in here
    // printf("args->x_star[0] = %f\n", (*args->x_star)[0]);
    // printf("args->x_star[1] = %f\n", (*args->x_star)[1]);
    // printf("args->d_x_star[0] = %f\n", args->d_x_star[0]);
    // printf("args->d_x_star[1] = %f\n", args->d_x_star[1]);
    cudaMemcpy(args->x_star, args->d_x_star, args->vec_size*sizeof(double), cudaMemcpyDeviceToHost);
    // printf("args->x_star[0] = %f\n", (*args->x_star)[0]);
    // printf("args->x_star[1] = %f\n", (*args->x_star)[1]);
    // printf("args->d_x_star[0] = %f\n", args->d_x_star[0]);
    // printf("args->d_x_star[1] = %f\n", args->d_x_star[1]);

    // printf("args->normed_residuals[0] = %f", args->normed_residuals[0]);
    // printf("args->normed_residuals[1] = %f", args->normed_residuals[1]);
    // // printf("args->d_normed_residuals[0] = %f", args->d_normed_residuals[0]);
    // // printf("args->d_normed_residuals[1] = %f", args->d_normed_residuals[1]);
    // cudaMemcpy(args->normed_residuals, args->d_normed_residuals, 1*sizeof(double), cudaMemcpyDeviceToHost);

    // TODO: I GUESS! :(
    // cudaMemcpy(args->normed_residuals, args->d_normed_residuals, (args->loop_params->max_iters / args->loop_params->residual_check_len + 1)*sizeof(double), cudaMemcpyDeviceToHost);
    // printf("args->normed_residuals[0] = %f", args->normed_residuals[0]);
    // printf("args->normed_residuals[1] = %f", args->normed_residuals[1]);
    // // printf("args->d_normed_residuals[0] = %f", args->d_normed_residuals[0]);
    // // printf("args->d_normed_residuals[1] = %f", args->d_normed_residuals[1]);
    // // cudaDeviceSynchronize();
#endif

    // x_star = args->x_star;
    // normed_residuals = args->normed_residuals;
    // std::string solver_type = args->solver_type;
    // LoopParams *loop_params = args->loop_params;
    // double total_time_elapsed = args->total_time_elapsed;
    // double calc_time_elapsed = args->calc_time_elapsed;

    // TODO: include SCS vs. CRS comparison for validation
    // if(flags.compare_direct){
    //     compare_with_direct(crs_mat, matrix_file_name, loop_params, x_star, (*normed_residuals)[loop_params.residual_count]);
    // }

    args->timers->total_wtime->end_stopwatch();
    // std::cout << "TOTAL TIME ELAPSED: " << args->timers->total_wtime->get_wtime() << std::endl;

    if(args->flags->print_summary){
        summary_output(
            args,
            args->normed_residuals, 
            &args->solver_type, 
            *args->loop_params, 
            *args->flags
        );
    }

    print_timers(args);

#ifdef DEBUG_MODE_FINE
    std::cout << "The solution vector is x = [" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        printf("%f, ", args->x_star[i]);
    }
    std::cout << "]" << std::endl;
#endif
    
    //sufficent to just print to stdout for now
    // if(flags.print_residuals){
    //     write_residuals_to_file(normed_residuals);
    // }
}