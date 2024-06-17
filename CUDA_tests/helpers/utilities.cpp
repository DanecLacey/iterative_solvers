#include "structs.hpp"
#include <stdio.h>
#include <time.h>
#include <limits>
#include <mkl.h>
#include <algorithm>
#include <numeric>

using ST=long;

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
bool will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0)
    {
        return false;
    }
    else if (a < 0 && b > 0)
    {
        return std::numeric_limits<T>::min() / b > a;
    }
    else if (a > 0 && b < 0)
    {
        return std::numeric_limits<T>::min() / a > b;
    }
    else if (a > 0 && b > 0)
    {
        return std::numeric_limits<T>::max() / a < b;
    }
    else
    {
        T difference =
            std::numeric_limits<T>::max() + std::numeric_limits<T>::min();

        if (difference == 0)
        { // symmetric case
            return std::numeric_limits<T>::min() / a < b * T{-1};
        }
        else
        { // abs(min) > max
            T c = std::numeric_limits<T>::min() - difference;

            if (a < c || b < c)
                return true;

            T ap = a * T{-1};
            T bp = b * T{-1};

            return std::numeric_limits<T>::max() / ap < bp;
        }
    }
}

void apply_permutation(
    double *permuted_vec,
    double *vec_to_permute,
    int *perm,
    int num_elems_to_permute
){
    // #pragma omp parallel for
    for(int i = 0; i < num_elems_to_permute; ++i){
        permuted_vec[i] = vec_to_permute[perm[i]];
        // std::cout << "Permuting:" << vec_to_permute[i] <<  " to " << vec_to_permute[perm[i]] << " using " << perm[i] <<  std::endl;
    }
    // printf("\n");
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
bool will_add_overflow(T a, T b)
{
    if (a > 0 && b > 0)
    {
        return std::numeric_limits<T>::max() - a < b;
    }
    else if (a < 0 && b < 0)
    {
        return std::numeric_limits<T>::min() - a > b;
    }

    return false;
}

template <typename IT>
void generate_inv_perm(
    int *perm,
    int *inv_perm,
    int perm_len
){
    for(int i = 0; i < perm_len; ++i){
        inv_perm[perm[i]] = i;
    }
}

void generate_x_and_y(
    double *x,
    double *y,
    int N,
    int nnz,
    bool rand_flag,
    double *values,
    double initial_val
){
    if(rand_flag){
        double upper_bound = std::numeric_limits<double>::min();
        double lower_bound = std::numeric_limits<double>::max();
        for(int i = 0; i < nnz; ++i){
            if(values[i] > upper_bound) upper_bound = values[i];
            if(values[i] < lower_bound) lower_bound = values[i];
        }
        srand(time(nullptr));

        double range = (upper_bound - lower_bound); 
        double div = RAND_MAX / range;

        for(int i = 0; i < N; ++i){
            x[i] = lower_bound + (rand() / div); //NOTE: expensive?
        }
    }
    else{
        for(int i = 0; i < N; ++i){
            x[i] = initial_val;
        }
    }

    for(int i = 0; i < N; ++i) y[i] = 0.0;

}

void convert_to_crs(
    COOMtxData *coo_mat,
    CRSMtxData *crs_mat
    )
{
    crs_mat->n_rows = coo_mat->n_rows;
    crs_mat->n_cols = coo_mat->n_cols;
    crs_mat->nnz = coo_mat->nnz;

    crs_mat->row_ptr = new int[crs_mat->n_rows+1];
    int *nnzPerRow = new int[crs_mat->n_rows];

    crs_mat->col = new int[crs_mat->nnz];
    crs_mat->val = new double[crs_mat->nnz];

    for(int idx = 0; idx < crs_mat->nnz; ++idx)
    {
        crs_mat->col[idx] = coo_mat->J[idx];
        crs_mat->val[idx] = coo_mat->values[idx];
    }

    for(int i = 0; i < crs_mat->n_rows; ++i)
    { 
        nnzPerRow[i] = 0;
    }

    //count nnz per row
    for(int i=0; i < crs_mat->nnz; ++i)
    {
        ++nnzPerRow[coo_mat->I[i]];
    }

    crs_mat->row_ptr[0] = 0;
    for(int i=0; i < crs_mat->n_rows; ++i)
    {
        crs_mat->row_ptr[i+1] = crs_mat->row_ptr[i]+nnzPerRow[i];
    }

    if(crs_mat->row_ptr[crs_mat->n_rows] != crs_mat->nnz)
    {
        printf("ERROR: converting to CRS.\n");
        exit(1);
    }

    delete[] nnzPerRow;
}

// template <typename VT, typename IT>
void convert_to_scs(
    COOMtxData *local_mtx,
    ST C,
    ST sigma,
    SCSMtxData<double, int> *scs,
    int *work_sharing_arr = nullptr,
    int my_rank = 0
)
{
    scs->nnz    = local_mtx->nnz;
    scs->n_rows = local_mtx->n_rows;
    scs->n_cols = local_mtx->n_cols;

    scs->C = C;
    scs->sigma = sigma;

    if (scs->sigma % scs->C != 0 && scs->sigma != 1) {
#ifdef DEBUG_MODE
    if(my_rank == 0){fprintf(stderr, "NOTE: sigma is not a multiple of C\n");}
#endif
    }

    if (will_add_overflow(scs->n_rows, scs->C)) {
#ifdef DEBUG_MODE
    if(my_rank == 0){fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");}
    exit(1);
#endif        
        // return false;
    }
    scs->n_chunks      = (local_mtx->n_rows + scs->C - 1) / scs->C;

    if (will_mult_overflow(scs->n_chunks, scs->C)) {
#ifdef DEBUG_MODE
    if(my_rank == 0){fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");}
    exit(1);
#endif   
        // return false;
    }
    scs->n_rows_padded = scs->n_chunks * scs->C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < local_mtx->nnz; ++i) {
        ++n_els_per_row[local_mtx->I[i]].second;
    }

    // sort rows in the scope of sigma
    if (will_add_overflow(scs->n_rows_padded, scs->sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        // return false;
    }

    for (ST i = 0; i < scs->n_rows_padded; i += scs->sigma) {
        auto begin = &n_els_per_row[i];
        auto end   = (i + scs->sigma) < scs->n_rows_padded
                        ? &n_els_per_row[i + scs->sigma]
                        : &n_els_per_row[scs->n_rows_padded];

        std::sort(begin, end,
                  // sort longer rows first
                  [](const auto & a, const auto & b) {
                    return a.second > b.second;
                  });
    }

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow
    // std::cout << d.n_chunks << std::endl;
    scs->chunk_lengths = V<int, int>(scs->n_chunks); // init a vector of length d.n_chunks
    scs->chunk_ptrs    = V<int, int>(scs->n_chunks + 1);

    int cur_chunk_ptr = 0;
    
    for (ST i = 0; i < scs->n_chunks; ++i) {
        auto begin = &n_els_per_row[i * scs->C];
        auto end   = &n_els_per_row[i * scs->C + scs->C];

        scs->chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, scs->chunk_lengths[i] * (int)scs->C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            // return false;
        }

        scs->chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += scs->chunk_lengths[i] * scs->C;
    }

    ST n_scs_elements = scs->chunk_ptrs[scs->n_chunks - 1]
                        + scs->chunk_lengths[scs->n_chunks - 1] * scs->C;

    // std::cout << "n_scs_elements = " << n_scs_elements << std::endl;
    scs->chunk_ptrs[scs->n_chunks] = n_scs_elements;

    // construct permutation vector

    scs->old_to_new_idx = V<int, int>(scs->n_rows);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        int old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < scs->n_rows) {
            scs->old_to_new_idx[old_row_idx] = i;
        }
    }
    

    scs->values   = V<double, int>(n_scs_elements);
    scs->col_idxs = V<int, int>(n_scs_elements);

    int padded_col_idx = 0;

    if(work_sharing_arr != nullptr){
        padded_col_idx = work_sharing_arr[my_rank];
    }

    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->values[i]   = double{};
        // scs->col_idxs[i] = int{};
        scs->col_idxs[i] = padded_col_idx;
    }

    std::vector<int> col_idx_in_row(scs->n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < scs->nnz; ++i) {
        int row_old = local_mtx->I[i];

        int row = scs->old_to_new_idx[row_old];

        ST chunk_index = row / scs->C;

        int chunk_start = scs->chunk_ptrs[chunk_index];
        int chunk_row   = row % scs->C;

        int idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;

        scs->col_idxs[idx] = local_mtx->J[i];
        scs->values[idx]   = local_mtx->values[i];

        col_idx_in_row[row]++;
    }

    // Sort inverse permutation vector, based on scs->old_to_new_idx
    std::vector<int> inv_perm(scs->n_rows);
    std::vector<int> inv_perm_temp(scs->n_rows);
    std::iota(std::begin(inv_perm_temp), std::end(inv_perm_temp), 0); // Fill with 0, 1, ..., scs->n_rows.

    generate_inv_perm<int>(scs->old_to_new_idx.data(), &inv_perm[0],  scs->n_rows);

    scs->new_to_old_idx = inv_perm;

    scs->n_elements = n_scs_elements;

    // Experimental 2024_02_01, I do not want the rows permuted yet... so permute back
    // if sigma > C, I can see this being a problem
    // for (ST i = 0; i < scs->n_rows_padded; ++i) {
    //     IT old_row_idx = n_els_per_row[i].first;

    //     if (old_row_idx < scs->n_rows) {
    //         scs->old_to_new_idx[old_row_idx] = i;
    //     }
    // }    

    // return true;
}

void compare_with_mkl(
    const double *y,
    const double *mkl_result,
    const int N,
    bool verbose_compare
){
    double max_error = 0.0;
    double error;

    for (int i = 0; i < N; i++)
        max_error = std::max(max_error, y[i] - mkl_result[i]);
    printf("Max error: %f\n", max_error);

    if(verbose_compare){
        for (int i = 0; i < N; i++){
            printf("idx: %i, error: %f\n", i, y[i] - mkl_result[i]);
        }
    }
}

void validate_dp_result(
    CRSMtxData *crs_mat,
    double const *x,
    double const *y
){    
    int n_rows = crs_mat->n_rows;
    int n_cols = crs_mat->n_cols;

    double *mkl_result = new double[n_rows];

    char transa = 'n';
    double alpha = 1.0;
    double beta = 0.0; 
    char matdescra [4] = {
        'G', // general matrix
        ' ', // ignored
        ' ', // ignored
        'C'}; // zero-based indexing (C-style)

    // Computes y := alpha*A*x + beta*y, for A -> m * k, 
    // mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    mkl_dcsrmv(
        &transa, 
        &n_rows, 
        &n_cols, 
        &alpha, 
        &matdescra[0], 
        crs_mat->val, 
        crs_mat->col, 
        crs_mat->row_ptr, 
        &(crs_mat->row_ptr)[1], 
        x, 
        &beta, 
        mkl_result
    );
    
    // Bool toggles verbose error checking
    compare_with_mkl(y, mkl_result, n_cols, false);
}