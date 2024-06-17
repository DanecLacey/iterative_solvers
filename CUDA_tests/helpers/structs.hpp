#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>
#include "vectors.h"
#include "mmio.h"

using ST=long;

template <typename VT, typename IT>
using V = Vector<VT, IT>;

struct COOMtxData
{
    int n_rows{};
    int n_cols{};
    int nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<int> I;
    std::vector<int> J;
    std::vector<double> values;
};


struct CRSMtxData
{
    int n_rows{};
    int n_cols{};
    int nnz{};

    int *row_ptr, *col;
    double *val;
};

template <typename VT, typename IT>
struct SCSMtxData
{
    ST C{};
    ST sigma{};

    ST n_rows{};
    ST n_cols{};
    ST n_rows_padded{};
    ST n_chunks{};
    ST n_elements{}; // No. of nz + padding.
    ST nnz{};        // No. of nz only.

    V<IT, IT> chunk_ptrs;    // Chunk start offsets into col_idxs & values.
    V<IT, IT> chunk_lengths; // Length of one row in a chunk.
    V<IT, IT> col_idxs;
    V<VT, IT> values;
    V<IT, IT> old_to_new_idx;
    std::vector<int> new_to_old_idx; //inverse of above
    // TODO: ^ make V object as well?

    void permute(IT *_perm_, IT*  _invPerm_);
    void write_to_mtx_file(std::string file_out_name);
    
};

template <typename VT, typename IT>
void SCSMtxData<VT, IT>::permute(IT *_perm_, IT*  _invPerm_){
    int nrows = n_rows; // <- stupid

    // TODO: not efficient, but a workaround
    IT *rowPtr = new IT[nrows+1];
    IT *col = new IT[nnz];
    VT *val = new VT[nnz];

    for(int i = 0; i < nrows + 1; ++i){
        rowPtr[i] = (chunk_ptrs.data())[i];
    }
    for(int i = 0; i < nnz; ++i){
        col[i] = (col_idxs.data())[i];
        val[i] = (values.data())[i];
    } 
    

    VT* newVal = (VT*)malloc(sizeof(VT)*nnz);
        //new double[block_size*block_size*nnz];
    IT* newCol = (IT*)malloc(sizeof(IT)*nnz);
        //new int[nnz];
    IT* newRowPtr = (IT*)malloc(sizeof(IT)*(nrows+1));
        //new int[nrows+1];
/*
    double *newVal = (double*) malloc(sizeof(double)*nnz);
    int *newCol = (int*) malloc(sizeof(int)*nnz);
    int *newRowPtr = (int*) malloc(sizeof(int)*(nrows+1));
*/

    newRowPtr[0] = 0;

    if(_perm_ != NULL)
    {
        //first find newRowPtr; therefore we can do proper NUMA init
        IT _perm_Idx=0;
#ifdef DEBUG_MODE
    // if(my_rank == 0){printf("nrows = %d\n", nrows);}
#endif
        
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            IT _perm_Row = _perm_[row];
            for(int idx=rowPtr[_perm_Row]; idx<rowPtr[_perm_Row+1]; ++idx)
            {
                ++_perm_Idx;
            }
            newRowPtr[row+1] = _perm_Idx;
        }
    }
    else
    {
        for(int row=0; row<nrows+1; ++row)
        {
            newRowPtr[row] = rowPtr[row];
        }
    }

    if(_perm_ != NULL)
    {
        //with NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            IT _perm_Row = _perm_[row];

            for(int _perm_Idx=newRowPtr[row],idx=rowPtr[_perm_Row]; _perm_Idx<newRowPtr[row+1]; ++idx,++_perm_Idx)
            {
                //_perm_ute column-wise also
                // guard added 22.12.22
                if(col[idx] < nrows){ // col[_perm_Idx] < nrows) ?
                    newCol[_perm_Idx] = _invPerm_[col[idx]]; // permute column of "local" elements
                }
                else{
                    newCol[_perm_Idx] = col[idx]; //do not permute columns of remote elements
                }
                
                // newCol[_perm_Idx] = _invPerm_[col[idx]]; // <- old
                // printf("permIdx = %d, idx = %d, col[permIdx] = %d, col[idx] = %d\n",_perm_Idx, idx, col[_perm_Idx],col[idx] );

                newVal[_perm_Idx] = val[idx]; // in both cases, value is permuted

                // if(newCol[_perm_Idx] >= n_rows && col[_perm_Idx] < nrows){
                //     printf("permute ERROR: local element from index %d and col %d was permuted out of it's bounds to %d.\n", idx, col[idx], newCol[_perm_Idx]);
                //     exit(1);
                // }
                if (newCol[_perm_Idx] >= n_cols){
                    printf("SCS permute ERROR: Element at index %d has blow up column index: %d.\n", _perm_Idx,newCol[_perm_Idx]);
                    exit(1);     
                }
                if (newCol[_perm_Idx] < 0){
                    printf("SCS permute ERROR: Element at index %d has negative column index: %d.\n", _perm_Idx, newCol[_perm_Idx]);
                    exit(1);
                }
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            for(int idx=newRowPtr[row]; idx<newRowPtr[row+1]; ++idx)
            {
                newCol[idx] = col[idx];
                newVal[idx] = val[idx];
            }
        }
    }

    for(int i = 0; i < nrows + 1; ++i){
        chunk_ptrs[i] = newRowPtr[i];
    } 
    for(int i = 0; i < nnz; ++i){
        col_idxs[i] = newCol[i];
        values[i] = newVal[i];
    }

    //free old _perm_utations
    delete[] val;
    delete[] rowPtr;
    delete[] col;
    delete[] newVal;
    delete[] newRowPtr;
    delete[] newCol;
}

template <typename VT, typename IT>
void SCSMtxData<VT, IT>::write_to_mtx_file(
    std::string file_out_name)
{
    // NOTE: will write all scs matrices as double precision. Doesn't matter now, but might in the future
    // Convert csr back to coo for mtx format printing
    // TODO: Haven't quite figured out for SCS, C>1
    // std::vector<int> temp_rows(n_elements);
    // std::vector<int> temp_cols(n_elements);
    // std::vector<double> temp_values(n_elements);

    ST n_scs_elements = chunk_ptrs[n_chunks - 1]
                        + chunk_lengths[n_chunks - 1] * C;

    std::vector<int> temp_rows(n_scs_elements);
    std::vector<int> temp_cols(n_scs_elements);
    std::vector<double> temp_values(n_scs_elements);

    int tmp_row_cntr = 0;
    int line_cntr = 0;
    int elems_in_row = 0;
    for(int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx){ // for each chunk
        int elems_in_chunk = chunk_ptrs[chunk_idx + 1] - chunk_ptrs[chunk_idx];
        for(int chunk_elem = 0; chunk_elem < elems_in_chunk; ++chunk_elem){ 
            // scan each element in this chunk
            temp_rows[tmp_row_cntr] = line_cntr + 1;
            ++elems_in_row;
           
            if(elems_in_row * C == elems_in_chunk){
                // If we reach the end of the row in this chunk
                ++line_cntr;
                elems_in_row = 0;
            }
            ++tmp_row_cntr;
        }
        ++line_cntr;
    }

    // TODO: temp_cols and temp_values should only have n_elements, not n_scs_elements...?
    // or should they actually be n_scs_elements?
    for(int idx = 0; idx < n_scs_elements; ++idx){
        temp_cols[idx] = col_idxs[idx] + 1;
        temp_values[idx] = values[idx]; // Just keep as doubles for writing to mtx file
    }

    std::string out_file_name = file_out_name + ".mtx"; 

    mm_write_mtx_crd(
        &out_file_name[0], 
        n_rows, 
        n_cols, 
        n_elements, 
        &(temp_rows)[0], 
        &(temp_cols)[0], 
        &(temp_values)[0], 
        "MCRG" // TODO: <- make more general, i.e. flexible based on the matrix. Read from original mtx?
    );
}

#endif