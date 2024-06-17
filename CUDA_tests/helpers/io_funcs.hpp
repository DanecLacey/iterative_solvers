#ifndef IO_FUNCS_H
#define IO_FUNCS_H

#include "structs.hpp"
#include <string>

inline void sort_perm(int *arr, int *perm, int len, bool rev=false);

void read_mtx(
    COOMtxData *coo_mat,
    const std::string matrix_file_name
);

#endif