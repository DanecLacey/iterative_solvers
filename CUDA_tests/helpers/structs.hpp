#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>

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

#endif