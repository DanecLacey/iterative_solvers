#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>
#include <iostream>

// NOTE: Is not having default bools really bad?
struct Flags
{
    bool print_iters;
    bool print_summary;
    bool print_residuals;
    bool convergence_flag;
    bool export_errors;
};

// TODO: Make user-enetered option available
struct LoopParams
{
    int iter_count;
    int max_iters;
    double tol;
};

// TODO: Make class with member funcitons
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

    bool operator==(COOMtxData &rhs)
    {
    return (
        (n_rows == rhs.n_rows) &&
        (n_cols == rhs.n_cols) &&
        (nnz == rhs.nnz) &&
        (is_sorted == rhs.is_sorted) &&
        (is_symmetric == rhs.is_symmetric) &&
        (I == rhs.I) &&
        (J == rhs.J) &&
        (values == rhs.values)
    );
    }

    void operator^(COOMtxData &rhs)
    {
        if (n_rows != rhs.n_rows){
            std::cout << "n_rows != rhs.n_rows" << std::endl;
        }
        if (n_cols != rhs.n_cols){
            std::cout << "n_cols != rhs.n_cols" << std::endl;
        }
        if (nnz != rhs.nnz){
            std::cout << "nnz != rhs.nnz" << std::endl;
        }
        if (is_sorted != rhs.is_sorted){
            std::cout << "is_sorted == rhs.is_sorted" << std::endl;
        }
        if (is_symmetric != rhs.is_symmetric){
            std::cout << "is_symmetric != rhs.is_symmetric" << std::endl;
        }
        if(I != rhs.I){
            std::cout << "I != rhs.I" << std::endl;
        }
        if(I.size() != rhs.I.size()){
            std::cout << "I.size() " << I.size() << " != rhs.I.size() " << rhs.I.size() << std::endl;
        }
        if(J != rhs.J){
            std::cout << "J != rhs.J" << std::endl;
        }
        if(J.size() != rhs.J.size()){
            std::cout << "J.size() " << J.size() << " != rhs.J.size() " << rhs.I.size() << std::endl;
        }
        if(values != rhs.values){
            std::cout << "values != rhs.values" << std::endl;
        }
        if(values.size() != rhs.values.size()){
            std::cout << "values.size() " << values.size() << " != rhs.values.size() " << rhs.values.size() << std::endl;
        }
    }
    void print(void){
        std::cout << "n_rows = " << n_rows << std::endl;
        std::cout << "n_cols = " << n_cols << std::endl;
        std::cout << "nnz = " << nnz << std::endl;
        std::cout << "is_sorted = " << is_sorted << std::endl;
        std::cout << "is_symmetric = " << is_symmetric << std::endl;

        std::cout << "I = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << I[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "J = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << J[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "values = [";
        for(int i = 0; i < nnz; ++i){
            std::cout << values[i] << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // COOMtxData operator+(COOMtxData &rhs)
    // {
    //     COOMtxData sum_coo_mtx;
    //     // Dimension stays constant
    //     sum_coo_mtx.n_rows = n_rows;
    //     sum_coo_mtx.n_cols = n_cols;

    //     // add nnz
    //     sum_coo_mtx.nnz = nnz + rhs.nnz;

    //     // NOTE: not sorted anymore?
    //     sum_coo_mtx.is_sorted = false;
    //     sum_coo_mtx.is_symmetric = false;

    //     // NOTE: how to best arrange? Concatenate for now
    //     sum_coo_mtx.I.reserve( I.size() + rhs.I.size() ); // preallocate memory
    //     sum_coo_mtx.I.insert( sum_coo_mtx.I.end(), I.begin(), I.end() );
    //     sum_coo_mtx.I.insert( sum_coo_mtx.I.end(), rhs.I.begin(), rhs.I.end() );

    //     sum_coo_mtx.J.reserve( J.size() + rhs.J.size() ); // preallocate memory
    //     sum_coo_mtx.J.insert( sum_coo_mtx.J.end(), J.begin(), J.end() );
    //     sum_coo_mtx.J.insert( sum_coo_mtx.J.end(), rhs.J.begin(), rhs.J.end() );

    //     sum_coo_mtx.values.reserve( values.size() + rhs.values.size() ); // preallocate memory
    //     sum_coo_mtx.values.insert( sum_coo_mtx.values.end(), values.begin(), values.end() );
    //     sum_coo_mtx.values.insert( sum_coo_mtx.values.end(), rhs.values.begin(), rhs.values.end() );

    //     return sum_coo_mtx;
    // }
};

// TODO
// struct CRSMtxData
// {

// };
#endif /*STRUCTS_H*/