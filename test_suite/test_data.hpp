#ifndef TESTDATA_H
#define TESTDATA_H

#include "structs.hpp"

//////////////////////////// Mn test data ////////////////////////////
extern COOMtxData Mn;

//////////////////////////// Mz test data ////////////////////////////
extern COOMtxData Mz;

//////////////////////////// M0 test data ////////////////////////////
extern COOMtxData M0;

//////////////////////////// M1 test data ////////////////////////////
extern COOMtxData M1;

extern COOMtxData expected_U_coo_mtx_M1;

extern COOMtxData expected_L_coo_mtx_M1;

extern std::vector<double> expected_D_inv_coo_vec_M1;

extern COOMtxData expected_M1_sum_M1;

//////////////////////////// M2 test data ////////////////////////////
// Explicit zero on diagonal
extern COOMtxData M2;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M2;
extern COOMtxData expected_L_coo_mtx_M2;
extern std::vector<double> expected_D_inv_coo_vec_M2;

//////////////////////////// M3 test data ////////////////////////////
// Explicit zero on corners
extern COOMtxData M3;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M3;
extern COOMtxData expected_L_coo_mtx_M3;
extern std::vector<double> expected_D_inv_coo_vec_M3;

// Operation should be commutative, or something
extern COOMtxData expected_M1_sum_M3;

//////////////////////////// M4 test data ////////////////////////////
// Fully dense, and symmetric
extern COOMtxData M4;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M4;

extern COOMtxData expected_L_coo_mtx_M4;
extern std::vector<double> expected_D_inv_coo_vec_M4;

extern COOMtxData expected_M2_sum_M4;
//////////////////////////// M5 test data ////////////////////////////
// larger, sparser matrix
extern COOMtxData M5;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M5;

extern COOMtxData expected_L_coo_mtx_M5;

extern std::vector<double> expected_D_inv_coo_vec_M5;

//////////////////////////// M6 test data ////////////////////////////
// larger, purely diagonal matrix
extern COOMtxData M6;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M6;

extern COOMtxData expected_L_coo_mtx_M6;

extern std::vector<double> expected_D_inv_coo_vec_M6;

extern COOMtxData expected_M5_sum_M6;

//////////////////////////// M7 test data ////////////////////////////
// larger, lower triangular matrix
extern COOMtxData M7;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M7;
extern COOMtxData expected_L_coo_mtx_M7;

extern std::vector<double> expected_D_inv_coo_vec_M7;

extern COOMtxData expected_M5_sum_M7;
//////////////////////////// M8 test data ////////////////////////////
// larger, upper triangular matrix
extern COOMtxData M8;

// Declare expected data
extern COOMtxData expected_U_coo_mtx_M8;

extern COOMtxData expected_L_coo_mtx_M8;

extern std::vector<double> expected_D_inv_coo_vec_M8;

extern COOMtxData expected_M7_sum_M8;

//////////////////////////// M9 test data ////////////////////////////
// matrix1.mtx
extern COOMtxData M9;

//////////////////////////// V1 test data ////////////////////////////
extern std::vector<double> V1;
extern std::vector<double> expected_V1_sum_V1;
extern std::vector<double> expected_V1_mult_V1;
extern std::vector<double> expected_V1_mult_V1_init;
extern std::vector<double> expected_M1_mult_V1;
extern std::vector<double> V2;
extern std::vector<double> V3;
extern std::vector<double> expected_V1_sum_V3;
extern std::vector<double> expected_V1_mult_V3;
extern std::vector<double> V4;
extern std::vector<double> V5;
extern std::vector<double> expected_V1_mult_V4;
extern std::vector<double> expected_V1_mult_V5;
extern std::vector<double> expected_M1_mult_V4;
extern std::vector<double> expected_Mz_mult_V5;
extern std::vector<double> expected_Mn_mult_V1;
extern std::vector<double> expected_recip_V1;
extern std::vector<double> expected_recip_V3;
extern std::vector<double> expected_recip_V5;
extern std::vector<double> expected_neg_inv_coo_vec_V1;
extern std::vector<double> expected_neg_inv_coo_vec_V3;
extern std::vector<double> expected_neg_inv_coo_vec_V5;
extern std::vector<double> V6;

#endif /*TESTDATA_H*/