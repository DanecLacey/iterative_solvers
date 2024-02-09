#include "structs.hpp"

//////////////////////////// Mn test data ////////////////////////////
COOMtxData Mn {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {-5.,-10.,-11.,-4.,-9.,0.,-15.2,-2.,-19.} // values
};

//////////////////////////// Mz test data ////////////////////////////
COOMtxData Mz {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {0.,0.,0.,0.,0.,0.,0.,0.,0.} // values
};

//////////////////////////// M0 test data ////////////////////////////
COOMtxData M0 {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {1.,1.,1.,1.,1.,1.,1.,1.,1.} // values
};

//////////////////////////// M1 test data ////////////////////////////
COOMtxData M1 {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {11.,12.,13.,21.,22.,23.,31.,32.,33.} // values
};

COOMtxData expected_U_coo_mtx_M1 {
    3, // n_rows
    3, // n_cols
    3, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,1}, // I
    std::vector<int> {1,2,2}, // J
    std::vector<double> {12.,13.,23.} // values
};

COOMtxData expected_L_coo_mtx_M1 {
    3, // n_rows
    3, // n_cols
    3, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {1,2,2}, // I
    std::vector<int> {0,0,1}, // J
    std::vector<double> {21.,31.,32.} // values
};

std::vector<double> expected_D_inv_coo_vec_M1 {1/(double)11., 1/(double)22., 1/(double)33.};

COOMtxData expected_M1_sum_M1 {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {22.,24.,26.,42.,44.,46.,62.,64.,66.} // values
};

//////////////////////////// M2 test data ////////////////////////////
// Explicit zero on diagonal
COOMtxData M2 {
    4, // n_rows
    4, // n_cols
    7, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,1,1,2,3,3}, // I
    std::vector<int> {0,2,1,3,2,0,3}, // J
    std::vector<double> {11.,13.,0.,24.,33.,41.,44.} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M2 {
    4, // n_rows
    4, // n_cols
    2, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1}, // I
    std::vector<int> {2,3}, // J
    std::vector<double> {13.,24.} // values
};
COOMtxData expected_L_coo_mtx_M2 {
    4, // n_rows
    4, // n_cols
    1, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {3}, // I
    std::vector<int> {0}, // J
    std::vector<double> {41.} // values

};
std::vector<double> expected_D_inv_coo_vec_M2 {1/(double)11., 0, 1/(double)33., 1/(double)44.};

//////////////////////////// M3 test data ////////////////////////////
// Explicit zero on corners
COOMtxData M3 {
    3, // n_rows
    3, // n_cols
    5, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,1,2,2}, // I
    std::vector<int> {0,2,1,0,2}, // J
    std::vector<double> {0,13.,22.,31.,0} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M3 {
    3, // n_rows
    3, // n_cols
    1, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0}, // I
    std::vector<int> {2}, // J
    std::vector<double> {13.} // values
};
COOMtxData expected_L_coo_mtx_M3 {
    3, // n_rows
    3, // n_cols
    1, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {2}, // I
    std::vector<int> {0}, // J
    std::vector<double> {31.} // values

};
std::vector<double> expected_D_inv_coo_vec_M3 {0, 1/(double)22., 0};

// Operation should be commutative, or something
COOMtxData expected_M1_sum_M3 {
    3, // n_rows
    3, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2}, // I
    std::vector<int> {0,1,2,0,1,2,0,1,2}, // J
    std::vector<double> {11.,12.,26.,21.,44.,23.,62.,32.,33.} // values
};

//////////////////////////// M4 test data ////////////////////////////
// Fully dense, and symmetric
COOMtxData M4 {
    4, // n_rows
    4, // n_cols
    16, // nnz
    true, // is_sorted
    true, // is_symmetric
    std::vector<int> {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3}, // I
    std::vector<int> {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}, // J
    std::vector<double> {11.,12.,13.,14.,
                         12.,22.,23.,24.,
                         13.,23.,33.,34.,
                         14.,24.,34.,44.} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M4 {
    4, // n_rows
    4, // n_cols
    6, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,2}, // I
    std::vector<int> {1,2,3,2,3,3}, // J
    std::vector<double> {12.,13.,14.,23.,24.,34} // values
};

COOMtxData expected_L_coo_mtx_M4 {
    4, // n_rows
    4, // n_cols
    6, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {1,2,2,3,3,3}, // I
    std::vector<int> {0,0,1,0,1,2}, // J
    std::vector<double> {12.,13.,23.,14.,24.,34.} // values

};
std::vector<double> expected_D_inv_coo_vec_M4 {1/(double)11., 1/(double)22., 1/(double)33., 1/(double)44.};

COOMtxData expected_M2_sum_M4 {
    4, // n_rows
    4, // n_cols
    16, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3}, // I
    std::vector<int> {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}, // J
    std::vector<double> {22.,12.,26.,14.,
                         12.,22.,23.,48.,
                         13.,23.,66.,34.,
                         55.,24.,34.,88.} // values
};
//////////////////////////// M5 test data ////////////////////////////
// larger, sparser matrix
COOMtxData M5 {
    10, // n_rows
    10, // n_cols
    14, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,2,3,4,4,4,5,6,7,8,8,9,9}, // I
    std::vector<int> {0,1,2,3,4,6,7,5,6,7,0,8,8,9}, // J
    std::vector<double> {11.,22.,33.,44.,55.,57.,58.,66.,77.,88.,91.,99.,109.,1010.} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M5 {
    10, // n_rows
    10, // n_cols
    2, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {4,4}, // I
    std::vector<int> {6,7}, // J
    std::vector<double> {57.,58.} // values
};

COOMtxData expected_L_coo_mtx_M5 {
    10, // n_rows
    10, // n_cols
    2, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {8,9}, // I
    std::vector<int> {0,8}, // J
    std::vector<double> {91.,109.} // values

};

std::vector<double> expected_D_inv_coo_vec_M5 {
1/(double)11., 
1/(double)22., 
1/(double)33.,
1/(double)44., 
1/(double)55.,
1/(double)66., 
1/(double)77.,
1/(double)88.,
1/(double)99.
};

//////////////////////////// M6 test data ////////////////////////////
// larger, purely diagonal matrix
COOMtxData M6 {
    10, // n_rows
    10, // n_cols
    10, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,2,3,4,5,6,7,8,9}, // I
    std::vector<int> {0,1,2,3,4,5,6,7,8,9}, // J
    std::vector<double> {11.,22.,33.,44.,55.,66.,77.,88.,99.,1010.} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M6 {
    10, // n_rows
    10, // n_cols
    0, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {}, // I
    std::vector<int> {}, // J
    std::vector<double> {} // values
};

COOMtxData expected_L_coo_mtx_M6 {
    10, // n_rows
    10, // n_cols
    0, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {}, // I
    std::vector<int> {}, // J
    std::vector<double> {} // values

};

std::vector<double> expected_D_inv_coo_vec_M6 {
1/(double)11., 
1/(double)22., 
1/(double)33.,
1/(double)44., 
1/(double)55.,
1/(double)66., 
1/(double)77.,
1/(double)88.,
1/(double)99.
};

COOMtxData expected_M5_sum_M6 {
    10, // n_rows
    10, // n_cols
    14, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,2,3,4,4,4,5,6,7,8,8,9,9}, // I
    std::vector<int> {0,1,2,3,4,6,7,5,6,7,0,8,8,9}, // J
    std::vector<double> {22.,44.,66.,88.,110.,57.,58.,132.,154.,176.,91.,198.,109.,2020.} // values
};

//////////////////////////// M7 test data ////////////////////////////
// larger, lower triangular matrix
COOMtxData M7 {
    10, // n_rows
    10, // n_cols
    19, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9}, // I
    std::vector<int> {0,0,1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9}, // J
    std::vector<double> {11.,                                                  
                         21.,22.,
                         31.,    33.,
                         41.,        44.,
                         51.,            55.,
                         61.,                66.,
                         71.,                   77.,
                         81.,                      88.,
                         91.,                         99.,
                         101.,                           1010.} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M7 {
    10, // n_rows
    10, // n_cols
    0, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {}, // I
    std::vector<int> {}, // J
    std::vector<double> {} // values
};

COOMtxData expected_L_coo_mtx_M7 {
    10, // n_rows
    10, // n_cols
    9, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {1,2,3,4,5,6,7,8,9}, // I
    std::vector<int> {0,0,0,0,0,0,0,0,0}, // J
    std::vector<double> {21.,31.,41.,51.,61.,71.,81.,91.,101.} // values

};

std::vector<double> expected_D_inv_coo_vec_M7 {
1/(double)11., 
1/(double)22., 
1/(double)33.,
1/(double)44., 
1/(double)55.,
1/(double)66., 
1/(double)77.,
1/(double)88.,
1/(double)99.
};

COOMtxData expected_M5_sum_M7 {
    10, // n_rows
    10, // n_cols
    22, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,1,1,2,2,3,3,4,4,4,4,5,5,6,6,7,7,8,8,9,9,9}, // I
    std::vector<int> {0,0,1,0,2,0,3,0,4,6,7,0,5,0,6,0,7,0,8,0,8,9}, // J
    std::vector<double> {22.,21.,44.,31.,66.,41.,88.,51.,110.,57.,
        58.,61.,132.,71.,154.,81.,176.,182.,198.,101.,109., 2020.} // values
};
//////////////////////////// M8 test data ////////////////////////////
// larger, upper triangular matrix
COOMtxData M8 {
    10, // n_rows
    10, // n_cols
    17, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,2,2,3,4,5,6,6,6,6,7,8,9}, // I
    std::vector<int> {0,3,9,1,3,2,3,3,4,5,6,7,8,9,7,8,9}, // J
    std::vector<double> {11.,14.,110.,22.,24.,33.,34.,44.,55.,66.,77.,78.,79.,710.,88.,99.,1010.} // values
};

// Declare expected data
COOMtxData expected_U_coo_mtx_M8 {
    10, // n_rows
    10, // n_cols
    7, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,1,2,6,6,6}, // I
    std::vector<int> {3,9,3,3,7,8,9}, // J
    std::vector<double> {14.,110.,24.,34.,78.,79.,710} // values
};

COOMtxData expected_L_coo_mtx_M8 {
    10, // n_rows
    10, // n_cols
    0, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {}, // I
    std::vector<int> {}, // J
    std::vector<double> {} // values

};

std::vector<double> expected_D_inv_coo_vec_M8 {
1/(double)11., 
1/(double)22., 
1/(double)33.,
1/(double)44., 
1/(double)55.,
1/(double)66., 
1/(double)77.,
1/(double)88.,
1/(double)99.
};

COOMtxData expected_M7_sum_M8 {
    10, // n_rows
    10, // n_cols
    26, // nnz
    true, // is_sorted
    false, // is_symmetric
    std::vector<int> {0,0,0,1,1,1,2,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,8,8,9,9}, // I
    std::vector<int> {0,3,9,0,1,3,0,2,3,0,3,0,4,0,5,0,6,7,8,9,0,7,0,8,0,9}, // J
    std::vector<double> {22.,14.,110.,21.,44.,24.,31.,66.,34.,41.,88.,51.,
        110.,61.,132.,71.,154.,78.,79.,710.,81.,176.,91.,198.,101.,2020.} // values
};

// TODO: Organize
//////////////////////////// V1 test data ////////////////////////////
std::vector<double> V1 {1., 2., 3.};
std::vector<double> expected_recip_V1 {1/1., 1/2., 1/3.};
std::vector<double> expected_neg_inv_coo_vec_V1 {-1/1., -1/2., -1/3.};
std::vector<double> expected_V1_sum_V1 {2., 4., 6.};
std::vector<double> expected_V1_mult_V1 {1., 4., 9.};
std::vector<double> expected_V1_mult_V1_init {11., 14., 19.};
std::vector<double> expected_M1_mult_V1 {74., 134., 194.};

std::vector<double> V2 {};

std::vector<double> V3 {100., 200., -200.};
std::vector<double> expected_recip_V3 {1/100., 1/200., -1/200.};
std::vector<double> expected_neg_inv_coo_vec_V3 {-1/100., -1/200., 1/200.};
std::vector<double> expected_V1_sum_V3 {101., 202., -197.};
std::vector<double> expected_V1_mult_V3 {100., 400., -600.};

std::vector<double> V4 {0., 0., 0.};
// When the vector is 0
std::vector<double> expected_M1_mult_V4 {0., 0., 0.};
std::vector<double> V5 {1., 1., 1.};
std::vector<double> expected_recip_V5 {1., 1., 1.};
std::vector<double> expected_neg_inv_coo_vec_V5 {-1., -1., -1.};

std::vector<double> expected_V1_mult_V4 {1., 2., 3.};

// When the matrix is 0
std::vector<double> expected_Mz_mult_V5 {0., 0., 0.};
std::vector<double> expected_Mn_mult_V1 {-58., -22., -76.2};

std::vector<double> V6 {-1., -2., -3.};

