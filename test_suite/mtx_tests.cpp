#include "catch.hpp"
#include "structs.hpp"
#include "funcs.hpp"
#include "test_data.hpp"

TEST_CASE("Validate COO matrix splitting function 'split_upper_lower_diagonal' ", "[require]"){
    SECTION("Test A = L+D+U splitting with COO Matrix M1"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M1, L_coo_mtx_M1;
        std::vector<double> D_inv_coo_vec_M1;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M1.reserve(M1.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M1, &U_coo_mtx_M1, &L_coo_mtx_M1, &D_inv_coo_vec_M1);

        // std::cout << "U_coo_mtx_M1.nnz = " << U_coo_mtx_M1.nnz << std::endl;
        // std::cout << "U_coo_mtx_M1.n_rows = " << U_coo_mtx_M1.n_rows << std::endl;
        // std::cout << "U_coo_mtx_M1.n_cols = " << U_coo_mtx_M1.n_cols << std::endl;
        // std::cout << "U_coo_mtx_M1.I[0] = " << U_coo_mtx_M1.I[0] << std::endl;
        // std::cout << "U_coo_mtx_M1.J[0] = " << U_coo_mtx_M1.J[0] << std::endl;
        // std::cout << "U_coo_mtx_M1.values[0] = " << U_coo_mtx_M1.values[0] << std::endl;

        // // Test resulting COO mtx structures
        // std::cout << U_coo_mtx_M1.nnz << " == " << expected_U_coo_mtx_M1.nnz << "?" << std::endl; // why is nnz 0?
        // for(int i = 0; i < expected_U_coo_mtx_M1.nnz; ++i){
        //     std::cout << "row: " << U_coo_mtx_M1.I[i] << " == " << expected_U_coo_mtx_M1.I[i] << "?" << std::endl;
        //     std::cout << "col: " << U_coo_mtx_M1.J[i] << " == " << expected_U_coo_mtx_M1.J[i] << "?" << std::endl;
        //     std::cout << "val: " << U_coo_mtx_M1.values[i] << " == " << expected_U_coo_mtx_M1.values[i] << "?" << std::endl;
        // }
        // for(int i = 0; i < expected_D_inv_coo_vec_M1.size(); ++i){
        //     std::cout << "val: " << D_inv_coo_vec_M1[i] << " == " << expected_D_inv_coo_vec_M1[i] << "?" << std::endl;
        // }
        REQUIRE((U_coo_mtx_M1 == expected_U_coo_mtx_M1));
        REQUIRE((L_coo_mtx_M1 == expected_L_coo_mtx_M1));
        REQUIRE(std::equal(D_inv_coo_vec_M1.begin(),D_inv_coo_vec_M1.end(),expected_D_inv_coo_vec_M1.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M2"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M2, L_coo_mtx_M2;
        std::vector<double> D_inv_coo_vec_M2;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M2.reserve(M2.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M2, &U_coo_mtx_M2, &L_coo_mtx_M2, &D_inv_coo_vec_M2);

        REQUIRE((U_coo_mtx_M2 == expected_U_coo_mtx_M2));
        REQUIRE((L_coo_mtx_M2 == expected_L_coo_mtx_M2));
        REQUIRE(std::equal(D_inv_coo_vec_M2.begin(), D_inv_coo_vec_M2.end(), expected_D_inv_coo_vec_M2.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M3"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M3, L_coo_mtx_M3;
        std::vector<double> D_inv_coo_vec_M3;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M3.reserve(M3.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M3, &U_coo_mtx_M3, &L_coo_mtx_M3, &D_inv_coo_vec_M3);

        REQUIRE((U_coo_mtx_M3 == expected_U_coo_mtx_M3));
        REQUIRE((L_coo_mtx_M3 == expected_L_coo_mtx_M3));
        REQUIRE(std::equal(D_inv_coo_vec_M3.begin(), D_inv_coo_vec_M3.end(), expected_D_inv_coo_vec_M3.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M4"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M4, L_coo_mtx_M4;
        std::vector<double> D_inv_coo_vec_M4;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M4.reserve(M4.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M4, &U_coo_mtx_M4, &L_coo_mtx_M4, &D_inv_coo_vec_M4);

        REQUIRE((U_coo_mtx_M4 == expected_U_coo_mtx_M4));
        REQUIRE((L_coo_mtx_M4 == expected_L_coo_mtx_M4));
        REQUIRE(std::equal(D_inv_coo_vec_M4.begin(), D_inv_coo_vec_M4.end(), expected_D_inv_coo_vec_M4.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M5"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M5, L_coo_mtx_M5;
        std::vector<double> D_inv_coo_vec_M5;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M5.reserve(M5.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M5, &U_coo_mtx_M5, &L_coo_mtx_M5, &D_inv_coo_vec_M5);

        REQUIRE((U_coo_mtx_M5 == expected_U_coo_mtx_M5));
        REQUIRE((L_coo_mtx_M5 == expected_L_coo_mtx_M5));
        REQUIRE(std::equal(D_inv_coo_vec_M5.begin(), D_inv_coo_vec_M5.end(), expected_D_inv_coo_vec_M5.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M6"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M6, L_coo_mtx_M6;
        std::vector<double> D_inv_coo_vec_M6;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M6.reserve(M6.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M6, &U_coo_mtx_M6, &L_coo_mtx_M6, &D_inv_coo_vec_M6);

        REQUIRE((U_coo_mtx_M6 == expected_U_coo_mtx_M6));
        REQUIRE((L_coo_mtx_M6 == expected_L_coo_mtx_M6));
        REQUIRE(std::equal(D_inv_coo_vec_M6.begin(), D_inv_coo_vec_M6.end(), expected_D_inv_coo_vec_M6.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M7"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M7, L_coo_mtx_M7;
        std::vector<double> D_inv_coo_vec_M7;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M7.reserve(M7.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M7, &U_coo_mtx_M7, &L_coo_mtx_M7, &D_inv_coo_vec_M7);

        REQUIRE((U_coo_mtx_M7 == expected_U_coo_mtx_M7));
        REQUIRE((L_coo_mtx_M7 == expected_L_coo_mtx_M7));
        REQUIRE(std::equal(D_inv_coo_vec_M7.begin(), D_inv_coo_vec_M7.end(), expected_D_inv_coo_vec_M7.begin()));
    }
    SECTION("Test A = L+D+U splitting with COO Matrix M8"){

        // Declare structs to be tested
        COOMtxData U_coo_mtx_M8, L_coo_mtx_M8;
        std::vector<double> D_inv_coo_vec_M8;

        // Can allocate space for diagonal_inv, since we know its size
        D_inv_coo_vec_M8.reserve(M8.n_rows);
        
        // Populate structs using function we are testing
        split_upper_lower_diagonal(&M8, &U_coo_mtx_M8, &L_coo_mtx_M8, &D_inv_coo_vec_M8);

        REQUIRE((U_coo_mtx_M8 == expected_U_coo_mtx_M8));
        REQUIRE((L_coo_mtx_M8 == expected_L_coo_mtx_M8));
        REQUIRE(std::equal(D_inv_coo_vec_M8.begin(), D_inv_coo_vec_M8.end(), expected_D_inv_coo_vec_M8.begin()));
    }
}

TEST_CASE("Validate COO matrix sum operator", "[require]"){
    SECTION("Test M1 + M1"){
        COOMtxData M1_sum_M1;

        sum_matrices(&M1_sum_M1, &M1, &M1);

        REQUIRE((expected_M1_sum_M1 == M1_sum_M1));
    }
    SECTION("Test M1 + M3"){
        COOMtxData M1_sum_M3;

        sum_matrices(&M1_sum_M3, &M1, &M3);

        // expected_M1_sum_M3.print();
        // printf("\n");
        // M1_sum_M3.print();

        // // expected_M1_sum_M3^M1_sum_M3;
        // exit(1);

        REQUIRE((expected_M1_sum_M3 == M1_sum_M3));
    }
    SECTION("Test M3 + M1"){
        COOMtxData M3_sum_M1;

        sum_matrices(&M3_sum_M1, &M3, &M1);

        // Using same expected matrix for assertion
        REQUIRE((expected_M1_sum_M3 == M3_sum_M1));
    }
    SECTION("Test M2 + M4"){
        COOMtxData M2_sum_M4;

        sum_matrices(&M2_sum_M4, &M2, &M4);

        REQUIRE((expected_M2_sum_M4 == M2_sum_M4));
    }
    SECTION("Test M4 + M2"){
        COOMtxData M4_sum_M2;

        sum_matrices(&M4_sum_M2, &M4, &M2);

        REQUIRE((expected_M2_sum_M4 == M4_sum_M2));
    }
    SECTION("Test M5 + M6"){
        COOMtxData M5_sum_M6;

        sum_matrices(&M5_sum_M6, &M5, &M6);

        REQUIRE((expected_M5_sum_M6 == M5_sum_M6));
    }
    SECTION("Test M6 + M5"){
        COOMtxData M6_sum_M5;

        sum_matrices(&M6_sum_M5, &M6, &M5);

        REQUIRE((expected_M5_sum_M6 == M6_sum_M5));
    }
    SECTION("Test M5 + M7"){
        COOMtxData M5_sum_M7;

        sum_matrices(&M5_sum_M7, &M5, &M7);

        REQUIRE((expected_M5_sum_M7 == M5_sum_M7));
    }
    SECTION("Test M7 + M5"){
        COOMtxData M7_sum_M5;

        sum_matrices(&M7_sum_M5, &M7, &M5);

        REQUIRE((expected_M5_sum_M7 == M7_sum_M5));
    }
    SECTION("Test M7 + M8"){
        COOMtxData M7_sum_M8;

        sum_matrices(&M7_sum_M8, &M7, &M8);

        REQUIRE((expected_M7_sum_M8 == M7_sum_M8));
    }
    SECTION("Test M8 + M7"){
        COOMtxData M8_sum_M7;

        sum_matrices(&M8_sum_M7, &M8, &M7);

        REQUIRE((expected_M7_sum_M8 == M8_sum_M7));
    }
}

TEST_CASE("Validate infty_mtx_coo_norm", "[require]"){
    SECTION("Test Mn"){
        REQUIRE(infty_mtx_coo_norm(&Mn) == 36.2);
    }
    SECTION("Test Mz"){
        REQUIRE(infty_mtx_coo_norm(&Mz) == 0.);
    }
    SECTION("Test M0"){
        REQUIRE(infty_mtx_coo_norm(&M0) == 3.);
    }
    SECTION("Test M1"){
        REQUIRE(infty_mtx_coo_norm(&M1) == 96.);
    }
    SECTION("Test M2"){
        REQUIRE(infty_mtx_coo_norm(&M2) == 85.);
    }
    SECTION("Test M3"){
        REQUIRE(infty_mtx_coo_norm(&M3) == 31.);
    }
}

TEST_CASE("Validate mtx_spmv_coo", "[require]"){
    SECTION("Test mtx_spmv_coo M1 * V1"){
        std::vector<double> spmv_vec(V1.size());

        mtx_spmv_coo(&spmv_vec, &M1, &V1);

        REQUIRE((spmv_vec == expected_M1_mult_V1));
    }
    SECTION("Test mtx_spmv_coo M1 * V4"){
        std::vector<double> spmv_vec(V4.size());

        mtx_spmv_coo(&spmv_vec, &M1, &V4);

        REQUIRE((spmv_vec == expected_M1_mult_V4));
    }
    SECTION("Test mtx_spmv_coo Mz * V5"){
        std::vector<double> spmv_vec(V5.size());

        mtx_spmv_coo(&spmv_vec, &Mz, &V5);

        REQUIRE((spmv_vec == expected_Mz_mult_V5));
    }
    SECTION("Test mtx_spmv_coo Mn * V1"){
        std::vector<double> spmv_vec(V1.size());

        mtx_spmv_coo(&spmv_vec, &Mn, &V1);

        REQUIRE((spmv_vec == expected_Mn_mult_V1));
    }
}
