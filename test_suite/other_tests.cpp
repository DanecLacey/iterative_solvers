#include "catch.hpp"
#include "structs.hpp"
#include "funcs.hpp"
#include "test_data.hpp"

TEST_CASE("Validate calc_residual", "[require]"){
    SECTION("Test calc_residual with Mz, V1, and V1"){

        REQUIRE(calc_residual(&Mz, &V1, &V1) == 3);
    }
    SECTION("Test calc_residual with M1, V1, and V1"){

        REQUIRE(calc_residual(&M1, &V1, &V1) == 191);
    }
    SECTION("Test calc_residual with M1, V3, and V1"){

        REQUIRE(calc_residual(&M1, &V3, &V1) == 2897);
    }
    SECTION("Test calc_residual with M1, V1, and V3"){

        REQUIRE(calc_residual(&M1, &V1, &V3) == 394);
    }
}

TEST_CASE("Validate gen_neg_inv", "[require]"){
    SECTION("Test gen_neg_inv with V1"){
        std::vector<double> neg_inv_coo_vec(V1.size());
        std::vector<double> inv_coo_vec(V1.size());
        gen_neg_inv(&neg_inv_coo_vec, &inv_coo_vec, &V1);

        REQUIRE(neg_inv_coo_vec == expected_neg_inv_coo_vec_V1);
    }
    SECTION("Test gen_neg_inv with V3"){
        std::vector<double> neg_inv_coo_vec(V3.size());
        std::vector<double> inv_coo_vec(V3.size());
        gen_neg_inv(&neg_inv_coo_vec, &inv_coo_vec, &V3);

        REQUIRE(neg_inv_coo_vec == expected_neg_inv_coo_vec_V3);
    }
    SECTION("Test gen_neg_inv with V5"){
        std::vector<double> neg_inv_coo_vec(V5.size());
        std::vector<double> inv_coo_vec(V5.size());
        gen_neg_inv(&neg_inv_coo_vec, &inv_coo_vec, &V5);

        REQUIRE(neg_inv_coo_vec == expected_neg_inv_coo_vec_V5);
    }
}

TEST_CASE("Validate recip_elems", "[require]"){
    SECTION("Test recip_elems with V1"){
        std::vector<double> recip_elems_vec(V1.size());

        recip_elems(&recip_elems_vec, &V1);

        REQUIRE(recip_elems_vec == expected_recip_V1);
    }
    SECTION("Test recip_elems with V3"){
        std::vector<double> recip_elems_vec(V3.size());

        recip_elems(&recip_elems_vec, &V3);

        REQUIRE(recip_elems_vec == expected_recip_V3);
    }
    SECTION("Test recip_elems with V5"){
        std::vector<double> recip_elems_vec(V5.size());

        recip_elems(&recip_elems_vec, &V5);

        REQUIRE(recip_elems_vec == expected_recip_V5);
    }
}

TEST_CASE("Validate infty_vec_norm", "[require]"){
    SECTION("Test inifinity norm with V1"){
        REQUIRE(infty_vec_norm(&V1) == 3.);
    }
    SECTION("Test inifinity norm with V3"){
        REQUIRE(infty_vec_norm(&V3) == 200.);
    }
    SECTION("Test inifinity norm with V4"){
        REQUIRE(infty_vec_norm(&V4) == 0.);
    }
    SECTION("Test inifinity norm with V5"){
        REQUIRE(infty_vec_norm(&V5) == 1.);
    }
    SECTION("Test inifinity norm with V6"){
        REQUIRE(infty_vec_norm(&V6) == 3.);
    }
}

TEST_CASE("Validate sum_vectors", "[require]"){
    SECTION("Test sum_vectors V1 + V1"){
        std::vector<double> sum_vec(V1.size());

        sum_vectors(&sum_vec, &V1, &V1);

        REQUIRE((sum_vec == expected_V1_sum_V1));
    }
    SECTION("Test sum_vectors V1 + V3"){
        std::vector<double> sum_vec(V1.size());

        sum_vectors(&sum_vec, &V1, &V3);

        REQUIRE((sum_vec == expected_V1_sum_V3));
    }
    SECTION("Test sum_vectors V3 + V1"){
        std::vector<double> sum_vec(V1.size());

        sum_vectors(&sum_vec, &V3, &V1);

        REQUIRE((sum_vec == expected_V1_sum_V3));
    }
    SECTION("Test sum_vectors V4 + V5"){
        std::vector<double> sum_vec(V4.size());

        sum_vectors(&sum_vec, &V4, &V5);

        REQUIRE((sum_vec == V5));
    }
}

TEST_CASE("Validate vec_spmv_coo", "[require]"){
    SECTION("Test vec_spmv_coo V1 * V1, no initialization"){
        std::vector<double> mult_vec(V1.size());

        vec_spmv_coo(&mult_vec, &V1, &V1);

        // std::cout << "[";
        // for(int i = 0; i < expected_V1_mult_V1.size(); ++i){
        //     std::cout << expected_V1_mult_V1[i] << ", ";
        // }
        // std::cout << "]" << std::endl;

        // std::cout << "[";
        // for(int i = 0; i < expected_V1_mult_V1.size(); ++i){
        //     std::cout << mult_vec[i] << ", ";
        // }
        // std::cout << "]" << std::endl;

        REQUIRE((mult_vec == expected_V1_mult_V1));
    }

    SECTION("Test vec_spmv_coo V1 * V1, with initialization"){
        std::vector<double> mult_vec(V1.size(), 10);

        vec_spmv_coo(&mult_vec, &V1, &V1);

        REQUIRE((mult_vec == expected_V1_mult_V1_init));
    }

    SECTION("Test vec_spmv_coo V1 * V1, with initialization"){
        std::vector<double> mult_vec(V1.size(), 10);

        vec_spmv_coo(&mult_vec, &V1, &V1);

        REQUIRE((mult_vec == expected_V1_mult_V1_init));
    }
    SECTION("Test vec_spmv_coo V1 * V3"){
        std::vector<double> mult_vec(V1.size());

        vec_spmv_coo(&mult_vec, &V1, &V3);

        REQUIRE((mult_vec == expected_V1_mult_V3));
    }
    SECTION("Test vec_spmv_coo V3 * V1"){
        std::vector<double> mult_vec(V1.size());

        vec_spmv_coo(&mult_vec, &V3, &V1);

        REQUIRE((mult_vec == expected_V1_mult_V3));
    }
    SECTION("Test vec_spmv_coo V1 * V4"){
        std::vector<double> mult_vec(V1.size());

        vec_spmv_coo(&mult_vec, &V1, &V4);

        REQUIRE((mult_vec == V4));
    }
    SECTION("Test vec_spmv_coo V1 * V5"){
        std::vector<double> mult_vec(V1.size());

        vec_spmv_coo(&mult_vec, &V1, &V5);

        REQUIRE((mult_vec == V1));
    }
}