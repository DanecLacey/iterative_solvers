CCX=g++
CFLAGS=-I.

# CXX += -pg

main: main.o funcs.o kernels.o mmio.o
	$(CCX) -o main main.o funcs.o kernels.o mmio.o
	-rm *.o
	
# main only depends on funcs, mmio, and structs header, not kernels
main.o: main.cpp funcs.hpp structs.hpp
	$(CCX) -c main.cpp -o main.o
	
# funcs depends on kernels
funcs.o: funcs.cpp funcs.hpp kernels.o structs.hpp kernels.hpp mmio.o
	$(CCX) -c funcs.cpp -o funcs.o

# only depends on "kernels" src and header, and structs header
kernels.o: kernels.cpp kernels.hpp structs.hpp
	$(CCX) -c kernels.cpp -o kernels.o

# only depends on "mmio" src and header
mmio.o: mmio.cpp mmio.h
	$(CCX) -c mmio.cpp -o mmio.o

#################### Test Suite ####################
TEST_INC_DIR = /home/danel/iterative_solvers/splitting_type_solvers

tests: test_suite/catch.o test_suite/mtx_tests.o test_suite/other_tests.o test_suite/test_data.o test_suite/catch.hpp 
	$(CCX) -I$(TEST_INC_DIR) test_suite/mtx_tests.o test_suite/other_tests.o test_suite/test_data.o test_suite/catch.o funcs.o mmio.o kernels.o -o test_suite/tests
	# -rm *.o
	# -rm test_suite/*.o

test_suite/catch.o: test_suite/catch.cpp test_suite/catch.hpp
	$(CCX) -I$(TEST_INC_DIR) -c test_suite/catch.cpp -o test_suite/catch.o

# Also need funcs.o here, since we are testing the function implementations
test_suite/mtx_tests.o: test_suite/mtx_tests.cpp test_suite/catch.hpp test_suite/test_data.hpp funcs.o
	$(CCX) -I$(TEST_INC_DIR) -c test_suite/mtx_tests.cpp -o test_suite/mtx_tests.o

test_suite/other_tests.o: test_suite/other_tests.cpp test_suite/catch.hpp test_suite/test_data.hpp funcs.o
	$(CCX) -I$(TEST_INC_DIR) -c test_suite/other_tests.cpp -o test_suite/other_tests.o

# test_suite/other_tests.o: test_suite/other_tests.cpp test_suite/catch.hpp test_suite/test_data.hpp funcs.o
# 	$(CCX) -I$(TEST_INC_DIR) -c test_suite/other_tests.cpp -o test_suite/other_tests.o

test_suite/test_data.o: test_suite/test_data.cpp
	$(CCX) -I$(TEST_INC_DIR) -c test_suite/test_data.cpp -o test_suite/test_data.o

# funcs.o: funcs.cpp funcs.hpp kernels.o structs.hpp kernels.hpp mmio.o
# 	$(CCX) -c funcs.cpp structs.hpp kernels.cpp kernels.hpp -o funcs.o

####################################################

# #################### Profiling ####################
# profs: main.o funcs.o kernels.o mmio.o
# 	$(CCX) -pg -o main_gprof main.o funcs.o kernels.o mmio.o
# 	-rm *.o
	
# # main only depends on funcs, mmio, and structs header, not kernels
# main.o: main.cpp funcs.hpp structs.hpp
# 	$(CCX) -pg -c main.cpp -o main.o
	
# # funcs depends on kernels
# funcs.o: funcs.cpp funcs.hpp kernels.o structs.hpp kernels.hpp mmio.o
# 	$(CCX) -pg -c funcs.cpp -o funcs.o

# # only depends on "kernels" src and header, and structs header
# kernels.o: kernels.cpp kernels.hpp structs.hpp
# 	$(CCX) -pg -c kernels.cpp -o kernels.o

# # only depends on "mmio" src and header
# mmio.o: mmio.cpp mmio.h
# 	$(CCX) -pg -c mmio.cpp -o mmio.o

# ####################################################

clean:
	-rm *.o
	-rm test_suite/*.o
