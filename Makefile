# Options: gcc, icc, icx, nvcc
COMPILER=nvcc

# Only applicable for gpu builds
GPGPU_ARCH=a40
THREADS_PER_BLOCK=32
BLOCKS_PER_GRID=256

DEBUG_MODE = 0
DEBUG_MODE_FINE = 0
USE_MPI = 0
USE_LIKWID = 0
USE_EIGEN = 0
USE_GPROF = 0

USE_USPMV = 0
CHUNK_SIZE = 8
SIGMA = 1

# TODO
# USE_METIS = 1
# USE_SPMP = 0


# compiler options
ifeq ($(COMPILER),gcc)
  CXX       = g++
  OPT_LEVEL = -O3
  OPT_ARCH  = -march=native
  CXXFLAGS += $(OPT_LEVEL) -Wall -fopenmp $(OPT_ARCH)
endif

ifeq ($(COMPILER),icc)
  CXX       = icpc
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -xhost
  CXXFLAGS += $(OPT_LEVEL) -Wall -fopenmp $(OPT_ARCH)
endif

ifeq ($(COMPILER),icx)
  CXX       = icpx
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -xhost
  AVX512_fix= #-Xclang -target-feature -Xclang +prefer-no-gather -xCORE-AVX512 -qopt-zmm-usage=high

  CXXFLAGS += $(OPT_LEVEL) -Wall -fopenmp $(AVX512_fix) $(OPT_ARCH)
endif

ifeq ($(COMPILER),nvcc)
  CXX       = nvcc
  MPICXX     = # TODO
  OPT_LEVEL = -O3
  OPT_HOST_ARCH  = #-tp=native how to optimizer host code for particular arch?
  OPT_DEVICE_ARCH  = -gencode arch=compute_86,code=sm_86 # Assuming A40 card
  HOST_COMPILER_FLAGS= -Xcompiler -Wall 

  CXXFLAGS += $(OPT_LEVEL) $(HOST_COMPILER_FLAGS) $(OPT_HOST_ARCH) $(OPT_DEVICE_ARCH)

ifeq ($(GPGPU_ARCH),a40)
	GPGPU_ARCH_FLAGS = -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp
endif
endif

ifeq ($(DEBUG_MODE),1)
  DEBUGFLAGS += -g -DDEBUG_MODE
endif

ifeq ($(USE_USPMV),1)
  CHUNK_SIZE = 2
  SIGMA = 2
  VECTOR_LENGTH = 4 # Assuming AVX instructions
  CXXFLAGS  += -DUSE_USPMV -DCHUNK_SIZE=$(CHUNK_SIZE) -DSIGMA=$(SIGMA) -DVECTOR_LENGTH=$(VECTOR_LENGTH)
  ifeq ($(COMPILER),nvcc)
    $(error CUDA with USpMV not yet supported)
  endif
endif

ifeq ($(USE_LIKWID),1)
  # !!! include your own file paths !!! (I'm just loading module, which comes with file paths)
  # LIKWID_INC =
  # LIKWID_LIB = 
  ifeq ($(LIKWID_INC),)
    $(error USE_LIKWID selected, but no include path given in LIKWID_INC)
  endif
  ifeq ($(LIKWID_LIB),)
    $(error USE_LIKWID selected, but no library path given in LIKWID_LIB)
  endif
  CXXFLAGS  += -DUSE_LIKWID -DLIKWID_PERFMON $(LIKWID_INC) $(LIKWID_LIB) -llikwid
endif

# Header-only library
ifeq ($(USE_EIGEN),1)
  # !!! include your own file paths !!! (I'm just loading module, which comes with file paths)
  # EIGEN_INC =
  ifeq ($(EIGEN_ROOT),)
    $(error USE_EIGEN selected, but no include path given in EIGEN_ROOT)
  endif
  CXXFLAGS  += -DUSE_EIGEN -I$(EIGEN_ROOT)/include/eigen3/
endif

ifeq ($(USE_GPROF),1)
  PROFFLAGS  += -pg -fno-inline 
endif

iterative_solvers: main.o utility_funcs.o io_funcs.o kernels.o mmio.o solvers.o
ifeq ($(COMPILER),nvcc)
	nvcc main.o utility_funcs.o io_funcs.o kernels.o mmio.o solvers.o $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o iterative_solvers_gpu
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) utility_funcs.o io_funcs.o kernels.o mmio.o solvers.o main.o -o iterative_solvers_cpu
endif

# main only depends on funcs, mmio, and structs header, not kernels
main.o: main.cpp utility_funcs.hpp io_funcs.hpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu -c main.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o main.o
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) -c main.cpp -o main.o
endif
	
solvers.o: solvers.cpp solvers.hpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu -c solvers.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o solvers.o
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) -c solvers.cpp -o solvers.o
endif

# funcs depends on kernels
utility_funcs.o: utility_funcs.cpp utility_funcs.hpp kernels.o
ifeq ($(COMPILER),nvcc)
	nvcc -x cu -c utility_funcs.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o utility_funcs.o
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) -c utility_funcs.cpp -o utility_funcs.o
endif

# funcs depends on kernels
io_funcs.o: io_funcs.cpp io_funcs.hpp utility_funcs.hpp mmio.o
ifeq ($(COMPILER),nvcc)
	nvcc -x cu -c io_funcs.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o io_funcs.o
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) -c io_funcs.cpp -o io_funcs.o
endif

# only depends on "kernels" src and header, and structs header
kernels.o: kernels.cpp kernels.hpp structs.hpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu -c kernels.cpp -o kernels.o
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) -c kernels.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o kernels.o
endif

# only depends on "mmio" src and header
mmio.o: mmio.cpp mmio.h
ifeq ($(COMPILER),nvcc)
	nvcc -x cu -c mmio.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o mmio.o
else
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(PROFFLAGS) -c mmio.cpp -o mmio.o
  endif

#################### Test Suite ####################
TEST_INC_DIR = /home/danel/iterative_solvers/splitting_type_solvers

tests: test_suite/catch.o test_suite/mtx_tests.o test_suite/other_tests.o test_suite/test_data.o test_suite/catch.hpp 
	$(CXX) $(CXXFLAGS) -I$(TEST_INC_DIR) test_suite/mtx_tests.o test_suite/other_tests.o test_suite/test_data.o test_suite/catch.o funcs.o mmio.o kernels.o -o test_suite/tests
	# -rm *.o
	# -rm test_suite/*.o

test_suite/catch.o: test_suite/catch.cpp test_suite/catch.hpp
	$(CXX) $(CXXFLAGS) -I$(TEST_INC_DIR) -c test_suite/catch.cpp -o test_suite/catch.o

# Also need funcs.o here, since we are testing the function implementations
test_suite/mtx_tests.o: test_suite/mtx_tests.cpp test_suite/catch.hpp test_suite/test_data.hpp funcs.o
	$(CXX) $(CXXFLAGS) -I$(TEST_INC_DIR) -c test_suite/mtx_tests.cpp -o test_suite/mtx_tests.o

test_suite/other_tests.o: test_suite/other_tests.cpp test_suite/catch.hpp test_suite/test_data.hpp funcs.o
	$(CXX) $(CXXFLAGS) -I$(TEST_INC_DIR) -c test_suite/other_tests.cpp -o test_suite/other_tests.o

# test_suite/other_tests.o: test_suite/other_tests.cpp test_suite/catch.hpp test_suite/test_data.hpp funcs.o
# 	$(CXX) -I$(TEST_INC_DIR) -c test_suite/other_tests.cpp -o test_suite/other_tests.o

test_suite/test_data.o: test_suite/test_data.cpp
	$(CXX) $(CXXFLAGS) -I$(TEST_INC_DIR) -c test_suite/test_data.cpp -o test_suite/test_data.o

# funcs.o: funcs.cpp funcs.hpp kernels.o structs.hpp kernels.hpp mmio.o
# 	$(CXX) -c funcs.cpp structs.hpp kernels.cpp kernels.hpp -o funcs.o

####################################################

# #################### Profiling ####################
# profs: main.o funcs.o kernels.o mmio.o
# 	$(CXX) -pg -o main_gprof main.o funcs.o kernels.o mmio.o
# 	-rm *.o
	
# # main only depends on funcs, mmio, and structs header, not kernels
# main.o: main.cpp funcs.hpp structs.hpp
# 	$(CXX) -pg -c main.cpp -o main.o
	
# # funcs depends on kernels
# funcs.o: funcs.cpp funcs.hpp kernels.o structs.hpp kernels.hpp mmio.o
# 	$(CXX) -pg -c funcs.cpp -o funcs.o

# # only depends on "kernels" src and header, and structs header
# kernels.o: kernels.cpp kernels.hpp structs.hpp
# 	$(CXX) -pg -c kernels.cpp -o kernels.o

# # only depends on "mmio" src and header
# mmio.o: mmio.cpp mmio.h
# 	$(CXX) -pg -c mmio.cpp -o mmio.o

# ####################################################

clean:
	-rm *.o
	-rm test_suite/*.o
