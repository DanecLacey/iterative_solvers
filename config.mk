# [gcc, icc, icx, nvcc]
COMPILER=icx
# [a40, a100]
GPGPU_ARCH=a100
# [int]
THREADS_PER_BLOCK=32
# [int]
BLOCKS_PER_GRID=256
# [int]
VECTOR_LENGTH = 8
# [c++14]
CPP_VERSION = c++23

### Solver Parameters ###
# [int]
MAX_ITERS=5000
# [float]
TOL=1e-14
 # [int]
GMRES_RESTART_LEN=100
 # [double/float/half]
WORKING_PRECISION=double


### Debugging ###
 # [1/0]
DEBUG_MODE = 0
 # [1/0]
DEBUG_MODE_FINE = 0
 # [1/0]
OUTPUT_SPARSITY = 0
 # [1/0]
FINE_TIMERS = 0
 # [1/0]
USE_GPROF = 0

### External Libraries ###
# [1/0]
USE_LIKWID = 0
# LIKWID_INC =
# LIKWID_LIB = 

# [1/0]
USE_EIGEN = 0
# EIGEN_INC =

# [1/0]
USE_SCAMAC = 0
SCAMAC_INC = -I/home/hpc/k107ce/k107ce17/linking_it_solve/SCAMAC/build/scamac/include/
SCAMAC_LIB = /home/hpc/k107ce/k107ce17/linking_it_solve/SCAMAC/build/library/libscamac.a

# NOTE: We assume USpMV is in the same directory
# [1/0]
USE_USPMV = 0
# [1/0]
USE_AP = 0
# ['"none"', '"ap[dp_sp]"', '"ap[dp_hp]"', '"ap[sp_hp]"', '"ap[dp_sp_hp]"']
AP_VALUE_TYPE = '"ap[dp_hp]"'
# [float]
AP_THRESHOLD_1 = 10000000000000000000.0
# AP_THRESHOLD_1 = 0.0
# [float]
AP_THRESHOLD_2 = 0.0
# [int]
CHUNK_SIZE = 1
# [int]
SIGMA = 1
