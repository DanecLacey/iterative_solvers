### Compilers ###
# Options: gcc, icc, icx, nvcc
COMPILER=icx
# Only applicable for gpu builds
# Options: a40, a100
GPGPU_ARCH=a100
THREADS_PER_BLOCK=32
BLOCKS_PER_GRID=256

### Solver Parameters ###
MAX_ITERS=500
TOL=1e-12
GMRES_RESTART_LEN=110

### Debugging ###
DEBUG_MODE = 0
DEBUG_MODE_FINE = 0
OUTPUT_SPARSITY = 0
FINE_TIMERS = 0
USE_GPROF = 0

### External Libraries ###
USE_LIKWID = 0
# LIKWID_INC =
# LIKWID_LIB = 

USE_EIGEN = 0
# EIGEN_INC =

USE_SCAMAC = 0
# SCAMAC_INC = -I/home/hpc/k107ce/k107ce17/linking_it_solve/SCAMAC/build/scamac/include/
# SCAMAC_LIB = /home/hpc/k107ce/k107ce17/linking_it_solve/SCAMAC/build/library/libscamac.a

# NOTE: We assume USpMV is in the same directory
USE_USPMV = 1
USE_AP = 0
AP_THRESHOLD = 0
CHUNK_SIZE = 16
SIGMA = 1
VECTOR_LENGTH = 8
