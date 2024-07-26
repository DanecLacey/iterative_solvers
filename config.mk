### Compilers ###
# Options: gcc, icc, icx, nvcc
COMPILER=icx
# Only applicable for gpu builds
# Options: a40, a100
GPGPU_ARCH=a100
THREADS_PER_BLOCK=32
BLOCKS_PER_GRID=256

### Solver Parameters ###
# [int]
MAX_ITERS=20000
# [float]
TOL=1e-12 
 # [int]
GMRES_RESTART_LEN=110
 # ["double"/"float"]
PRECISION=double

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
USE_LIKWID = 1
# LIKWID_INC =
# LIKWID_LIB = 

# [1/0]
USE_EIGEN = 0
# EIGEN_INC =

# [1/0]
USE_SCAMAC = 0
# SCAMAC_INC = -I/home/hpc/k107ce/k107ce17/linking_it_solve/SCAMAC/build/scamac/include/
# SCAMAC_LIB = /home/hpc/k107ce/k107ce17/linking_it_solve/SCAMAC/build/library/libscamac.a

# NOTE: We assume USpMV is in the same directory
# [1/0]
USE_USPMV = 1
# [1/0]
USE_AP = 1
# [float]
AP_THRESHOLD = 0.11316002119063188
# [int]
CHUNK_SIZE = 1
# [int]
SIGMA = 1
# [int]
VECTOR_LENGTH = 8
