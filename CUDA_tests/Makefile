#[gcc, nvcc]
COMPILER=gcc

#[a40]
GPGPU_ARCH=a40
THREADS_PER_BLOCK=32
BLOCKS_PER_GRID=256

# Don't change
# MKL=-I${MKLROOT}/include -Xcompiler="-Wl,--no-as-needed" -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
MKL=-I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

# Validate inputs

ifneq ($(GPGPU_ARCH),a40)
$(info GPGPU_ARCH=$(GPGPU_ARCH))
$(error Please select a GPGPU architecture in: [a40])
endif

ifneq ($(COMPILER),gcc)
ifneq ($(COMPILER),nvcc)
$(info COMPILER=$(COMPILER))
$(error Please select a compiler in: [gcc, nvcc])
endif
endif

# Assign flags to variables

ifeq ($(GPGPU_ARCH),a40)
	GPGPU_ARCH_FLAGS = -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp
endif

CPU_FLAGS = -O3 -march=native -fopenmp

.DEFAULT_GOAL := all
.PHONY : all
all : saxpy vecadd matadd max_in_array spmv

################################## SAXPY #######################################
saxpy: saxpy.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu saxpy.cpp $(GPGPU_ARCH_FLAGS) -o saxpy_gpu 
endif
ifeq ($(COMPILER),gcc)
	g++ saxpy.cpp $(CPU_FLAGS) -o saxpy_cpu
endif

################################## VECADD #######################################
vecadd: vecadd.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu vecadd.cpp $(GPGPU_ARCH_FLAGS) -o vecadd_gpu 
endif
ifeq ($(COMPILER),gcc)
	g++ vecadd.cpp $(CPU_FLAGS) -o vecadd_cpu
endif

################################## MATADD #######################################
matadd: matadd.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu matadd.cpp $(GPGPU_ARCH_FLAGS) -o matadd_gpu 
endif
ifeq ($(COMPILER),gcc)
	g++ matadd.cpp $(CPU_FLAGS) -o matadd_cpu
endif

################################## MAX_IN_ARRAY #######################################
max_in_array: max_in_array.o helpers/max_in_array_kernels.o
ifeq ($(COMPILER),nvcc)
	nvcc max_in_array.o helpers/max_in_array_kernels.o $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -o max_in_array_gpu
endif
ifeq ($(COMPILER),gcc)
	g++ max_in_array.o helpers/max_in_array_kernels.o $(CPU_FLAGS) -o max_in_array_cpu
endif

max_in_array.o: max_in_array.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu max_in_array.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -c -o max_in_array.o
endif
ifeq ($(COMPILER),gcc)
	g++ max_in_array.cpp $(CPU_FLAGS) -c -o max_in_array.o
endif

helpers/max_in_array_kernels.o: helpers/max_in_array_kernels.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu helpers/max_in_array_kernels.cpp $(GPGPU_ARCH_FLAGS) -DBLOCKS_PER_GRID=$(BLOCKS_PER_GRID) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -c -o helpers/max_in_array_kernels.o
endif
ifeq ($(COMPILER),gcc)
	g++ helpers/max_in_array_kernels.cpp $(CPU_FLAGS) -c -o helpers/max_in_array_kernels.o
endif

################################## SPMV ####################################
spmv: spmv.o helpers/mmio.o helpers/io_funcs.o helpers/utilities.o 
ifeq ($(COMPILER),nvcc)
	nvcc $(MKL) spmv.o helpers/io_funcs.o helpers/utilities.o helpers/mmio.o $(GPGPU_ARCH_FLAGS) -o spmv_gpu
endif
ifeq ($(COMPILER),gcc)
	g++ $(MKL) spmv.o helpers/io_funcs.o helpers/utilities.o helpers/mmio.o $(CPU_FLAGS) -o spmv_cpu
endif

spmv.o: spmv.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu spmv.cpp -c $(GPGPU_ARCH_FLAGS) -o $@
endif
ifeq ($(COMPILER),gcc)
	g++ spmv.cpp -c $(CPU_FLAGS) -o $@
endif

helpers/utilities.o: helpers/utilities.cpp helpers/io_funcs.hpp helpers/structs.hpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu $(MKL) helpers/utilities.cpp -c $(GPGPU_ARCH_FLAGS) -o $@
endif
ifeq ($(COMPILER),gcc)
	g++ $(MKL) helpers/utilities.cpp -c $(CPU_FLAGS) -o $@
endif

helpers/io_funcs.o: helpers/io_funcs.cpp helpers/utilities.cpp helpers/mmio.h helpers/structs.hpp helpers/utilities.hpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu helpers/io_funcs.cpp -c $(GPGPU_ARCH_FLAGS) -o $@
endif
ifeq ($(COMPILER),gcc)
	g++ helpers/io_funcs.cpp -c $(CPU_FLAGS) -o $@
endif

helpers/mmio.o: helpers/mmio.cpp helpers/mmio.h
ifeq ($(COMPILER),nvcc)
	nvcc -x cu helpers/mmio.cpp -c $(GPGPU_ARCH_FLAGS) -o $@
endif
ifeq ($(COMPILER),gcc)
	g++ helpers/mmio.cpp -c $(CPU_FLAGS) -o $@
endif
#############################################################################

make clean:
	rm helpers/*.o
	rm *.o
	