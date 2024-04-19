#[gcc, nvcc]
COMPILER=gcc

#[a40]
GPGPU_ARCH=a40

# Don't change
MKL=-I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

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
	GPGPU_ARCH_FLAGS = -gencode arch=compute_86,code=sm_86
endif

CPU_FLAGS = -O3 -march=native -fopenmp

################################## SAXPY #######################################
saxpy: saxpy.cpp
ifeq ($(COMPILER),nvcc)
	nvcc -x cu saxpy.cpp $(GPGPU_ARCH_FLAGS) -o saxpy_gpu 
endif
ifeq ($(COMPILER),gcc)
	g++ saxpy.cpp $(CPU_FLAGS) -o saxpy_cpu
endif

################################## SPMV ####################################
spmv: spmv.o helpers/mmio.o helpers/io_funcs.o helpers/utilities.o 
ifeq ($(COMPILER),nvcc)
	nvcc -x cu spmv.o helpers/io_funcs.o helpers/utilities.o helpers/mmio.o $(GPGPU_ARCH_FLAGS) -o spmv_gpu
endif
ifeq ($(COMPILER),gcc)
	g++ $(MKL) spmv.o helpers/io_funcs.o helpers/utilities.o helpers/mmio.o $(CPU_FLAGS) -o spmv_cpu
endif
	rm *.o

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
	