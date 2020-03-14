# =======================================================================
# Please set the following variables
# =======================================================================

# Path to your gurobi installation. E.g., /home/username/gurobi801/linux64
GRB =
# Gurobi version number. E.g., 80
GRB_VER =


# =======================================================================
# DO NOT edit anything after this line unless you know what you are doing
# =======================================================================
CC = nvcc
EIGEN = ../eigen-3.3.7
INC = -I$(EIGEN) -I$(GRB)/include
LIB = -lm -lcublas_static -lcusolver_static -lcusparse_static -lculibos -L$(GRB)/lib -lgurobi$(GRB_VER)
SRC = $(wildcard *.cpp) $(wildcard *.cu)
HDR = $(wildcard *.h)
COMPILER_OPTIONS = -Wall -Ofast -fopenmp -DFAC_DIR=$(FAC_DIR)
CUDA_COMPILER_OPTIONS = -w -std=c++11 --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

MAKE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKE_DIR = $(dir $(MAKE_PATH))
FAC_DIR = $(MAKE_DIR)factorization

# From https://stackoverflow.com/a/10858332
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1.$(if $2, $2)))

$(call check_defined, \
	GRB GRB_VER, \
	Please edit the Makefile to set all required variables \
)

test: Timer.o util.o main.cu
	$(CC) $(CUDA_COMPILER_OPTIONS) $(INC) main.cu util.o Timer.o -o main $(LIB)
lib: $(SRC) $(HDR)
	$(CC) $(CUDA_COMPILER_OPTIONS) --compiler-options '-fPIC' --shared $(INC) $(SRC) -o cadmm.so $(LIB)
Timer.o: Timer.cpp Timer.h
	$(CC) $(CUDA_COMPILER_OPTIONS) $(INC) -dc Timer.cpp -o Timer.o
util.o: util.cpp util.h
	$(CC) $(CUDA_COMPILER_OPTIONS) $(INC) -dc util.cpp -o util.o
clean:
	rm -f *.o main *.so
