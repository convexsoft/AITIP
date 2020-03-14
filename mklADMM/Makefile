# =======================================================================
# Please set the following variables
# =======================================================================

# Path to MKL root. E.g., /home/username/intel/mkl
MKLROOT =
# Path to your gurobi installation. E.g., /home/username/gurobi801/linux64
GRB =
# Gurobi version number. E.g., 80
GRB_VER =

# =======================================================================
# Please use the Intel® Math Kernel Library Link Line Advisor at 
# 	https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
# 	and set the following variables.
# When using the Link Line Advisor, be sure to set compiler to 
# 	"GNU C/C++" and interface layer to "32-bit integer"
# =======================================================================
MKL_LINK_LINE = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
MKL_COMPILER_OPTIONS = -m64 -I${MKLROOT}/include

# =======================================================================
# DO NOT edit anything after this line unless you know what you are doing
# =======================================================================
CXX = g++-7
EIGEN = ../eigen-3.3.7
IDIR = -I$(EIGEN) -I$(GRB)/include
LIBS = -L$(GRB)/lib -lgurobi$(GRB_VER) $(MKL_LINK_LINE)
CXXFLAGS = -Wall -Wno-unused-result -Wno-deprecated-declarations -Ofast -std=gnu++14 -fopenmp -march=native -DFAC_DIR=$(FAC_DIR) $(MKL_COMPILER_OPTIONS)

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
	MKLROOT GRB GRB_VER, \
	Please edit the Makefile to set all required variables \
)
$(call check_defined, \
	MKL_LINK_LINE MKL_COMPILER_OPTIONS, \
	Please visit https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor and set the variables in Makefile \
)

test: Timer.o util.o main.cpp csr.h dense.h
	$(CXX) $(IDIR) $(CXXFLAGS) main.cpp util.o Timer.o -o main $(LDIR) $(LIBS)
lib: Timer.cpp Timer.h util.cpp util.h main.cpp csr.h dense.h
	$(CXX) $(IDIR) $(CXXFLAGS) -fPIC -shared main.cpp util.cpp Timer.cpp -o cadmm.so $(LDIR) $(LIBS)
Timer.o: Timer.h Timer.cpp
	$(CXX) $(IDIR) $(CXXFLAGS) -c Timer.cpp -o Timer.o
util.o: util.h util.cpp
	$(CXX) $(IDIR) $(CXXFLAGS) -c util.cpp -o util.o
clean:
	rm -f *.o main *.so
