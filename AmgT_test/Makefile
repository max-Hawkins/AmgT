# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


# mpicc -o test_new test_new.o -g -Wall -DHYPRE_EXAMPLE_USING_CUDA -DH_TIME -L/u/mhawki
# ns/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src/local_install/lib -lHYPRE -lm -L/sw/spack/deltas11-2023-03/apps/linux
# -rhel8-zen3/nvhpc-24.1/cuda-12.3.0-ser66n7/lib64 -lcudart -lcublas -lcusparse -lcurand -lstdc++  -lstdc++


########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
F77       = mpif77
CXX       = nvc++
F90       = mpifort
# CUDA_HOME = /usr/local/cuda-12.0
#XL_DIR=$(dir $(shell which xlc))..
# ########################################################################
# # CUDA
# ########################################################################
   CUDA_INCL = -I${CUDA_HOME}/include
   CUDA_LIBS = -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusparse -lcurand -lstdc++ -lnvidia-ml #-L$(XL_DIR)/xlC/16.1.1/lib -libmc++
#    CUDA_ARCH = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=sm_80
   NVCC_LDFLAGS = -ccbin=${CXX} ${CUDA_ARCH}
   COPTS_CUDA = -DHYPRE_EXAMPLE_USING_CUDA -DH_TIME
   FOPTS_CUDA = -DHYPRE_EXAMPLE_USING_CUDA -qsuppress=cmpmsg
#    HYPRE_DIR = /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src/hypre
   echo "HYPRE_DIR: ${HYPRE_DIR}"



########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall $(COPTS_CUDA)
CINCLUDES = -I$(HYPRE_DIR)/include $(CUDA_INCL)
#CDEFS     = -DHYPRE_EXVIS
CDEFS     =
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)
FOPTS     = -g $(FOPTS_CUDA)
FINCLUDES = $(CINCLUDES)
FFLAGS    = $(FOPTS) $(FINCLUDES)
CXXOPTS   = $(COPTS) -Wno-deprecated
CXXINCLUDES = $(CINCLUDES) -I..
CXXDEFS   = $(CDEFS)
IFLAGS_BXX =
CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)
IF90FLAGS =
F90FLAGS = $(FFLAGS) $(IF90FLAGS)


LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm $(CUDA_LIBS)
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++
LFLAGS_B =\
 -L${HYPRE_DIR}/lib\
 -lbHYPREClient-C\
 -lbHYPREClient-CX\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl -ldl -lxml2
LFLAGS77 = $(LFLAGS)
LFLAGS90 =

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c .f .cxx .f90

.c.o:
	$(CXX) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
.cxx.o:
	$(CXX) $(CXXFLAGS) -c $<

########################################################################
# List of all programs to be compiled
########################################################################
ALLPROGS = test

all: $(ALLPROGS)

default: all

########################################################################
# Example 1
########################################################################
test: test.o
	$(CXX) -o $@ $^ $(LFLAGS)
test_new: test_new.o
	$(CC) -o $@ $^ $(LFLAGS)
########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o) $(ALLPROGS) test_new.o test_new
cleanout:
	rm -rf IJ.out.*
distclean: clean
	rm -f $(ALLPROGS) $(ALLPROGS:=*~)
