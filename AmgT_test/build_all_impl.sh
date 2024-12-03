#!/bin/bash

echo "Building for all AmgT implementations..."
HYPRE_HOME=/u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src
# AmgT_FP64, AmgT_Mixed, cuSPARSE

echo "\tAmgT_FP64..."

# cp /u/mhawkins/iter_methods_proj_amgT/AmgT/config_files/AmgT_FP64.h ${HYPRE_HOME}/seq_mv/seq_mv.h
# echo /u/mhawkins/iter_methods_proj_amgT/AmgT/config_files/AmgT_FP64.h

cd $HYPRE_HOME
make install -j 1
cd -
# export HYPRE_DIR=/u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src/hypre_amgt_fp64
export HYPRE_DIR=/u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src/hypre_cusparse
echo "Hypre_dir=${HYPRE_DIR}"

make clean && make test_new
mv test_new /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test/runnable_files/AmgT_FP64
# todo copy
echo ""

# echo "\tAmgT_Mixed..."
# export HYPRE_DIR=/u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src/hypre_amgt_mixed
# echo "Hypre_dir=${HYPRE_DIR}"
# make clean && make test_new
# mv test_new /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test/runnable_files/AmgT_Mixed
# echo ""

# echo "\tcuSparse..."
# export HYPRE_DIR=/u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_HYPRE/src/hypre_cusparse
# echo "Hypre_dir=${HYPRE_DIR}"
# make clean && make test_new
# mv test_new /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test/runnable_files/cuSPARSE
# echo ""