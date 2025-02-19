#!/bin/bash

# Parse input arguments
# num_rows, num_cols, num_nnz_per_block, exec_mode

# Check if the correct number of arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <num_rows> <num_cols> <num_nnz_per_block> <exec_mode> <num_trials>"
    echo "Example: ./run_bench.sh 16384 16384 16 TC 10000"
    exit 1
fi

# Assign input arguments to variables
num_rows=$1
num_cols=$2
num_nnz_per_block=$3
exec_mode=$4
num_trials=$5
printf "Running with %d rows, %d cols, %d nnz per block, %s mode, %d trials\n" $num_rows $num_cols $num_nnz_per_block $exec_mode $num_trials

# Determine the sparse matrix file name
sparse_matrix_file="/work/hdd/bdiy/mhawkins/sparse_efficiency/sparse_matrices/diy_nnz_per_4x4/sparse_matrix_rows_${num_rows}_cols_${num_cols}_nnz_per_4x4_${num_nnz_per_block}.mtx"

# Load python environmnt
source /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test/plot_venv/bin/activate

# If it doesn't exist, run the Python script with the provided arguments
if [ ! -f "$sparse_matrix_file" ]; then
    echo "no sparse matrix file found, running Python script to generate it..."
    cmd="python make_sparse_matrix.py $num_rows $num_cols $num_nnz_per_block"
    echo "Running command: $cmd"
    eval $cmd
fi

# Predetermine the output folder
output_folder="data/sparse_matrix_rows_${num_rows}_cols_${num_cols}_nnz_per_4x4_${num_nnz_per_block}_${exec_mode}"
mkdir -p "$output_folder"

executable="/u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test/test_new"
# Run the C++ executable with the provided arguments
cmd="nsys profile --force-overwrite true --output=$output_folder/nsys_data --trace=nvtx $executable $sparse_matrix_file $num_trials $exec_mode"
echo "Running command: $cmd"
eval $cmd

# Run nsys stats
cmd="nsys stats --force-overwrite true --force-export true --report nvtxsum --format=csv --output=$output_folder/nsys_data $output_folder/nsys_data.nsys-rep"
echo "Running command: $cmd"
eval $cmd

# Run the Python script with the provided arguments
python plot_nvml_data.py $output_folder
