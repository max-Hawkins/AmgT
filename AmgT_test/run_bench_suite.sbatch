#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --job-name=amgt_bench_suite
#SBATCH --output=amgt_bench_suite.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mhawkins60@gatech.edu
#SBATCH --account=bdiy-delta-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --mem=32G

# Load the modules
cd /u/mhawkins/local/MFC
. mfc.sh load -c d -m g

cd /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test
# Run all the benchmarks

num_rows=4096
num_cols=4096

for nnz_per_block in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
    for exec_mode in cusparse CC TC; do
        echo "Running with $nnz_per_block nnz per block and $exec_mode exec mode"
        /u/mhawkins/iter_methods_proj_amgT/AmgT/AmgT_test/run_bench.sh $num_rows $num_cols $nnz_per_block $exec_mode 80000
    done
done
