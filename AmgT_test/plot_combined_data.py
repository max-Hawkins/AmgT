import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

def parse_directory_name(dir_name):
    """Extract parameters from directory name."""
    pattern = r'sparse_matrix_rows_(\d+)_cols_(\d+)_nnz_per_4x4_(\d+)_(\w+)'
    match = re.match(pattern, os.path.basename(dir_name))
    if match:
        return {
            'num_rows': int(match.group(1)),
            'num_cols': int(match.group(2)),
            'nnz_per_4x4': int(match.group(3)),
            'exec_mode': match.group(4)
        }
    return None

def process_directory(directory):
    """Process CSV files in directory and return averaged metrics."""
    params = parse_directory_name(directory)
    print("Params: ", params)
    if not params:
        return None

    # Find all CSV files in the directory
    csv_file = os.path.join(directory, "gpuStats.csv")
    timing_csv_file = os.path.join(directory, "timeStats.csv")
    nvtx_csv_file = os.path.join(directory, "nsys_data_nvtxsum.csv")

    df = pd.read_csv(csv_file)
    timing_df = pd.read_csv(timing_csv_file)
    nvtx_df = pd.read_csv(nvtx_csv_file)

    # Initialize lists to store metrics
    avg_power = df[' power_draw_w'][1250:-1250].mean()
    avg_gpu_util = df[' utilization_gpu'][1250:-1250].mean()
    avg_mem_util = df[' utilization_memory'][1250:-1250].mean()

    median_kernel_time = timing_df['time_ms'].median()
    mean_kernel_time = timing_df['time_ms'].mean()
    max_kernel_time = timing_df['time_ms'].max()
    min_kernel_time = timing_df['time_ms'].min()

    # Calculate final averages across all files
    params.update({
        'avg_power': avg_power,
        'avg_gpu_util': avg_gpu_util,
        'avg_mem_util': avg_mem_util,
        'median_kernel_time': median_kernel_time,
        'mean_kernel_time': mean_kernel_time,
        'max_kernel_time': max_kernel_time,
        'min_kernel_time': min_kernel_time
    })

    return params

def main(root_dir):
    # Process all directories
    results = []
    for dir_name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(full_path):
            result = process_directory(full_path)
            if result:
                results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle('Performance Metrics vs NNZ per 4x4 Block')

    # For nicer plotting, sort the df by nnz_per_4x4
    df = df.sort_values(by='nnz_per_4x4')

    # Plot for each execution mode
    for exec_mode in df['exec_mode'].unique():
        mode_data = df[df['exec_mode'] == exec_mode]

        ax1.plot(mode_data['nnz_per_4x4'], mode_data['avg_power'],
                marker='o', label=exec_mode)
        ax2.plot(mode_data['nnz_per_4x4'], mode_data['avg_gpu_util'],
                marker='o', label=f"{exec_mode} GPU Util")
        ax2.plot(mode_data['nnz_per_4x4'], mode_data['avg_mem_util'],
                marker='o', label=f"{exec_mode} Mem Util")
        ax3.plot(mode_data['nnz_per_4x4'], mode_data['median_kernel_time'],
                marker='o', label=f"{exec_mode} Median Kernel Time")
        ax3.plot(mode_data['nnz_per_4x4'], mode_data['min_kernel_time'],
                marker='x', label=f"{exec_mode} Min Kernel Time")

        ax4.plot(mode_data['nnz_per_4x4'],
                mode_data['median_kernel_time'] * mode_data['avg_power'] / 1000,
                marker='o',
                label=f"{exec_mode}")

    # Customize plots
    ax1.set_ylabel('Average Power (W)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_ylabel('Average Utilization (%)')
    ax2.legend()
    ax2.grid(True)

    ax3.set_xlabel('NNZ per 4x4 Block')
    ax3.set_ylabel('Runtime (ms)')
    ax3.legend()
    ax3.grid(True)

    ax4.set_xlabel('NNZ per 4x4 Block')
    ax4.set_ylabel('Joules per SpMV')
    ax4.set_ylim(0, ax4.get_ylim()[1])
    ax4.legend()
    ax4.grid(True)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'performance_metrics.png'))
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_combined_data.py <root_data_directory>")
        sys.exit(1)

    main(sys.argv[1])
