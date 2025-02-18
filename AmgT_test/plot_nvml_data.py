import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

if len(sys.argv) != 2:
    print("Usage: python plot_nvml_data.py <root_folder>")
    sys.exit(1)

root_folder = sys.argv[1]
csv_file = os.path.join(root_folder, "gpuStats.csv")
timing_csv_file = os.path.join(root_folder, "timeStats.csv")
nvtx_csv_file = os.path.join(root_folder, "nsys_data_nvtxsum.csv")
plot_file = os.path.join(root_folder, "gpuStats.png")

df = pd.read_csv(csv_file)
timing_df = pd.read_csv(timing_csv_file)
nvtx_df = pd.read_csv(nvtx_csv_file)
# Index(['timestamp', ' temperature_gpu', ' power_draw_w', ' power_limit_w',
#        ' utilization_gpu', ' utilization_memory', ' memory_used_mib',
#        ' memory_free_mib', ' clocks_throttle_reasons_active',
#        ' clocks_current_sm_mhz', ' clocks_applications_graphics_mhz',
#        ' clocks_current_memory_mhz', ' clocks_max_memory_mhz', ' pstate'],
# Create figure with 3 subplots
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 14))

# Plot clocks in first subplot
ax4.plot(df[' clocks_current_sm_mhz'], label='SM Clock (MHz)')
ax4.plot(df[' clocks_current_memory_mhz'], label='Memory Clock (MHz)')
ax4.plot(df[' clocks_applications_graphics_mhz'], label='Graphics Clock (MHz)')
ax4.set_ylabel('Clock Speed (MHz)')
ax4.legend()
ax4.grid(True)

# Plot temperature and power in second subplot
ax3.plot(df[' temperature_gpu'], label='Temperature (°C)', color='red')
ax3.set_ylabel('Temperature (°C)')
ax3.legend()
ax3.grid(True)

ax1.plot(df[' power_draw_w'], label='Power Draw (W)', color='orange')
ax1.set_ylabel('Power (W)')
ax1.legend()
ax1.grid(True)

# Plot utilization in third subplot
ax2.plot(df[' utilization_gpu'], label='GPU Utilization %', color='green')
ax2.plot(df[' utilization_memory'], label='Memory Utilization %', color='blue')
ax2.set_ylabel('Utilization %')
ax2.set_xlabel('Time Steps')
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True)

ax5.plot(timing_df['time_ms'], label='Time (ms)')
ax5.set_ylabel('Time (ms)')
ax5.set_xlabel('Trial')
ax5.legend()
ax5.set_yscale('log')
ax5.grid(True)

# Create figure with 6 subplots
# Add histogram of timing data with range from min to 90th percentile
min_time = timing_df['time_ms'].min()
median_time = timing_df['time_ms'].median()
mean_time = timing_df['time_ms'].mean()
max_time = timing_df['time_ms'].max()
p90_time = timing_df['time_ms'].quantile(0.9)
ax6.hist(timing_df['time_ms'], bins=200, edgecolor='black', range=(min_time, p90_time))
ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Trial Times')
ax6.text(0.8, 0.95, f'Total counts: {len(timing_df)}\nMin time: {min_time:.2f} ms\nMedian time: {median_time:.2f} ms\nMean time: {mean_time:.2f} ms\nMax time: {max_time:.2f} ms', ha='right', va='center', transform=ax6.transAxes)
ax6.grid(True)


plt.tight_layout()
plt.savefig(plot_file)

