#!/bin/bash

# The default output of ncu is extensive. This script let you run ncu profiler
# with the specific metrics for a specific kernel in the cuda app.
#
# Usage: bash ./my_ncu.sh --kernel-name your_kernel ./your_cuda_app




# Define metric keys and friendly labels in parallel arrays
# TODO: add more metric keys introduced in
#       https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
METRIC_KEYS=(
  # 1. instruction throughput
  smsp__average_inst_executed_per_warp.ratio
  smsp__inst_executed.sum
  smsp__inst_executed.avg.per_cycle_active

  # 2. Occupancy and Resource Usage.
  sm__warps_active.avg.pct_of_peak_sustained_active
  smsp__cycles_active.avg.pct_of_peak_sustained_elapsed

  # 3. Memory Access Efficiency
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
  dram__bytes_read.sum.per_second

  # 4. Warp & Thread Efficiency
  #smsp__thread_inst_executed_per_inst_executed.ratio
  #smsp__sass_average_branch_targets_threads_uniform.pct

  # 5. Duration
  gpu__time_duration.sum

)

METRIC_LABELS=(
  # 1. Instruction Throughput
  "inst_per_warp" # Avg. number of instructions per warp
  "inst_executed" # Total instructions executed
  "inst_per_cycle" # Instructions per cycle

  # 2. Occupancy and Resource Usage
  "achieved occupancy"
  "sm_efficiency"

  # 3. Memory Access Efficiency
  "gld_throughput"
  "gld_efficiency"
  "gst_throughput"
  "gst_efficiency"
  "dram_read_throughput"

  # 4. Warp & Thread Efficiency
  #"warp_execution_efficiency" # Ratio of active threads in warps
  #"branch_efficiency" # higher if there are less branches. why not working properly??

  # 5. Duration
  "duration"
)

# Delimit with pipe character instead of space
METRIC_STRING=$(IFS=, ; echo "${METRIC_KEYS[*]}")
KEYS_JOINED=$(IFS='|'; echo "${METRIC_KEYS[*]}")
LABELS_JOINED=$(IFS='|'; echo "${METRIC_LABELS[*]}")


# run and print
# printf "[args...]: %s\n" "$*"
echo "[kernel-name]: $2"
ncu --metrics "$METRIC_STRING" "$@" 2>/dev/null \
| awk -v keys="$KEYS_JOINED" -v labels="$LABELS_JOINED" '
  BEGIN {
    n = split(keys, keyArr, "|")
    split(labels, labelArr, "|")
  }

  /[0-9]+,[0-9]+,[0-9]+\)/ {kernel=$0}

  {
    for (i = 1; i <= n; ++i) {
      if ($0 ~ keyArr[i]) {
        value[keyArr[i]] = $(NF) " " $(NF - 1)
      }
    }
  }

  END {
    print "--------------------------------------"
    for (i = 1; i <= n; ++i) {
      printf "%-25s %8s\n", labelArr[i] ":", value[keyArr[i]]
    }
    print "--------------------------------------"
  }'
