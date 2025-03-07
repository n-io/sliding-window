#!/usr/bin/env cs_python

import argparse
import csv
import json
import struct
import time
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
from cerebras.sdk import sdk_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='the test name')
    parser.add_argument("--cmaddr", help="IP:port for CS system")
    
    args = parser.parse_args()
    name = args.name
    cmaddr = args.cmaddr
    
    # Parse the compile metadata
    with open(f"{name}/out.json", encoding="utf-8") as json_file:
        compile_data = json.load(json_file)
    
    kernel_width = int(compile_data["params"]["kernel_width"])
    num_elems = int(compile_data["params"]["num_elems"])
    
    left_pe   = 0
    right_pe  = kernel_width-1
    middle_pe = right_pe//2
    
    print()
    print("Run parameters:")
    print("Kernel width = ", kernel_width)
    print("Elems to transfer = ", num_elems)
    
    start_time = time.time()
    # Construct a runner using SdkRuntime
    runner = SdkRuntime(name, cmaddr=cmaddr)
    
    # Grab symbols from ELF files for copying data on/ off device
    symbol_maxmin_time = runner.get_id("maxmin_time")
    
    # Set type and order of the memcpys
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    
    print()
    
    # Load and run the program
    print()
    print("Starting run...")
    runner.load()
    runner.run()
    
    print("Launch kernel...")
    runner.launch("main_fn", nonblock=False)
    
    # Copy back timestamps from device
    data = np.zeros([kernel_width*3], dtype=np.float32)
    runner.memcpy_d2h(data, symbol_maxmin_time, 0, 0, kernel_width, 1, 3,
                      streaming=False, data_type=memcpy_dtype,
                      order=memcpy_order, nonblock=False)
    maxmin_time = data.view(np.float32).reshape((1, kernel_width, 3))
    print("Copied back timestamps.")
    
    # Stop the program
    runner.stop()
    
    end_time = time.time()
    walltime = end_time-start_time
    print("Done.")

    # Calculate and print cycles on active PEs
    left_cycles   = sdk_utils.calculate_cycles(maxmin_time[0, left_pe])
    middle_cycles = sdk_utils.calculate_cycles(maxmin_time[0, middle_pe])
    right_cycles  = sdk_utils.calculate_cycles(maxmin_time[0, right_pe])
    
    print()
    print(f"Real walltime: {walltime}s")
    print()
    print("Cycle counts:")
    print("Left:   ", left_cycles)
    print("Middle: ", middle_cycles)
    print("Right:  ", right_cycles)

    print("All:    ", [sdk_utils.calculate_cycles(t) for t in maxmin_time[0, :]])
    print()
    
    # Write a CSV
    if cmaddr:
        csv_name = "out_cs2.csv"
    else:
        csv_name = "out_sim.csv"
    
    with open(csv_name, mode='a') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([kernel_width, num_elems, left_cycles,
                             middle_cycles, right_cycles, walltime])


if __name__ == "__main__":
    main()
