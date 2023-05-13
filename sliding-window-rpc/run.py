#!/usr/bin/env cs_python

import argparse
import csv
import json
import struct
import time
import numpy as np

from cerebras.sdk.runtime import runtime_utils
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder


# Utilities for calculating cycle counts
def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
    return make_u48(words[3:]) - make_u48(words[0:3])

def calculate_cycles(timestamp_buf):
    hex_t0 = int(float_to_hex(timestamp_buf[0]), base=16)
    hex_t1 = int(float_to_hex(timestamp_buf[1]), base=16)
    hex_t2 = int(float_to_hex(timestamp_buf[2]), base=16)
  
    tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
    tsc_tensor_d2h[0] = hex_t0 & 0x0000ffff
    tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000ffff
    tsc_tensor_d2h[2] = hex_t1 & 0x0000ffff
    tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000ffff
    tsc_tensor_d2h[4] = hex_t2 & 0x0000ffff
    tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000ffff
  
    return sub_ts(tsc_tensor_d2h)


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
    arr0_symbol = runner.get_id("arr0")
    arr1_symbol = runner.get_id("arr1")
    symbol_maxmin_time = runner.get_id("maxmin_time")
    
    # Set type and order of the memcpys
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    
    # Input data
    arr0_in_data = np.arange(kernel_width * num_elems, dtype=np.float32)
    arr0_left_pe = arr0_in_data[:num_elems]
    arr0_right_pe = arr0_in_data[(kernel_width-1)*num_elems:]
    
    print()
    print("Left  PE's data at start: ", arr0_left_pe)
    print("Right PE's data at start: ", arr0_right_pe)
    
    # Load and run the program
    print()
    print("Starting run...")
    runner.load()
    runner.run()
    
    # Copy data to arr0 in device
    print("Copy data...")
    runner.memcpy_h2d(arr0_symbol, arr0_in_data, 0, 0, kernel_width, 1, num_elems,
                      streaming=False, data_type=memcpy_dtype,
                      order=memcpy_order, nonblock=False)
    
    print("Launch kernel...")
    runner.call("main_fn", [], nonblock=False)
    
    # Copy back data in arr1 from device
    arr1_out_data = np.zeros([kernel_width*num_elems], dtype=np.float32)
    runner.memcpy_d2h(arr1_out_data, arr1_symbol, 0, 0, kernel_width, 1, num_elems,
                      streaming=False, data_type=memcpy_dtype,
                      order=memcpy_order, nonblock=True)
    print("Copied back data.")
    
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
    
    # Output data
    arr1_left_pe = arr1_out_data[:num_elems]
    arr1_right_pe = arr1_out_data[(kernel_width-1)*num_elems:]
    
    print()
    print("Left  PE's data at end: ", arr1_left_pe)
    print("Right PE's data at end: ", arr1_right_pe)
    
    # Test output data on left is same as input on right, and vice versa
    print()
    print("Testing equality...")
    np.testing.assert_allclose(arr0_left_pe, arr1_right_pe)
    np.testing.assert_allclose(arr1_left_pe, arr0_right_pe)
    print("SUCCESS!")
    
    # Calculate and print cycles on active PEs
    left_cycles   = calculate_cycles(maxmin_time[0, left_pe])
    middle_cycles = calculate_cycles(maxmin_time[0, middle_pe])
    right_cycles  = calculate_cycles(maxmin_time[0, right_pe])
    
    print()
    print(f"Real walltime: {walltime}s")
    print()
    print("Cycle counts:")
    print("Left:   ", left_cycles)
    print("Middle: ", middle_cycles)
    print("Right:  ", right_cycles)
    
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
