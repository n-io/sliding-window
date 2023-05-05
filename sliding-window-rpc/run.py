#!/usr/bin/env cs_python

import argparse
import json
import struct
import numpy as np

from cerebras.sdk.runtime import runtime_utils
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module


def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])


parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Parse the compile metadata
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)

kernel_width = int(compile_data["params"]["kernel_width"])
num_elems = int(compile_data["params"]["num_elems"])

# Construct a runner using SdkRuntime
runner = SdkRuntime(name, cmaddr=args.cmaddr)

arr0_symbol = runner.get_id("arr0")
arr1_symbol = runner.get_id("arr1")
symbol_maxmin_time = runner.get_id("maxmin_time")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Load and run the program
runner.load()
runner.run()

in0_data = np.zeros(kernel_width * num_elems, dtype=np.float32)
in1_data = np.zeros(kernel_width * num_elems, dtype=np.float32)

in0_data[0] = 2.0;
in0_data[kernel_width * num_elems - 1] = 3.0;

print(in0_data)
print(in1_data)

print("Copy data...")
runner.memcpy_h2d(arr0_symbol, in0_data, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

runner.memcpy_h2d(arr1_symbol, in1_data, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)


print("Launch kernel...")
runner.call("main_fn", [], nonblock=False)


print("Copy back data...")
out0_data = np.zeros([kernel_width*num_elems], dtype=np.float32)
out1_data = np.zeros([kernel_width*num_elems], dtype=np.float32)

runner.memcpy_d2h(out0_data, arr0_symbol, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_d2h(out1_data, arr1_symbol, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

  # Copy back timestamps from device
data = np.zeros([kernel_width*3], dtype=np.float32)
runner.memcpy_d2h(data, symbol_maxmin_time, 0, 0, kernel_width, 1, 3,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)
maxmin_time_hwl = data.view(np.float32).reshape((1, kernel_width, 3))
print("Copied back timestamps.")

# Stop the program
runner.stop()

print(out0_data)
print(out1_data)

tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
for w in range(kernel_width):
  hex_t0 = int(float_to_hex(maxmin_time_hwl[(0, w, 0)]), base=16)
  hex_t1 = int(float_to_hex(maxmin_time_hwl[(0, w, 1)]), base=16)
  hex_t2 = int(float_to_hex(maxmin_time_hwl[(0, w, 2)]), base=16)
  tsc_tensor_d2h[0] = hex_t0 & 0x0000ffff
  tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000ffff
  tsc_tensor_d2h[2] = hex_t1 & 0x0000ffff
  tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000ffff
  tsc_tensor_d2h[4] = hex_t2 & 0x0000ffff
  tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000ffff

  cycles = sub_ts(tsc_tensor_d2h)
  print(w, cycles)
