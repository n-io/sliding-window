#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime import runtime_utils
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

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

runner.memcpy_h2d(arr0_symbol, in0_data, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

runner.memcpy_h2d(arr1_symbol, in1_data, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

print("make it here?")

runner.call("main_fn", [], nonblock=False)

print("make it here?")

out0_data = np.zeros([kernel_width*num_elems], dtype=np.float32)
out1_data = np.zeros([kernel_width*num_elems], dtype=np.float32)

print("make it here?")

runner.memcpy_d2h(out0_data, arr0_symbol, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.memcpy_d2h(out1_data, arr1_symbol, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

print("make it here?")

# Stop the program
runner.stop()

print(out0_data)
print(out1_data)
