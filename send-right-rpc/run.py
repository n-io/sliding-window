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

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Load and run the program
runner.load()
runner.run()

in_data = np.arange(num_elems*kernel_width, dtype=np.float32)

print(in_data)

runner.memcpy_h2d(arr0_symbol, in_data, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

runner.call("main_fn", [], nonblock=True)

out_data = np.zeros([num_elems*kernel_width], dtype=np.float32)

runner.memcpy_d2h(out_data, arr0_symbol, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

# Stop the program
runner.stop()

print("hello...")
print(out_data)
