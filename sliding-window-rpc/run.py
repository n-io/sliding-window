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

print("make it here?")

runner.call("main_fn", [], nonblock=False)

print("make it here?")

out_data = np.zeros([kernel_width*num_elems], dtype=np.int32)

print("make it here?")

runner.memcpy_d2h(out_data, arr0_symbol, 0, 0, kernel_width, 1, num_elems,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=True)

print("make it here?")
print(out_data)

# Stop the program
runner.stop()
