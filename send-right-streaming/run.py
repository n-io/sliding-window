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

# Color used for memcpy H2D streaming
memcpy_h2d_color = 0
memcpy_d2h_color = 1

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Load and run the program
runner.load()
runner.run()

data = np.arange(num_elems, dtype=np.int32)
print(data)

runner.memcpy_h2d(memcpy_h2d_color, data, 0, 0, 1, 1, num_elems,
                  streaming=True, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

right_out_data = np.zeros([num_elems], dtype=np.int32)

runner.memcpy_d2h(right_out_data, memcpy_d2h_color, kernel_width-1, 0, 1, 1, num_elems,
                  streaming=True, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

print(right_out_data)

# Stop the program
runner.stop()
