#!/usr/bin/env python

import os
import argparse
import math
import numpy as np
import subprocess
import pandas as pd
from matplotlib import pyplot as plt

####

clock_speed = 850.E6 # Hz
peta = 1.E15

####

parser = argparse.ArgumentParser()
parser.add_argument('--cs2', help='run on CS-2', default=False, action='store_true')
args = parser.parse_args()

cs2 = args.cs2

####

if cs2:
  csv_name = test_name + '_cs2.csv'
else:
  csv_name = test_name + '_sim.csv'

df = pd.read_csv(csv_name)

max_left_cycles   = max(df['left_cycles'])
max_middle_cycles = max(df['middle_cycles'])
max_right_cycles  = max(df['right_cycles'])
max_cycles = max(max_left_cycles, max_middle_cycles, max_right_cycles)

max_kernel_width = max(df['kernel_width'])

# Create unique ordered list of all strides
num_elems = list(dict.fromkeys(df['num_elems'].to_list()))

# Plot bandwidth
for num_elem in num_elems:
  left_cycles = df.loc[df['num_elems'] == num_elem, 'left_cycles']
  kernel_width = df.loc[df['num_elems'] == num_elem, 'kernel_width']
  plt.plot(kernel_width, left_cycles, label=num_elem)

plt.legend(bbox_to_anchor=(1.05, 1.05))
plt.grid()
plt.xlabel('Kernel Width (Number of PEs)')
plt.ylabel('Left PE Cycles')
plt.xlim([0, max_kernel_width])
plt.ylim([0, max_cycles])

if not cs2:
  plt.title("Left Cycle Count, Simulation")
  plt.savefig('left_cycle_sim.png', bbox_inches='tight')
else:
  plt.title("Left Cycle Count, CS2")
  plt.savefig('left_cycle_cs2.png', bbox_inches='tight')
