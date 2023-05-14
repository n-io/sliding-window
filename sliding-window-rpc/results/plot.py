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
parser.add_argument('--side', help='left, middle, or right', type=str, default='left')
args = parser.parse_args()

cs2 = args.cs2
side = args.side
assert(side in {'left', 'middle', 'right'})


####

if cs2:
  csv_name = 'out_cs2.csv'
else:
  csv_name = 'out_sim.csv'

df = pd.read_csv(csv_name)

max_left_cycles   = max(df['left_cycles'])
max_middle_cycles = max(df['middle_cycles'])
max_right_cycles  = max(df['right_cycles'])
max_cycles = max(max_left_cycles, max_middle_cycles, max_right_cycles)
max_cycles = 2500

max_kernel_width = max(df['kernel_width'])

# Create unique ordered list of all strides
num_elems = list(dict.fromkeys(df['num_elems'].to_list()))

# Plot bandwidth
for num_elem in num_elems:
  cycles = df.loc[df['num_elems'] == num_elem, side + '_cycles']
  kernel_width = df.loc[df['num_elems'] == num_elem, 'kernel_width']
  plt.plot(kernel_width, cycles, '-o', label=num_elem)

plt.legend(bbox_to_anchor=(1.05, 1.05), title='Num elems')
plt.grid()
plt.xlabel('Kernel Width (Number of PEs)')
plt.ylabel(side + ' PE Cycles')
plt.xlim([0, max_kernel_width])
plt.ylim([0, max_cycles])

if not cs2:
  plt.title(side + ' Cycle Count, Simulation')
  plt.savefig(side + '_cycle_sim.png', bbox_inches='tight')
else:
  plt.title(side + ' Cycle Count, CS2')
  plt.savefig(side + '_cycle_cs2.png', bbox_inches='tight')
