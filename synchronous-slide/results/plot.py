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
  csv_name = 'out_cs2.csv'
else:
  csv_name = 'out_sim.csv'

df = pd.read_csv(csv_name)

max_left_cycles   = max(df['left_cycles'])
max_middle_cycles = max(df['middle_cycles'])
max_right_cycles  = max(df['right_cycles'])
max_cycles = max(max_left_cycles,max_middle_cycles,max_right_cycles)

max_kernel_width = max(df['kernel_width'])
max_elems_per_pe = max(df['elems_per_pe'])

# Create unique ordered list of all strides and elems
kernel_width_list = list(dict.fromkeys(df['kernel_width'].to_list()))
num_elem_list = list(dict.fromkeys(df['elems_per_pe'].to_list()))

# Plot bandwidth
for kernel_width in kernel_width_list:
  elems_per_pe = df.loc[df['kernel_width'] == kernel_width, 'elems_per_pe']
  left_cycles = df.loc[df['kernel_width'] == kernel_width, 'left_cycles']
  middle_cycles = df.loc[df['kernel_width'] == kernel_width, 'middle_cycles']
  right_cycles = df.loc[df['kernel_width'] == kernel_width, 'right_cycles']

  label_root = str(kernel_width) + ' PEs, '
  plt.plot(elems_per_pe*kernel_width, left_cycles, '-o', label=label_root + 'left')
  plt.plot(elems_per_pe*kernel_width, middle_cycles, '-o', label=label_root + 'middle')
  plt.plot(elems_per_pe*kernel_width, right_cycles, '-o', label=label_root + 'right')
  plt.plot(elems_per_pe*kernel_width, right_cycles, '-o', label=label_root + 'right')

plt.legend(bbox_to_anchor=(1.05, 1.05))
plt.grid()
plt.xlabel('Num Elems')
plt.ylabel('PE Cycles')
plt.xlim([0, max_elems_per_pe*max_kernel_width])
plt.ylim([0, max_cycles])

if not cs2:
  plt.title("Cycle Count, Simulation")
  plt.savefig('cycle_sim.png', bbox_inches='tight')
else:
  plt.title("Cycle Count, CS2")
  plt.savefig('cycle_cs2.png', bbox_inches='tight')

# Plot cycles per element
plt.clf()
plt.plot(elems_per_pe*kernel_width, right_cycles / (elems_per_pe*kernel_width), '-o')
plt.grid()
plt.xlabel('Num Elems')
plt.ylabel('Cycles per element')
plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4])

if not cs2:
  plt.title("Worst Cycles per Element, Simulation")
  plt.savefig('cycle_per_elem_sim.png', bbox_inches='tight')
else:
  plt.title("Worst Cycles per Element, CS2")
  plt.savefig('cycle_per_elem_cs2.png', bbox_inches='tight')

