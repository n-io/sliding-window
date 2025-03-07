#!/usr/bin/env python

import subprocess
import argparse


def parse_args():
  """ parse the command line """

  parser = argparse.ArgumentParser(description="Sweep sliding window parameters")
  parser.add_argument("--name", required=False, default="out",
                      help="Prefix of ELF files")
  parser.add_argument("--cmaddr", required=False, default="",
                      help="IP:port for CS system")

  args = parser.parse_args()

  return args


def cslc_compile(
    kernel_width: int,
    num_elems: int,
    name: str,
    cmaddr: str
  ):
  """Generate ELFs for the layout"""

  if cmaddr:
    fab_width = 757
    fab_height = 996
  else:
    fab_width = kernel_width+7
    fab_height = 3

  args = []
  args.append("cslc") # command
  args.append(f"layout.csl") # file
  args.append(f"--fabric-dims={fab_width},{fab_height}") # options
  args.append("--fabric-offsets=4,1")
  args.append(f"--params=kernel_width:{kernel_width},num_elems:{num_elems}")
  args.append(f"-o={name}")
  args.append("--arch=wse3")
  args.append("--memcpy")
  args.append("--channels=1")
  print(f"subprocess.check_call(args = {args}")
  subprocess.check_call(args)

def cs_run(
    name: str,
    cmaddr: str
  ):
  """Run with cs_python"""

  args = []
  args.append("cs_python")
  args.append("run.py")
  args.append(f"--name={name}")
  args.append(f"--cmaddr={cmaddr}")
  subprocess.check_call(args)


def compile_and_run(
    kernel_width: int,
    num_elems: int,
    name: str,
    cmaddr: str
  ):
  """Compile and run program."""

  cslc_compile(
    kernel_width,
    num_elems,
    name,
    cmaddr)

  cs_run(name, cmaddr)


def main():
  """Main method to run the example code."""

  args = parse_args()

  name = args.name # compilation output
  cmaddr = args.cmaddr

  # kernel_width has been hard-coded to 8
  kernel_width=8
  for num_elems in range(1,501,10):
    compile_and_run(
      kernel_width,
      num_elems,
      name,
      cmaddr)


if __name__ == "__main__":
  main()
