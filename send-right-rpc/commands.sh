#!/usr/bin/env bash

set -e

cslc layout.csl --fabric-dims=20,3 --fabric-offsets=4,1 --params=kernel_width:3,num_elems:3 --memcpy --channels=1
cs_python run.py --name out
