#!/usr/bin/env bash

set -e

cslc layout.csl --fabric-dims=20,3 --fabric-offsets=4,1 --params=MEMCPYH2D_DATA_1_ID:0,MEMCPYD2H_DATA_1_ID:1,kernel_width:3,num_elems:1 --memcpy --channels=1
cs_python run.py --name out
