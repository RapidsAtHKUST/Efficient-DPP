#!/bin/bash

for (( i=128;i<4096;i=i+256 ))
do
    #../bin/opencl_compare -n $i
    echo $i
done
