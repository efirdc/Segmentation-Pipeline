#!/bin/bash

salloc \
--nodes=1 \
--gres=gpu:v100l:1 \
--time=0-3:00:0 \
--ntasks-per-node=1 \
--cpus-per-task=4 \
--mem=16G \
--account=def-uofavis-ab