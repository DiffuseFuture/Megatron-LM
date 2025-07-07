#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/nas/njw1123/add_dit_new/

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

torchrun ${DISTRIBUTED_ARGS[@]} /nas/njw1123/add_dit/megatron/core/models/wan/test_vae.py