#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/root/add_dit/
export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2
# export NVTE_FUSED_ATTN=cd

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=/jizhicfs/marvinhjia/njw1123/Megatron-LM/gpt2
TENSORBOARD_LOGS_PATH=/root/add_dit/examples/video_fun/output
VOCAB_FILE=/root/Megatron-LM/examples/gpt3/gpt2-vocab.json
MERGE_FILE=/root/Megatron-LM/examples/gpt3/gpt2-merges.txt
DATA_PATH=/jizhicfs/marvinhjia/njw1123/Megatron-LM/examples/gpt3/web_content_document
# DATA_PATH

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    # --num-layers 96 
    # --hidden-size 12288 
    # --num-attention-heads 96 
    # --seq-length 2048 
    --max-position-embeddings 2048 
    # --attention-backend flash # Can use (flash/fused/unfused/local)
    --use-flash-attn
    --num-layers 30 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
   # --tensor-model-parallel-size 1 
   # --pipeline-model-parallel-size 1 
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 16
    --train-iters 1
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --untie-embeddings-and-output-weights
    --moe-token-dispatcher-type "alltoall"
)

MODEL_PARALLEL_ARGS=(
        # --tensor-model-parallel-size 8
        # --pipeline-model-parallel-size 16
        --tensor-model-parallel-size 4
        --pipeline-model-parallel-size 2
        # --transformer-impl local
)

DATA_ARGS=(
    --data-path $DATA_PATH
    # --dataloader-type cyclic
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
    --enable_bucket
    --training_with_video_token_length
    --random_hw_adapt
    --variable_seq_lengths
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    #--save $CHECKPOINT_PATH
    #--load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} /root/add_dit/examples/video_fun/pretrain_video_fun.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
