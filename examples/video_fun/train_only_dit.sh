#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTHONPATH=/nas/njw1123/add_dit_new/
export UB_SKIPMC=1
export TRANSFORMER_ENGINE_SKIP_UB=1

GPUS_PER_NODE=8
# Change for multinode config
# NODE_RANK=$MLP_ROLE_INDEX
NODE_RANK=0

# MASTER_ADDR=$MLP_WORKER_0_HOST
MASTER_PORT=$MLP_WORKER_0_PORT
MASTER_ADDR=localhost
# MASTER_PORT=23457
NUM_NODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=/jizhicfs/marvinhjia/njw1123/Megatron-LM/gpt2
TENSORBOARD_LOGS_PATH=/nas/njw1123/add_dit_new/examples/video_fun/output
VOCAB_FILE=/nas/njw1123/gpt/gpt2-vocab.json
MERGE_FILE=/nas/njw1123/gpt/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

GPT_MODEL_ARGS=(
    --pretrained_model_path "/nas/njw1123/models/Wan2.1-Fun-14B-InP"
    --transformer3d_config_path "/nas/njw1123/models/Wan2.1-Fun-14B-InP/config.json"
    --use-flash-attn
    --max-position-embeddings 2048 
    --num-layers 40
    --hidden-size 5120 
    --num-attention-heads 40 
    --seq-length 1024 # fake
    # --decoder-first-pipeline-num-layers 10
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 2
    --train-iters 10
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-3 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --untie-embeddings-and-output-weights
    --moe-token-dispatcher-type "alltoall"
    # --optimizer-cpu-offload
    --recompute-granularity 'full'
    --recompute-method "block"
    --recompute-num-layers 40
    # --recompute-modules layernorm mlp
    # --profile
    # --recompute-modules mlp
    # --log-memory-to-tensorboard
    --data-parallel-random-init
    # --use-distributed-optimizer\
)

MODEL_PARALLEL_ARGS=(
        # --tensor-model-parallel-size 8
        # --pipeline-model-parallel-size 1
        # --context-parallel-size 1
        --tensor-model-parallel-size 4
        --sequence-parallel
        # --tp-comm-overlap
        # --overlap-grad-reduce
        # --overlap-param-gather
        # --pipeline-model-parallel-size 1
        # --transformer-impl local
)

DATA_ARGS=(
    # --data-path $DATA_PATH
    --train_data_meta "/nas/njw1123/test_data/test.json"
    --train_data_dir "/nas/njw1123/test_data/"
    # --dataloader-type cyclic
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    # --split 949,50,1
    --enable_bucket
    --training_with_video_token_length
    --random_hw_adapt
    --variable_seq_lengths
    --token_sample_size 960
    --dataloader-type cyclic
    --dataloader_num_workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    #--save $CHECKPOINT_PATH
    #--load $CHECKPOINT_PATH
    --eval-iters 10
    --profile
    # --profile-ranks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    --profile-ranks 0 1 2 3 4 5 6 7
    --use-pytorch-profiler
    --profile-step-start 2
    --profile-step-end   3
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} /nas/njw1123/add_dit_new/examples/video_fun/pretrain_only_dit.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
# 