#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTHONPATH=/nas/njw1123/add_dit/

GPUS_PER_NODE=8
# Change for multinode config
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST
MASTER_PORT=$MLP_WORKER_0_PORT
NUM_NODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=/jizhicfs/marvinhjia/njw1123/Megatron-LM/gpt2
TENSORBOARD_LOGS_PATH=/nas/njw1123/add_dit/examples/video_fun/output
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
    --transformer3d_config_path "/nas/njw1123/models/Wan2.1-Fun-14B-InP/config_fake.json"
    --use-flash-attn
    --max-position-embeddings 2048 
    --num-layers 4
    --hidden-size 5120 
    --num-attention-heads 40 
    --seq-length 1024 # fake
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
    # --use-distributed-optimizer
)

MODEL_PARALLEL_ARGS=(
        # --tensor-model-parallel-size 8
        # --pipeline-model-parallel-size 16
        --context-parallel-size 2
        --tensor-model-parallel-size 2
        --pipeline-model-parallel-size 2
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
    --token_sample_size 512
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

torchrun ${DISTRIBUTED_ARGS[@]} /nas/njw1123/add_dit/examples/video_fun/pretrain_video_fun.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
