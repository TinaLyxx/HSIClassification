export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS # 总进程数
export RANK=$SLURM_PROCID 
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 main.py \
    --cfg config/config.yaml \
    --batch-size 32 \
    --patch-size 9 \
    --dataset IP \
    --sample-mode fixed \
    --train-size 70 \
    --test-size 20