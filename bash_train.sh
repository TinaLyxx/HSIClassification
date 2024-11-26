export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=29500 main.py \
    --cfg config.yaml \
    --batch-size 128 \
    --dataset IP 