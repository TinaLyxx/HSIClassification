#!/bin/bash

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=8000M


module load cuda/11.7
module load python/3.9
nvidia-smi

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip


pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

cd whl
pip install causal_conv1d-1.4.0-cp39-cp39-linux_x86_64.whl
pip install triton-2.1.0-cp39-cp39-linux_x86_64.whl 


cd ../
pip install -r requirements.txt

cd whl
pip install tree_generate-1.0-cp39-cp39-linux_x86_64.whl

cd ../
export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID 
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python \
    main_single.py \
    --cfg config/config.yaml \
    --batch-size 64 \
    --patch-size 17 \
    --dataset IP \
    --sample-mode fixed \
    --train-size 70 \
    --test-size 20

