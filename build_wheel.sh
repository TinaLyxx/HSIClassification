#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8000M


module load cuda/11.7
module load python/3.9

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cd third_party/triton-2.1.0/python
python setup.py bdist_wheel

# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install build
# pip install -r requirements.txt

# cd third_party/TreeGen 
# python setup.py bdist_wheel

# cd third_party/causal-conv1d-1.4.0
# python setup.py bdist_wheel

