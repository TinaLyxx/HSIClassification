#!/bin/bash

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8000M


module load cuda/11.7
module load python/3.9
nvidia-smi

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# install triton==2.1.0 from source
# wget https://github.com/triton-lang/triton/archive/refs/tags/v2.1.0.tar.gz
# tar -xzvf v2.1.0.tar.gz -C third_party
# cd third_party/triton-2.1.0
# pip install -e python

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

cd whl
pip install causal_conv1d-1.4.0-cp39-cp39-linux_x86_64.whl
pip install triton-2.1.0-cp39-cp39-linux_x86_64.whl 

# install causal-conv1d>=1.4.0 from source
# wget https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.4.0.tar.gz
# tar -xzvf v1.4.0.tar.gz -C third_party
# cd third_party/causal-conv1d-1.4.0
# pip install -v -e .

cd ../
pip install -r requirements.txt

cd whl
pip install tree_generate-1.0-cp39-cp39-linux_x86_64.whl

cd ../
chmod +x bash_train.sh
bash ./bash_train.sh
