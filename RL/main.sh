#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH -o out.log
#SBATCH -e err.log

module load nvidia/cuda/10.0
module load anaconda/3.7
source activate Zixuan
export PYTHONUNBUFFERED=1
which python

python DQNSimple.py \

    # --R_alpha 3
    # --pred \
    # --adv_method pgd
    # --save \
    # /home/zhanghaiyang/project/pretrained/bert-base-chinese
    # /home/zhanghaiyang/project/pretrained/ernie-3.0-base-chinese
        # --LSTM \
