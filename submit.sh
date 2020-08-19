#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=Legal
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1

module load python/anaconda3/2019.03
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.5.0__cuda-10.0
echo "CUDA Device(s) : $CUDA_VISIBLE_DEVICES"
nvidia-smi

# run the application
python3 main.py -m transformer
