#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=devel
#SBATCH --job-name=LegalReasoning
#SBATCH --gres=gpu:1

module load cuda/9.2

#echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
echo $PWD
# run the application
python3 main.py -m lstmibgan
