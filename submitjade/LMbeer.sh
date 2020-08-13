#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --job-name=LegalReasoning
#SBATCH --gres=gpu:1

module load cuda/9.2

#echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
echo $PWD
# run the application
python3 LanguageModel_beer.py > slurm-LMbeer-$SLURM_JOB_ID.out
