#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=htc
#SBATCH --job-name=LegalReasoning
#SBATCH --gres=gpu:1

module load gpu/cuda/9.2.148
#module load python/anaconda3/5.0.1

#echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
echo $PWD
# run the application
python3 main_small.py -m lstmibgan -s small  > slurm-charge_small_model-$SLURM_JOB_ID.out
