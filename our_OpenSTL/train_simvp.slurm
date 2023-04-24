#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_03

## change the partition number to use different number of GPUs
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=24:00:00
#SBATCH --output=simvp_24_pruned_%j.out
#SBATCH --error=simvp_24_pruned_%j.err
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

echo "SimVP Single GPU Training on less than 14k videos. Batch size = 4"

singularity exec --nv \
--bind /scratch \
--overlay /scratch/pj2251/overlay-25GB-500K.ext3:ro \
/scratch/pj2251/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/pj2251/DL/DL_Competition/our_OpenSTL
conda activate pDL

python tools/clevrer_train.py \
    --epoch 200 \
    --dataname "clevrer" \
    --data_root "../../../Dataset_Student" \
    --method "SimVP" \
    --ex_name "14000cleanvids_simvp_batch" \
    --auto_resume \
    --batch_size 1 \
    --val_batch_size 4 \
    --fp16 \
    --device "cuda" \
    --use_gpu True
"