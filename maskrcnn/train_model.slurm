#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_03

## change the partition number to use different number of GPUs
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=24:00:00
#SBATCH --output=mrcnn_%j.out
#SBATCH --error=mrcnn_%j.err
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

echo "MRCNN batch = 4 - New New"

singularity exec --nv \
--bind /scratch \
--overlay /scratch/pj2251/overlay-25GB-500K.ext3:ro \
/scratch/pj2251/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/pj2251/DL/DL_Competition/maskrcnn
conda activate pDL

python mask_rcnn_train.py  --data_dir="/scratch/pj2251/Dataset_Student/" --num_epochs=100 --num_workers=4 --batch_size=4 --model_save_path="../mrcnn_third_run/" --learning_rate=1e-5
"