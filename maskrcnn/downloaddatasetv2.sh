#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_03

## change the partition number to use different number of GPUs
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=12:00:00
#SBATCH --output=down_%j.out
#SBATCH --error=down_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

echo "Unzipping Dataset!"

singularity exec --nv \
--bind /scratch \
--overlay /scratch/pj2251/overlay-25GB-500K.ext3:rw \
/scratch/pj2251/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash -c "
unzip /scratch/pj2251/Dataset_Student_V2.zip
"