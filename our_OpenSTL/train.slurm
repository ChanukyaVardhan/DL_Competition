#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --export=ALL
##SBATCH --exclusive
#SBATCH --requeue

#SBATCH --time=24:00:00
#SBATCH --mem=32GB

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

##SBATCH --cpus-per-task=24
##SBATCH --gres=gpu:2

##SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:v100:4

#SBATCH --job-name=test_%j
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/overlay-25GB-500K.ext3:ro /scratch/crg9968/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/crg9968/DL_Competition/our_OpenSTL
conda activate torch_env

python tools/clevrer_train.py \
    --epoch 200 \
    --dataname "clevrer" \
    --data_root "../data" \
    --method "SimVP" \
    --ex_name "14000cleanvids_simvp_batch_normalized" \
    --auto_resume \
    --batch_size 4 \
    --val_batch_size 4 \
    --fp16 \
    --device "cuda" \
    --use_gpu True
"
