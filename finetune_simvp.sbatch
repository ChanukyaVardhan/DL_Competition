#!/bin/bash

#SBATCH --job-name=finetune_simvp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=32G
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00

module purge

singularity exec --nv \
	    --overlay /scratch/snm6477/singu/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "

source /ext3/env.sh
cd /scratch/snm6477/github/DL_Competition/
python finetune_simvp.py --config_path ./config/simvp_finetune.yaml

"


