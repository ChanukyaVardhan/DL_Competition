#!/bin/bash

#SBATCH --job-name=convttlst
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:2
#SBATCH --time=10:00:00

module purge

singularity exec --nv \
	    --overlay /scratch/snm6477/singu/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "

source /ext3/env.sh
cd /scratch/snm6477/github/DL_Competition/convttlstm
python3 -m torch.distributed.launch --nproc_per_node=2 model_train.py --use_distributed --batch_size=4 --no_sigmoid --valid_samples=500 --num_epochs=10 --future_frames=11 --output_frames=11 --use_amp --gradient_clipping --train_samples_epoch 3000

"


