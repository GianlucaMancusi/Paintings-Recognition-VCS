#!/bin/bash
#SBATCH --nodes=1                           # 1 node
#SBATCH --partition=students-prod
#SBATCH --job-name=UNet
#SBATCH --gres=gpu:1
#SBATCH --error=UNET_%j.err.txt            # standard output file
#SBATCH --output=UNET_%j.out.txt            # standard output file
srun python3 train.py --epochs 10 --e-tune 0 -b 4 -l 0.000002 $@