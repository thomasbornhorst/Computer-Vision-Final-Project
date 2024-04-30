#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32gb
#SBATCH --time=0-20:00:00
#SBATCH --mail-user=cwc63@case.edu
#SBATCH --mail-type=ALL
#SBATCH --account=dlw

module load Python/3.10.4-GCCcore-11.3.0 
python fast_RCNN.py
