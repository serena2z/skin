#!/bin/bash -l

#SBATCH --job-name=skin_segmentation
#SBATCH --output=skin_segmentation%j.out
#SBATCH --error=skin_segmentation%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=ogevaert-a100

# Load the necessary modules
module load anaconda
#module load cuda

# Activate the conda environment
source activate /share/pi/ogevaert/zhang/anaconda3/envs/sample_env

# Your Python script or command to run your program
#python /share/pi/ogevaert/zhang/segment_images.py
python /share/pi/ogevaert/zhang/correct_masks.py

# Deactivate the conda environment when the job is done
conda deactivate