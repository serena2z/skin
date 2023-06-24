#!/bin/bash -l

#SBATCH --job-name=umap
#SBATCH --output=/share/pi/ogevaert/zhang/body_classifier/umap%j.out
#SBATCH --error=/share/pi/ogevaert/zhang/body_classifier/umap%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=gpu

# Load the necessary modules
module load anaconda
#module load cuda

# Activate the conda environment
source activate /share/pi/ogevaert/zhang/anaconda3/envs/sample_env

# Your Python script or command to run your program
python /share/pi/ogevaert/zhang/body_classifier/body_umap.py

# Deactivate the conda environment when the job is done
conda deactivate