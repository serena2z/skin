#!/bin/bash -l

#SBATCH --job-name=body
#SBATCH --output=/share/pi/ogevaert/zhang/body_classifier/body%j.out
#SBATCH --error=/share/pi/ogevaert/zhang/body_classifier/body%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load the necessary modules
module load anaconda
#module load cuda

# Activate the conda environment
source activate /share/pi/ogevaert/zhang/anaconda3/envs/sample_env

# Your Python script or command to run your program
python /share/pi/ogevaert/zhang/body_classifier/body_classifier.py

# Deactivate the conda environment when the job is done
conda deactivate