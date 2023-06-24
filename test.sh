#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_err_%j.err
#SBATCH --partition=ogevaert-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module avail