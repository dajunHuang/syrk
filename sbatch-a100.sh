#!/bin/bash
#SBATCH --job-name='jobs-a100'
#SBATCH --partition=gpu7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-task=a100:1

source ~/.bashrc

echo "Job started on $(hostname)"
echo "SLURM allocated GPUs: ${CUDA_VISIBLE_DEVICES}"

./test-a100.sh

echo "Job finished"
