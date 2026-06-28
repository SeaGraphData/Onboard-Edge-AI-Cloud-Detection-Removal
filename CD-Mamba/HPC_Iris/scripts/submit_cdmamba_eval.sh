#!/bin/bash -l
#SBATCH --job-name=CDMamba_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_eval_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_eval_%j.err

source ~/.bashrc
conda activate cdmamba_backup

cd /scratch/users/jfernandezmartinez/CDMamba

python eval_cdmamba.py
