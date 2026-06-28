#!/bin/bash -l
#SBATCH --job-name=CDMamba_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_inference_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_inference_%j.err

source ~/.bashrc
conda activate cdmamba_backup

cd /scratch/users/jfernandezmartinez/CDMamba

echo "========================================"
echo "  CD-Mamba Inference + Metrics Job"
echo "  Started: $(date)"
echo "========================================"

python eval_cdmamba_inference.py

echo "========================================"
echo "  Inference Done: $(date)"
echo "========================================"
