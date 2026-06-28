#!/bin/bash -l
#SBATCH --job-name=omnicloud_metrics
#SBATCH --output=/scratch/users/jfernandezmartinez/GeoAI/logs/metrics_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/GeoAI/logs/metrics_%j.err
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1

source /scratch/users/jfernandezmartinez/miniconda3/etc/profile.d/conda.sh
conda activate GeoAI

echo "Start: $(date)"
cd /scratch/users/jfernandezmartinez/GeoAI
python compute_metrics.py
echo "End: $(date)"
