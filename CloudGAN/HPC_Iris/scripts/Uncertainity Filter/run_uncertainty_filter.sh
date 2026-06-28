#!/bin/bash -l
#SBATCH --job-name=unc_filter
#SBATCH --output=unc_filter_%j.out
#SBATCH --error=unc_filter_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "Uncertainty Filter | Job $SLURM_JOB_ID | $(date)"
source /scratch/users/jfernandezmartinez/miniconda3/etc/profile.d/conda.sh
conda activate cloudgan
cd /scratch/users/jfernandezmartinez/CloudGAN/evaluation
python evaluate_uncertainty_filter.py
echo "Finished: $(date)"
