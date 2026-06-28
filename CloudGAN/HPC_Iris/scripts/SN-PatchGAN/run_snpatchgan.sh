#!/bin/bash -l
#SBATCH -J SNGAN_Eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /scratch/users/jfernandezmartinez/CloudGAN/evaluation/sngan_%j.out
#SBATCH -e /scratch/users/jfernandezmartinez/CloudGAN/evaluation/sngan_%j.err

eval "$(/scratch/users/jfernandezmartinez/miniconda3/bin/conda shell.bash hook)"
conda activate cloudgan

cd /scratch/users/jfernandezmartinez/CloudGAN/evaluation/
python3 evaluate_snpatchgan.py

echo "SN-PatchGAN evaluation done!"