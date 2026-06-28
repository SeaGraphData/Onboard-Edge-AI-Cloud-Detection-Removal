#!/bin/bash -l
#SBATCH -J CloudGAN_Eval_UNet
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /scratch/users/jfernandezmartinez/CloudGAN/evaluation/fullUNet_%j.out
#SBATCH -e /scratch/users/jfernandezmartinez/CloudGAN/evaluation/fullUNet_%j.err

eval "$(/scratch/users/jfernandezmartinez/miniconda3/bin/conda shell.bash hook)"
conda activate cloudgan

cd /scratch/users/jfernandezmartinez/CloudGAN/evaluation/

python3 evaluate_fullUNet.py

echo "Done!"