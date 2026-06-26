#!/bin/bash -l
#SBATCH -J CloudGAN_Train_AE
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /scratch/users/jfernandezmartinez/CloudGAN/output/AE_%j.out
#SBATCH -e /scratch/users/jfernandezmartinez/CloudGAN/output/AE_%j.err

# Activate conda — use full path to miniconda on scratch, not ~/miniconda3/
eval "$(/scratch/users/jfernandezmartinez/miniconda3/bin/conda shell.bash hook)"
conda activate cloudgan

cd /scratch/users/jfernandezmartinez/CloudGAN/
python3 cloud_detection/networks/main.py --model AE

echo "Done!"
