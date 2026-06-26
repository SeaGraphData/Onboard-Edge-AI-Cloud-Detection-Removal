#!/bin/bash -l
#SBATCH -J CloudGAN_UNet
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o /scratch/users/jfernandezmartinez/CloudGAN/logs/UNet_%j.out
#SBATCH -e /scratch/users/jfernandezmartinez/CloudGAN/logs/UNet_%j.err

eval "$(/scratch/users/jfernandezmartinez/miniconda3/bin/conda shell.bash hook)"
conda activate cloudgan

mkdir -p /scratch/users/jfernandezmartinez/CloudGAN/logs

cd /scratch/users/jfernandezmartinez/CloudGAN/cloud_detection/networks/

# NOTE: --model argument is case-sensitive. Must be 'UNET' (uppercase), not 'UNet'.
# 'UNet' produces: error: argument --model: invalid choice: 'UNet' (choose from 'AE', 'UNET')
python3 main.py \
    --model UNET \
    --epochs 64 \
    --batch_size 16 \
    --augmentation \
    --save_dir /scratch/users/jfernandezmartinez/CloudGAN/output/UNet
