#!/bin/bash -l
#SBATCH -J CloudGAN_SNPatch_Eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /scratch/users/jfernandezmartinez/CloudGAN/logs/snpatch_eval_%j.out
#SBATCH -e /scratch/users/jfernandezmartinez/CloudGAN/logs/snpatch_eval_%j.err

eval "$(/scratch/users/jfernandezmartinez/miniconda3/bin/conda shell.bash hook)"
conda activate cloudgan

cd /scratch/users/jfernandezmartinez/CloudGAN/

python3 eval_efficiency_SNPatchGAN.py \
    --checkpoint weights/SN_PatchGAN/snap-1132000 \
    --output_dir output/snpatchgan_eval/
