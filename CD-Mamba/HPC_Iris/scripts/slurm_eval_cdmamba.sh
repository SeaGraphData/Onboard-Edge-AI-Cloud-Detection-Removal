#!/bin/bash -l
#SBATCH -J CDMamba_Eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_eval_%j.out
#SBATCH -e /scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_eval_%j.err

eval "$(/scratch/users/jfernandezmartinez/miniconda3/bin/conda shell.bash hook)"
conda activate cdmamba

mkdir -p /scratch/users/jfernandezmartinez/CDMamba/logs

cd /scratch/users/jfernandezmartinez/CDMamba/

# Evaluate all four folds sequentially in a single job.
# Output: cdmamba_metrics_<timestamp>.json
python eval_cdmamba_iris.py \
    --checkpoint_dir pt_models/ \
    --data_root CDMamba_patches/ \
    --output_dir output/cdmamba_eval/
