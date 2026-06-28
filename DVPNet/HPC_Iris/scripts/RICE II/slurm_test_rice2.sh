#!/bin/bash -l
#SBATCH --job-name=DVPNet_rice2_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/jfernandezmartinez/DVPNet/logs/dvpnet_rice2_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/DVPNet/logs/dvpnet_rice2_%j.err

source ~/.bashrc
conda activate DVPNet

cd /scratch/users/jfernandezmartinez/DVPNet

python eval_dvpnet_full.py \
  --opt option/rice2-DVPNet.yml \
  --weights experiments/pretrained_models/pretrained_models/rice2/net_g_best.pth \
  --input_dir datasets/RICE_DATASET/RICE2/cloud \
  --input_truth_dir datasets/RICE_DATASET/RICE2/label \
  --result_dir output/rice2-eval