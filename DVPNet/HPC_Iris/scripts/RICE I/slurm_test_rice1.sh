#!/bin/bash -l
#SBATCH --job-name=DVPNet_rice1_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/jfernandezmartinez/DVPNet/logs/dvpnet_rice1_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/DVPNet/logs/dvpnet_rice1_%j.err

source ~/.bashrc
conda activate DVPNet

cd /scratch/users/jfernandezmartinez/DVPNet

python eval_dvpnet_full.py \
  --opt option/rice1-DVPNet.yml \
  --weights experiments/pretrained_models/pretrained_models/rice1/net_g_best.pth \
  --input_dir datasets/RICE_DATASET/RICE1/cloud \
  --input_truth_dir datasets/RICE_DATASET/RICE1/label \
  --result_dir output/rice1-eval