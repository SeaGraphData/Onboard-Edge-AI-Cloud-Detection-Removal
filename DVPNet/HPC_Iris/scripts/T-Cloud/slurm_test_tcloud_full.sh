#!/bin/bash -l
#SBATCH --job-name=DVPNet_tcloud_full_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/jfernandezmartinez/DVPNet/logs/dvpnet_tcloud_full_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/DVPNet/logs/dvpnet_tcloud_full_%j.err

source ~/.bashrc
conda activate DVPNet

cd /scratch/users/jfernandezmartinez/DVPNet

python eval_dvpnet_tcloud_full.py \
  --opt option/T-cloud-DVPNet.yml \
  --weights experiments/pretrained_models/pretrained_models/t-cloud/net_g_best.pth \
  --input_dir /scratch/users/jfernandezmartinez/DVPNet/datasets/T-Cloud/T-Cloud/test/cloud \
  --input_truth_dir /scratch/users/jfernandezmartinez/DVPNet/datasets/T-Cloud/T-Cloud/test/reference \
  --result_dir output/tcloud-full-eval