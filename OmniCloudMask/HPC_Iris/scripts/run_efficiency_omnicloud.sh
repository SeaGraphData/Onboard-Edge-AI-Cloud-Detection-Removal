#!/bin/bash -l
#SBATCH --job-name=omnicloud_eff
#SBATCH --output=/scratch/users/jfernandezmartinez/GeoAI/logs/omnicloud_eff_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/GeoAI/logs/omnicloud_eff_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

source /scratch/users/jfernandezmartinez/miniconda3/etc/profile.d/conda.sh
conda activate GeoAI

echo "========================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
echo "========================================"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

cd /scratch/users/jfernandezmartinez/GeoAI
python eval_efficiency_omnicloud.py

echo "========================================"
echo "End : $(date)"
echo "========================================"
