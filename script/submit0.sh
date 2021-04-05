#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=mbzuai
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding/src/test.log
#SBATCH -N 1
#SBATCH -G 1

srun \
  --container-image=sungsahn0215/substruct-embedding:main \
  --no-container-mount-home \
  --container-mounts="/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding:/substruct-embedding" \
  --container-workdir="/substruct-embedding/src" \
  python3 /substruct-embedding/src/finetune.py --model_path "../resource/result/REL-66/model.pt"