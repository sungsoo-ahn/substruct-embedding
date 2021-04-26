#!/bin/bash
#SBATCH --job-name=moco
#SBATCH --partition=mbzuai
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding/resource/result/moco.log
#SBATCH -N 1
#SBATCH -G 1

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjY3ZDAyMWYtZWQ5MC00NGQwLTg4OWItN2U3YzU4YWE3YzJkIn0="

srun \
  --container-image=sungsahn0215/substruct-embedding:main \
  --no-container-mount-home \
  --container-mounts="/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding:/substruct-embedding" \
  --container-workdir="/substruct-embedding/src" \
  bash ../script/moco.sh $1 $2