#!/bin/bash
#SBATCH --job-name=node
#SBATCH --partition=mbzuai
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding/resource/result/node.log
#SBATCH -N 1
#SBATCH -G 1

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjY3ZDAyMWYtZWQ5MC00NGQwLTg4OWItN2U3YzU4YWE3YzJkIn0="

srun \
  --container-image=sungsahn0215/substruct-embedding:torch110 \
  --no-container-mount-home \
  --container-mounts="/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding:/substruct-embedding" \
  --container-workdir="/substruct-embedding/src" \
  bash ../script/node_contrast${1}.sh