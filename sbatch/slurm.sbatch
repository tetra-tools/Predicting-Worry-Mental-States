#!/bin/bash

# Job identification and notification
#SBATCH --job-name=wandb_agent
#SBATCH --mail-user=jpc211@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ihome/hkarim/%u/logs/%x_%j.out
#SBATCH --error=/ihome/hkarim/%u/logs/%x_%j.err

# Resource requests
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --array=1-4
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-20:00:00
#SBATCH --qos=short

module purge
module load python/ondemand-jupyter-python3.10

#conda activate
source activate /ix1/hkarim/envs/module-default_user-jpc211_project-lstmstate2

rm -r ~/.cache/wandb/artifacts/
cd /ix1/hkarim/users/jpc211/projects/DeepNetfMRIWorry

# Your commands here
date
wandb agent campionjp-argo/DeepNetfMRIWorry/kjoar9x6 --count 1
date

# End
echo "Job completed"
crc-job-stats