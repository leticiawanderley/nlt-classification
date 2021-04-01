#!/bin/bash
#SBATCH --mail-user=fariaswa@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-cepp
#SBATCH --array=1-2046
#SBATCH --time=0:30:00
#SBATCH --mem=2048M
module load python/3.7
source env/bin/activate
python train_classification_model.py -f features/input.$SLURM_ARRAY_TASK_ID