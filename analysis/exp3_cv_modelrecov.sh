#!/bin/bash

#SBATCH -J ShuzeTest                      # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liushuze@login.rc.fas.harvard.edu   # Where to send mail
#SBATCH --ntasks=1                                  # Run a single task, defaults to single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=70:00:00                          # Time limit hrs:min:sec
#SBATCH --array=0-63
#SBATCH -o ./learning_curves/sh_files/modelrecov."%j"_"%a".out                            # Standard output to current dir
#SBATCH -e ./learning_curves/sh_files/modelrecov."%j"_"%a".err                             # Error output to current dir
 

# Enable Additional Software
module load python/3.10.9-fasrc01

# Run the job commands
conda run -n pyro_env_new2 python ./learning_curves/exp3_cv_modelrecov.py --n_folds=5 --cv_seed=0 --fakedataset_modelfullname_combidx=$SLURM_ARRAY_TASK_ID