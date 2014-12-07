#!/bin/bash
#SBATCH -n 1               # (Max) number of tasks per job, for R usually 1
#SBATCH -o dump/out_xu_section6_1_%a.txt # File for the standard output
#SBATCH -e dump/err_xu_section6_1_%a.txt # File for the standard error
#SBATCH -p stats           # Partition to use
#SBATCH --mem-per-cpu=4096 # Memory required per CPU, in MegaBytes
#SBATCH -a 1-4             # Array of n jobs
#SBATCH -t 10080           # Set time limit to be maximum (7 days)
#SBATCH --mail-user=dtran@g.harvard.edu # Where to send mail
#SBATCH --mail-type=ALL    # When to send mail

# This file runs the simulation code for a given set of simulations.

Rscript xu_section6_1.R $SLURM_ARRAY_TASK_ID

# vim:filetype=sh
