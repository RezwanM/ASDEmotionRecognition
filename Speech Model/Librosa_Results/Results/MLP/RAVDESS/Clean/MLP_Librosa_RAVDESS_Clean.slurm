#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J MLP_Librosa_RAVDESS_Clean		# Job name

cd /home/r_m727/All_Files/Results/MLP/RAVDESS/Clean/

python3 MLP_Librosa_RAVDESS_Clean.py > mlp_librosa_ravdess_clean.txt

exit 0

