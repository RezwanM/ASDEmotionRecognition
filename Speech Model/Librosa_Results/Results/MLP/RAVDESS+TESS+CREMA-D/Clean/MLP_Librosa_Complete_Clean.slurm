#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J MLP_Librosa_Complete_Clean		# Job name

cd /home/r_m727/All_Files/Results/MLP/RAVDESS+TESS+CREMA-D/Clean/

python3 MLP_Librosa_Complete_Clean.py > mlp_librosa_complete_clean.txt

exit 0

