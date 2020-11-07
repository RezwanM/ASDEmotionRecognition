#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J MLP_Parselmouth_RAVDESS_Clean_Noise		# Job name

cd /home/r_m727/All_Files/Results/MLP/RAVDESS/Clean_Noise/

python3 MLP_Parselmouth_RAVDESS_Clean_Noise.py > mlp_parselmouth_ravdess_clean_noise.txt

exit 0

