#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J MLP_Parselmouth_Complete_Clean_Noise		# Job name

cd /home/r_m727/All_Files/Results/MLP/RAVDESS+TESS+CREMA-D/Clean_Noise/

python3 MLP_Parselmouth_Complete_Clean_Noise.py > mlp_parselmouth_complete_clean_noise.txt

exit 0

