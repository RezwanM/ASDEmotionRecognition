#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=64GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 2-00:00:00		# Time limit days-hrs:min:sec
#SBATCH -J MLP_Librosa_Complete_Clean_Noise		# Job name

cd /home/r_m727/All_Files/CV_10_Resample/

python3 MLP_Librosa_Complete_Clean_Noise.py > mlp_librosa_complete_clean_noise.txt

exit 0

