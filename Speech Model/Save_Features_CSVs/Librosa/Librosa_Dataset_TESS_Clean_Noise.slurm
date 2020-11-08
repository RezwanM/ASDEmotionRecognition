#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 14-00:00:00		# Time limit days-hrs:min:sec
#SBATCH -J Parselmouth_Dataset	# Job name

cd /home/r_m727/All_Files/New/

python3 Librosa_Dataset_TESS_Clean_Noise.py > tess_clean_noise.txt

exit 0

