#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J MLP_Parselmouth_Complete_Clean_Conf		# Job name

cd /home/r_m727/All_Files/Confusion_Matrices/MLP/RAVDESS+TESS+CREMA-D/Clean/

python3 MLP_Parselmouth_Complete_Clean_Conf.py > mlp_parselmouth_complete_clean_conf.txt

exit 0

