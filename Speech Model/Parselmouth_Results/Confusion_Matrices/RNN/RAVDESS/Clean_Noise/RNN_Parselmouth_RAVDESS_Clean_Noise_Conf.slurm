#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J RNN_Parselmouth_RAVDESS_Clean_Noise_Conf		# Job name

cd /home/r_m727/All_Files/Confusion_Matrices/RNN/RAVDESS/Clean_Noise/

python3 RNN_Parselmouth_RAVDESS_Clean_Noise_Conf.py > rnn_parselmouth_ravdess_clean_noise_conf.txt

exit 0

