#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J RNN_Librosa_CREMA-D_Clean_Noise_Conf		# Job name

cd /home/r_m727/All_Files/Confusion_Matrices/RNN/CREMA-D/Clean_Noise/

python3 RNN_Librosa_CREMA-D_Clean_Noise_Conf.py > rnn_librosa_crema-d_clean_noise_conf.txt

exit 0

