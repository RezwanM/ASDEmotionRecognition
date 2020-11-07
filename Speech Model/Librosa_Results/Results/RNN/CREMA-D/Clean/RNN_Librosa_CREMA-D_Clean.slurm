#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J RNN_Librosa_CREMA-D_Clean		# Job name

cd /home/r_m727/All_Files/Results/RNN/CREMA-D/Clean/

python3 RNN_Librosa_CREMA-D_Clean.py > rnn_librosa_crema-d_clean.txt

exit 0

