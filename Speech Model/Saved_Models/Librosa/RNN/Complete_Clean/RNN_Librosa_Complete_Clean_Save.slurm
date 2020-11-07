#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J RNN_Librosa_Complete_Clean_Save		# Job name

cd /home/r_m727/All_Files/Deploy_Resample_2/

python3 RNN_Librosa_Complete_Clean_Save.py > rnn_librosa_complete_clean_save.txt

exit 0

