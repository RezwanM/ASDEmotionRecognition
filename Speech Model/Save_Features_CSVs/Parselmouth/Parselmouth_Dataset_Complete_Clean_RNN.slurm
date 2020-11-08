#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 14-00:00:00		# Time limit days-hrs:min:sec
#SBATCH -J Com_C_R	# Job name

cd /home/r_m727/All_Files/New/

python3 Parselmouth_Dataset_Complete_Clean_RNN.py > complete_clean_rnn.txt

exit 0

