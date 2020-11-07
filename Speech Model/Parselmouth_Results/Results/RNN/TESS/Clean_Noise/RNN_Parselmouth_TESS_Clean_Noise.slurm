#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J RNN_Parselmouth_TESS_Clean_Noise		# Job name

cd /home/r_m727/All_Files/Results/RNN/TESS/Clean_Noise/

python3 RNN_Parselmouth_TESS_Clean_Noise.py > rnn_parselmouth_tess_clean_noise.txt

exit 0

