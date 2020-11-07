#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J SVM_Parselmouth_Complete_Clean_Noise		# Job name

cd /home/r_m727/All_Files/SVM_Time/

python3 SVM_Parselmouth_Complete_Clean_Noise.py > svm_parselmouth_complete_clean_noise.txt

exit 0

