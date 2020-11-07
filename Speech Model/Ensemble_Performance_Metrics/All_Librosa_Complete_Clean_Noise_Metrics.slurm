#!/bin/bash -l

#SBATCH -p shared		# Check partitions on the Wiki
#SBATCH --mem=12GB		# Memory size
#SBATCH -n 1			# Number of processors in the node
#SBATCH -t 02:00:00		# Time limit days-hrs:min:sec
#SBATCH -J All_Librosa_Complete_Clean_Noise_Metrics		# Job name

cd /home/r_m727/All_Files/Dep_Metrics_Resample/

python3 All_Librosa_Complete_Clean_Noise_Metrics.py > all_librosa_complete_clean_noise_metrics.txt

exit 0

