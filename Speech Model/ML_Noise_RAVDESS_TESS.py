#************************************************************************************
# Rezwan Matin
# Speech Emotion Recognition using SVM and RAVDESS+TESS corpora
# Filename: ML_Noise_RAVDESS_TESS.py
# Date: 01/26/20
#
# Objective:
# Add background sound (city center noise) to RAVDESS and TESS audio recordings.
#
#*************************************************************************************

import scipy
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as rosa
import glob
import os
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

# RAVDESS noise addition

# Save directory path in 'path'
path1 = r'C:\Books\Texas State Books\Fall 2019\Thesis A\Corpus\Simulated\RAVDESS\All'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path1)

i=0

for filename in file_list:
    
    file_name = file_list[i]
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs1' and audio signal in 'sig1'
    sig1, fs1 = rosa.core.load(path1 + '\\' + file_list[i], sr=None)
    
    # Store background track file path into 'path2'
    path2 = 'C:/Users/Maleeha/Documents/City_Centre-Hopeinawe-377331566.wav'
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs2' and audio signal in 'sig2'
    sig2, fs2 = rosa.core.load(path2, sr=48000)
    
    # Resize 6 seconds background track to size of RAVDESS file
    sig2.resize((sig1.shape[0]))
    
    # Mix (combine) the two audio files
    sig3 = sig1 + sig2
    
    # Export the combined audio track as .wav file
    #scipy.io.wavfile.write(file_name, 48000, sig3)

    # Write out audio as 24bit PCM WAV
    sf.write(file_name, sig3, 48000, subtype='PCM_16')
    i+=1

#*************************************************************#

# TESS noise addition

# Save directory path in 'path'
path1 = r'C:\Books\Texas State Books\Fall 2019\Thesis A\Corpus\Simulated\TESS\All_Renamed_RAVDESS_Style_Background Noise'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path1)

i=0

for filename in file_list:
    
    file_name = file_list[i]
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs1' and audio signal in 'sig1'
    sig1, fs1 = rosa.core.load(path1 + '\\' + file_list[i], sr=None)
    
    # Store background track file path into 'path2'
    path2 = 'C:/Users/Maleeha/Documents/City_Centre-Hopeinawe-377331566.wav'
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs2' and audio signal in 'sig2'
    sig2, fs2 = rosa.core.load(path2, sr=24414)
    
    # Resize 6 seconds background track to size of TESS file
    sig2.resize((sig1.shape[0]))
    
    # Mix (combine) the two audio files
    sig3 = sig1 + sig2
    
    # Export the combined audio track as .wav file
    #scipy.io.wavfile.write(file_name, 48000, sig3)

    # Write out audio as 24bit PCM WAV
    sf.write(file_name, sig3, 24414, subtype='PCM_16')
    i+=1