#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Librosa_Dataset_CREMA-D_Clean.py
# Date: 6/22/20
#
# Objective:
# 62 features - 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast (mean), 2 poly features (mean), and 1 RMS (mean).
#
#*************************************************************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as rosa
import os
import time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import statistics

# Save directory path in 'path'
path = r'/home/r_m727/All_Files/Corpus/Simulated/CREMA-D/All'

# Declare a dummy Numpy array (row vector)
result_array = np.empty([1,63])

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

i=0

for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '/' + file_list[i], sr=16000)
    
    # Calculate the average mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    avg_mfcc_feat = np.mean(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True).T,axis=0)
    
    # Calculate the standard deviation of mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.std' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    std_mfcc_feat = np.std(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True).T,axis=0)
	
    spec_feat = np.mean(rosa.feature.spectral_contrast(y=sig, sr=fs, n_fft=512, hop_length=256).T, axis=0)
    
    poly_feat = np.mean(rosa.feature.poly_features(y=sig, sr=fs, n_fft=512, hop_length=256).T, axis=0)
	
    rms_feat = np.mean(rosa.feature.rms(y=sig, frame_length=512, hop_length=256).T, axis=0)
	
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(avg_mfcc_feat, std_mfcc_feat, axis=0)
    
    feat1 = np.append(feat0, spec_feat, axis=0)
	
    feat2 = np.append(feat1, poly_feat, axis=0)
    
    feat3 = np.append(feat2, rms_feat, axis=0)
	
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '/' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '/' + file_list[i]))[0].split('-')[2]
    
    # Create a new Numpy array 'sample' to store features along with label
    sample = np.insert(feat3, obj=62, values=label)
    
    result_array = np.append(result_array, sample)
    
    i+=1

# Print out the 1D Numpy array
result_array

result_array.shape

# Convert 1D Numpy array to 2D array
result_array = np.reshape(result_array, (i+1,-1))

# Delete first dummy row from 2D array
result_array = np.delete(result_array, 0, 0)

# Print final 2D Numpy array 
print(result_array.shape)

df = pd.DataFrame(data=result_array)
# Label only the last (target) column
df = df.rename({62: "Emotion"}, axis='columns')
# Delete unnecessary emotion data (calm)
df.drop(df[df['Emotion'] == 2.0].index, inplace = True)
df['Emotion'].replace({1.0: "Neutral", 3.0: "Happy", 4.0: "Sad", 5.0: "Angry", 6.0: "Fearful", 7.0: "Disgust", 8.0: "Surprised"}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)

# Save as CSV file
df.to_csv('CREMA-D_Librosa_Clean.csv')

print("Done!")
