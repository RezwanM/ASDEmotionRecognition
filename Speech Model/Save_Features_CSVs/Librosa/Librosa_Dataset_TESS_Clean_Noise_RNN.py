#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Librosa_Dataset_TESS_Clean_Noise_RNN.py
# Date: 6/22/20
#
# Objective:
# 36 features - 26 MFCCs, 7 spectral contrast, 2 poly features, and 1 RMS.
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
path = r'/home/r_m727/All_Files/Corpus/Simulated/TESS/Clean_Noise'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

# Declare an empty list to store the length (no. of frames) for each sample (audio).
num_frames = []

i=0

sum = 0

# Loop for calculating averge number of frames for the dataset.
for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '/' + file_list[i], sr=16000)
    
    # 'rosa.feature.mfcc' extracts n_mfccs from signal and stores it into 'mfcc_feat'
    mfcc_feat = rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True)
    
    num_frames.insert(i, mfcc_feat.shape[1])
    
    i+=1

# Print the list containing the frame lengths of all the samples
num_frames

# Calculate the Median of the number of frames for all samples. This will then be used to cap the maximum number of frames per sample, which in turn will be used as the number of RNN units.
median_num_frames = statistics.median(num_frames)

# Calculate the Mean of the number of frames for all samples. This is just to cross-check with the Median value.
average_num_frames = statistics.mean(num_frames)

# Print the average number of frames for the dataset.
#average_num_frames

# Print the median number of frames for the dataset.
median_num_frames

# Convert float to integer.
median_num_frames = int(median_num_frames)

# Save directory path in 'path'
path = r'/home/r_m727/All_Files/Corpus/Simulated/TESS/Clean_Noise'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

# Declare a dummy Numpy array (row vector)
result_array = np.empty([1, (36*median_num_frames)+1])

i=0

# Loop for feature extraction.
for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '/' + file_list[i], sr=16000)
    
    # 'rosa.feature.mfcc' extracts n_mfccs from signal and stores it into 'mfcc_feat'
    mfcc_feat = rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True)
    
    spec_feat = rosa.feature.spectral_contrast(y=sig, sr=fs, n_fft=512, hop_length=256)
	
    poly_feat = rosa.feature.poly_features(y=sig, sr=fs, n_fft=512, hop_length=256)
	
    rms_feat = rosa.feature.rms(y=sig, frame_length=512, hop_length=256)
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(mfcc_feat, spec_feat, axis=0)
    
    feat1 = np.append(feat0, poly_feat, axis=0)
	
    feat2 = np.append(feat1, rms_feat, axis=0)
    
    # Transpose the array to flip the rows and columns. This is done so that the features become column parameters, making each row an audio frame.
    transp_feat = feat2.T
    
    # Note: The 'cap frame number' is basically the limit we set for the number of frames for each sample, so that all samples have equal dimensions.
    if transp_feat.shape[0]<median_num_frames:

        # If number of frames is smaller than the cap frame number, we pad the array in order to reach our desired dimensions.

        # Pad the array so that it matches the cap frame number. The second value in the argument contains two tuples which indicate which way to pad how much.  
        transp_feat = np.pad(transp_feat, ((0, median_num_frames-transp_feat.shape[0]), (0,0)), constant_values=0)

    elif transp_feat.shape[0]>median_num_frames:

        # If number of frames is larger than the cap frame number, we delete rows (frames) which exceed the cap frame number in order to reach our desired dimensions.

        # Define a tuple which contains the range of the row indices to delete.
        row_del_index = (range(median_num_frames, transp_feat.shape[0], 1))

        transp_feat = np.delete(transp_feat, row_del_index, axis=0)

    else:
        # If number of frames match the cap frame length, perfect!
        transp_feat = transp_feat
    
    # Transpose again to flip the rows and columns. This is done so that the features become row parameters, making each column an audio frame.
    transp2_feat = transp_feat.T
    
    # Flatten the entire 2D Numpy array into 1D Numpy array. So, the first 36 values of the 1D array represent the features for first frame, the second 36 represent the features for second frame, and so on till the final (cap) frame.
    # 'C' means row-major ordered flattening.
    mfcczcr_feat_flatten = transp2_feat.flatten('C')
    
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '/' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '/' + file_list[i]))[0].split('-')[2]
    
    # Create a new Numpy array 'sample' to store features along with label
    sample = np.insert(mfcczcr_feat_flatten, obj=36*median_num_frames, values=label)
    
    result_array = np.append(result_array, sample)
    
    i+=1

# Print out the 1D Numpy array
result_array

result_array.shape

# Convert 1D Numpy array to 2D array. Argument must be a Tuple. i+1 because we have i samples (audio files) plus a dummy row.
result_array = np.reshape(result_array, (i+1,-1))

# Delete first dummy row from 2D array
result_array = np.delete(result_array, 0, 0)

# Print final 2D Numpy array 
print(result_array.shape)

df = pd.DataFrame(data=result_array)
# Label only the last (target) column
df = df.rename({36*median_num_frames: "Emotion"}, axis='columns')
# Delete unnecessary emotion data (calm)
df.drop(df[df['Emotion'] == 2.0].index, inplace = True)
df['Emotion'].replace({1.0: "Neutral", 3.0: "Happy", 4.0: "Sad", 5.0: "Angry", 6.0: "Fearful", 7.0: "Disgust", 8.0: "Surprised"}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)

# Save as CSV file
df.to_csv('TESS_Librosa_Clean_Noise_RNN.csv')

print("Done!")
