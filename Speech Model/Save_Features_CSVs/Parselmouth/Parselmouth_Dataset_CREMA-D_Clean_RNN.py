#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Parselmouth_Dataset_CREMA-D_Clean_RNN.py
# Date: 6/22/20
#
# Objective:
# 15 features - Pitch (f0), Loudness(Intensity), MFCC, formants, formants BW, HNR, Jitter, and Shimmer.
#
#*************************************************************************************

import parselmouth
from parselmouth.praat import call
import glob
import os
import pandas as pd
import numpy as np


# All features together!
# Pitch, Loudness (Intensity), MFCCs, Formants, Jitter, Shimmer, and HNR
def extractFeatures(voiceID, f0min, f0max, unit):
    
    sound = parselmouth.Sound(voiceID) # read the sound
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    # Pitch (f0)
    pitch = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max, time_step=0.016) # 1 value per frame
    pitch = pitch.selected_array['frequency']
    # Convert 1D Numpy array to 2D array. Argument must be a Tuple.
    pitch = np.reshape(pitch, (1,-1))
    
    # Loudness (Intensity)
    loudness = sound.to_intensity(minimum_pitch=f0min, time_step=0.016) # 1 value per frame
    loudness = loudness.as_array()
    
    # MFCC (1-4)
    mfcc = sound.to_mfcc(number_of_coefficients=4, window_length=0.032, time_step=0.016, firstFilterFreqency=100.0, distance_between_filters=100.0) # 4 values per frame
    mfcc = mfcc.to_array()
    
    # Formants (f1, f2, and f3) & their Bandwidths
    formants = sound.to_formant_burg(time_step=0.016, max_number_of_formants=4, maximum_formant=5500.0, window_length=0.032, pre_emphasis_from=50.0)
    numPoints = call(pointProcess, "Get number of points")
    
    file_list = []
    f1_list = []
    f2_list = []
    f3_list = []
    f1_bw_list = []
    f2_bw_list = []
    f3_bw_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_bw = call(formants, "Get bandwidth at time", 1, t, 'Hertz', 'Linear')
        f2_bw = call(formants, "Get bandwidth at time", 2, t, 'Hertz', 'Linear')
        f3_bw = call(formants, "Get bandwidth at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f1_bw_list.append(f1_bw)
        f2_bw_list.append(f2_bw)
        f3_bw_list.append(f3_bw)
        
    
    f1 = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2 = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3 = [f3 for f3 in f3_list if str(f3) != 'nan']
    f1_bw = [f1_bw for f1_bw in f1_bw_list if str(f1_bw) != 'nan']
    f2_bw = [f2_bw for f2_bw in f2_bw_list if str(f2_bw) != 'nan']
    f3_bw = [f3_bw for f3_bw in f3_bw_list if str(f3_bw) != 'nan']
    
    # Jitter and Shimmer
    jitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.032, 1.3) # local absolute jitter
    shimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.032, 1.3, 1.6) # local dB shimmer
    
    # HNR
    hnr = sound.to_harmonicity_cc(time_step=0.016, minimum_pitch=f0min, silence_threshold=0.1, periods_per_window=4.5)
    hnr = hnr.as_array()
    
    return pitch, loudness, mfcc, f1, f2, f3, f1_bw, f2_bw, f3_bw, jitter, shimmer, hnr

# Define the median number of frames for the Complete dataset
median_num_frames = 149

# Declare a dummy Numpy array (row vector)
result_array = np.empty([1, (15*median_num_frames)+1]) # 15 features and 1 label

# Save directory path in 'path'
path = r'/home/r_m727/All_Files/Corpus/Simulated/CREMA-D/All'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

i=0

# Go through all the wave files in the folder and measure pitch
for filename in file_list:

    sound = parselmouth.Sound(path + '/' + file_list[i])
    sound = sound.resample(new_frequency=16000)
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '/' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '/' + file_list[i]))[0].split('-')[2]

    # Get audio and apply feature extraction
    # Pitch floor determines frame length. Frame length = 3/Pitch floor = 3/94 = 0.032 seconds
    (pitch, loudness, mfcc, f1, f2, f3, f1_bw, f2_bw, f3_bw, jitter, shimmer, hnr) = extractFeatures(sound, 94, 600, "LogHertz") # pitch floor, pitch ceiling, unit

    # Pitch (f0)

    # Note: The 'cap frame number' is basically the limit we set for the number of frames for each sample, so that all samples have equal dimensions.
    if pitch.shape[1]<median_num_frames:
        # If number of frames is smaller than the cap frame number, we pad the array in order to reach our desired dimensions.
        # Pad the array so that it matches the cap frame number. The second value in the argument contains two tuples which indicate which way to pad how much.
        pitch = np.pad(pitch, ((0,0), (0,median_num_frames-pitch.shape[1])), constant_values=0)
    elif pitch.shape[1]>median_num_frames:
        # If number of frames is larger than the cap frame number, we delete columns (frames) which exceed the cap frame number in order to reach our desired dimensions.
        # Define a tuple which contains the range of the column indices to delete.
        col_del_index = range(median_num_frames, pitch.shape[1], 1)
        pitch = np.delete(pitch, col_del_index, axis=1)
    else:
        # If number of frames match the cap frame length, perfect!
        pitch = pitch

    # Loudness (Intensity)
    
    if loudness.shape[1]<median_num_frames:
        loudness = np.pad(loudness, ((0,0), (0,median_num_frames-loudness.shape[1])), constant_values=0)
    elif loudness.shape[1]>median_num_frames:
        col_del_index = range(median_num_frames, loudness.shape[1], 1)
        loudness = np.delete(loudness, col_del_index, axis=1)
    else:
        loudness = loudness

    # MFCC (1-4)
    
    if mfcc.shape[1]<median_num_frames:
        mfcc = np.pad(mfcc, ((0,0), (0,median_num_frames-mfcc.shape[1])), constant_values=0)
    elif mfcc.shape[1]>median_num_frames:
        col_del_index = range(median_num_frames, mfcc.shape[1], 1)
        mfcc = np.delete(mfcc, col_del_index, axis=1)
    else:
        mfcc = mfcc


    mfcc = np.delete(mfcc, 0, 0) # Delete first row containing MFCC-0

    # Formants (f1, f2, and f3) & their Bandwidths

    f1 = np.asarray(f1) # Convert Lists into Numpy arrays
    f2 = np.asarray(f2)
    f3 = np.asarray(f3)
    f1_bw = np.asarray(f1_bw)
    f2_bw = np.asarray(f2_bw)
    f3_bw = np.asarray(f3_bw)

    f1 = np.reshape(f1, (1,-1)) # Reshape 1-D arrays into 2-D (rows x columns)
    f2 = np.reshape(f2, (1,-1))
    f3 = np.reshape(f3, (1,-1))
    f1_bw = np.reshape(f1_bw, (1,-1))
    f2_bw = np.reshape(f2_bw, (1,-1))
    f3_bw = np.reshape(f3_bw, (1,-1))

    f1_bw = np.vstack((f1,f1_bw)) # Vstack formants with their BW frequencies
    f2_bw = np.vstack((f2,f2_bw))
    f3_bw = np.vstack((f3,f3_bw))

    if f1_bw.shape[1]<median_num_frames:
        f1_bw = np.pad(f1_bw, ((0,0), (0,median_num_frames-f1_bw.shape[1])), constant_values=0)
    elif f1_bw.shape[1]>median_num_frames:
        col_del_index = range(median_num_frames, f1_bw.shape[1], 1)
        f1_bw = np.delete(f1_bw, col_del_index, axis=1)
    else:
        f1_bw = f1_bw

    if f2_bw.shape[1]<median_num_frames:
        f2_bw = np.pad(f2_bw, ((0,0), (0,median_num_frames-f2_bw.shape[1])), constant_values=0)
    elif f2_bw.shape[1]>median_num_frames:
        col_del_index = range(median_num_frames, f2_bw.shape[1], 1)
        f2_bw = np.delete(f2_bw, col_del_index, axis=1)
    else:
        f2_bw = f2_bw    

    if f3_bw.shape[1]<median_num_frames:
        f3_bw = np.pad(f3_bw, ((0,0), (0,median_num_frames-f3_bw.shape[1])), constant_values=0)
    elif f3_bw.shape[1]>median_num_frames:
        col_del_index = range(median_num_frames, f3_bw.shape[1], 1)
        f3_bw = np.delete(f3_bw, col_del_index, axis=1)
    else:
        f3_bw = f3_bw

    # Jitter and Shimmer
    jitter_array = median_num_frames*[jitter] # Copy and paste same value of Jitter through entire frame length
    jitter = np.asarray(jitter_array) # Convert List to Numpy array
    jitter = np.reshape(jitter, (1,-1)) # Reshape 1-D arrays into 2-D (rows x columns)

    shimmer_array = median_num_frames*[shimmer]
    shimmer = np.asarray(shimmer_array)
    shimmer = np.reshape(shimmer, (1,-1))
    
    # HNR
    
    if hnr.shape[1]<median_num_frames:
        hnr = np.pad(hnr, ((0,0), (0,median_num_frames-hnr.shape[1])), constant_values=0)
    elif hnr.shape[1]>median_num_frames:
        col_del_index = range(median_num_frames, hnr.shape[1], 1)
        hnr = np.delete(hnr, col_del_index, axis=1)
    else:
        hnr = hnr
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(pitch, loudness, axis=0)
    
    feat1 = np.append(feat0, mfcc, axis=0)
	
    feat2 = np.append(feat1, f1_bw, axis=0)
    
    feat3 = np.append(feat2, f2_bw, axis=0)
    
    feat4 = np.append(feat3, f3_bw, axis=0)
    
    feat5 = np.append(feat4, jitter, axis=0)
    
    feat6 = np.append(feat5, shimmer, axis=0)
    
    feat7 = np.append(feat6, hnr, axis=0)
    
    # Flatten the entire 2D Numpy array into 1D Numpy array. So, the first 15 values of the 1D array represent the features for first frame, the second 15 represent the features for second frame, and so on till the final (cap) frame.
    # 'C' means row-major ordered flattening.
    features_flat = feat7.flatten('C')
    
    # Append label
    features_label = np.insert(features_flat, obj=15*median_num_frames, values=label)
    result_array = np.append(result_array, features_label)

    i+=1

# Convert 1D Numpy array to 2D array. Argument must be a Tuple. i+1 because we have i samples (audio files) plus a dummy row.
result_array = np.reshape(result_array, (i+1,-1))

# Delete first dummy row from 2D array
result_array = np.delete(result_array, 0, 0)

# Arrange all feature lists into a Pandas dataframe
# Save data into a Pandas dataframe
df = pd.DataFrame(result_array)

# Label only the last (target) column
df = df.rename({15*median_num_frames: "Emotion"}, axis='columns')
# Delete unnecessary emotion data (calm)
df.drop(df[df['Emotion'] == 2.0].index, inplace = True)
df['Emotion'].replace({1.0: "Neutral", 3.0: "Happy", 4.0: "Sad", 5.0: "Angry", 6.0: "Fearful", 7.0: "Disgust", 8.0: "Surprised"}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)

df.to_csv('CREMA-D_Parselmouth_Clean_RNN.csv')

print("Done!")
