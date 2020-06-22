#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Best_RNN.py
# Date: 6/22/20
#
# Objective:
# 26 MFCCs (mean) and 26 MFCCs (standard deviation), ZCR; Transpose X_Train Tensor so that row=features and column=frames.
#
#*************************************************************************************

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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import tensorflow.keras as keras
import statistics

# Save directory path in 'path'
path = r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS/All'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

# Declare an empty list to store the length (no. of frames) for each sample (audio).
num_frames = []

i=0

sum = 0

# Loop for calculating averge number of frames for the dataset.
for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '/' + file_list[i], sr=None)
    
    # 'rosa.feature.mfcc' extracts n_mfccs from signal and stores it into 'mfcc_feat'
    mfcc_feat = rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26)
    
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
path = r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS/All'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

# Declare a dummy Numpy array (row vector)
result_array = np.empty([1, (27*median_num_frames)+1])

i=0

# Loop for feature extraction.
for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '/' + file_list[i], sr=None)
    
    # 'rosa.feature.mfcc' extracts n_mfccs from signal and stores it into 'mfcc_feat'
    mfcc_feat = rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26)
    
    # Calculate the average zero crossing rate (utterance-level feature) using 'rosa.feat.zero_crossing_rate()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    zcross_feat = rosa.feature.zero_crossing_rate(sig)
    
    # Append the two 2D arrays into a single 2D array called 'mfcczcr_feat'.
    mfcczcr_feat = np.append(mfcc_feat, zcross_feat, axis=0)
    
    # Transpose the array to flip the rows and columns. This is done so that the features become column parameters, making each row an audio frame.
    transp_mfcczcr_feat = mfcczcr_feat.T
    
    # Note: The 'cap frame number' is basically the limit we set for the number of frames for each sample, so that all samples have equal dimensions.
    if transp_mfcczcr_feat.shape[0]<median_num_frames:

        # If number of frames is smaller than the cap frame number, we pad the array in order to reach our desired dimensions.

        # Pad the array so that it matches the cap frame number. The second value in the argument contains two tuples which indicate which way to pad how much.  
        transp_mfcczcr_feat = np.pad(transp_mfcczcr_feat, ((0, median_num_frames-transp_mfcczcr_feat.shape[0]), (0,0)), 'mean')

    elif transp_mfcczcr_feat.shape[0]>median_num_frames:

        # If number of frames is larger than the cap frame number, we delete rows (frames) which exceed the cap frame number in order to reach our desired dimensions.

        # Define a tuple which contains the range of the row indices to delete.
        row_del_index = (range(median_num_frames, transp_mfcczcr_feat.shape[0], 1))

        transp_mfcczcr_feat = np.delete(transp_mfcczcr_feat, row_del_index, axis=0)

    else:
        # If number of frames match the cap frame length, perfect!
        transp_mfcczcr_feat = transp_mfcczcr_feat
    
    # Transpose again to flip the rows and columns. This is done so that the features become row parameters, making each column an audio frame.
    transp2_mfcczcr_feat = transp_mfcczcr_feat.T
    
    # Flatten the entire 2D Numpy array into 1D Numpy array. So, the first 27 values of the 1D array represent the 26 MFCC and ZCR features for first frame, the second 27 represent the features for second frame, and so on till the final (cap) frame.
    # 'C' means row-major ordered flattening.
    mfcczcr_feat_flatten = transp2_mfcczcr_feat.flatten('C')
    
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '/' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '/' + file_list[i]))[0].split('-')[2]
    
    # Create a new Numpy array 'sample' to store features along with label
    sample = np.insert(mfcczcr_feat_flatten, obj=27*median_num_frames, values=label)
    
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
result_array.shape

df = pd.DataFrame(data=result_array)
# Label only the last (target) column
df = df.rename({27*median_num_frames: "Emotion"}, axis='columns')
# Delete unnecessary emotion data (calm)
df.drop(df[df['Emotion'] == 2.0].index, inplace = True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)
df.tail(12)

df['Emotion'].value_counts()

# Balance the dataset for equal number of samples for each class.
# Separate majority and minority classes
df_minority = df[df.Emotion==1.0]
df_majority3 = df[df.Emotion==3.0]
df_majority4 = df[df.Emotion==4.0]
df_majority5 = df[df.Emotion==5.0]
df_majority6 = df[df.Emotion==6.0]
df_majority7 = df[df.Emotion==7.0]
df_majority8 = df[df.Emotion==8.0]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=192,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority3, df_majority4, df_majority5, df_majority6, df_majority7, df_majority8])
 
# Display new class counts
df_upsampled.Emotion.value_counts()

# Reset row (sample) indexing
df_upsampled = df_upsampled.reset_index(drop=True)

df_upsampled['Emotion'].value_counts()

# Extract target feature 'Emotion' in a vector y. Indexing from 0
y = df_upsampled.iloc[0:1344, 27*median_num_frames].values
# Extract features 'buying' and 'safety' in a vector X. Indexing from 0
X = df_upsampled.iloc[0:1344, list(range(27*median_num_frames))].values
print(y)

# Split data for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

# Standardize the inputs
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

np.random.seed(123)

tf.random.set_seed(123)

# One-Hot Encode the classes
y_train_onehot = keras.utils.to_categorical(y_train)

# Reshaping X_train and X_test to 3D Numpy arrays for feeding into the RNN. RNNs require 3D array input.
# 3D dimensions are (layers, rows, columns).
X_train_3D = np.reshape(X_train_centered, (X_train_centered.shape[0], median_num_frames, 27))
X_test_3D = np.reshape(X_test_centered, (X_test_centered.shape[0], median_num_frames, 27))

print(X_train_3D.shape, y_train.shape)
print(X_test_3D.shape, y_test.shape)

# Transpose tensors so that rows=features and columns=frames.
X_train_3D_posed = tf.transpose(X_train_3D, perm=[0, 2, 1])
X_test_3D_posed = tf.transpose(X_test_3D, perm=[0, 2, 1])

print(X_train_3D_posed.shape, y_train.shape)
print(X_test_3D_posed.shape, y_test.shape)

# Create an object/instance 'model' for the 'Sequential()' class.
model = keras.models.Sequential()
    
model.add(
    keras.layers.LSTM( units=27,
                input_shape=(27, median_num_frames),
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softsign'))

model.add(
    keras.layers.Dense( units=50,
                input_dim=27,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softsign'))

model.add(
    keras.layers.Dense( units=y_train_onehot.shape[1],
                input_dim=50,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softmax'))

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)

adam_optimizer = keras.optimizers.Adam(
                    learning_rate=lrate, beta_1=0.8, beta_2=0.999, epsilon=1e-06) #1e-06 gave better result than default value 1e-07

model.compile(optimizer=adam_optimizer,
                    loss='kullback_leibler_divergence')
                            
# Train the RNN
history = model.fit(X_train_3D_posed, y_train_onehot, batch_size=16, epochs=200, verbose=1, validation_split=0.1) # 90% training / 10% validation

y_train_pred = model.predict_classes(X_train_3D_posed, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test_3D_posed, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

print('Test accuracy: %.2f%%' % (test_acc * 100))
