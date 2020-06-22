#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Best_MLP.py
# Date: 6/22/20
#
# Objective:
# 26 MFCCs (mean) and 26 MFCCs (standard deviation), ZCR for BEST Adam so far!
#
#*************************************************************************************

import pandas as pd
import math
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
from tensorflow.keras.callbacks import LearningRateScheduler

# Save directory path in 'path'
path = r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS/All'

# Declare a dummy Numpy array (row vector)
result_array = np.empty([1,54])

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

i=0

for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '/' + file_list[i], sr=None)
    
    # Calculate the average mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    avg_mfcc_feat = np.mean(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26).T,axis=0)
    
    # Calculate the standard deviation of mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.std' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    std_mfcc_feat = np.std(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26).T,axis=0)
    
    # Calculate the average zero crossing rate (utterance-level feature) using 'rosa.feat.zero_crossing_rate()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    zcross_feat = rosa.feature.zero_crossing_rate(sig)
    avg_zcross_feat = np.mean(rosa.feature.zero_crossing_rate(y=sig).T,axis=0)
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(avg_mfcc_feat, std_mfcc_feat, axis=0)
    
    feat1 = np.append(feat0, avg_zcross_feat, axis=0)
    
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '\\' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '\\' + file_list[i]))[0].split('-')[2]
    
    # Create a new Numpy array 'sample' to store features along with label
    sample = np.insert(feat1, obj=53, values=label)
    
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
result_array.shape

df = pd.DataFrame(data=result_array)
# Label only the last (target) column
df = df.rename({53: "Emotion"}, axis='columns')
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
y = df_upsampled.iloc[0:1344, 53].values
# Extract features 'buying' and 'safety' in a vector X. Indexing from 0
X = df_upsampled.iloc[0:1344, list(range(53))].values
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

# Create an object/instance 'model' for the 'Sequential()' class.
model = keras.models.Sequential()
model.add(
    keras.layers.Dense( units=53,
                input_dim=X_train_centered.shape[1],
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', 
                activation='selu'))
model.add(
    keras.layers.Dense( units=45,
                input_dim=53,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='selu'))

model.add(
    keras.layers.Dense( units=54,
                input_dim=45,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='selu'))

model.add(
    keras.layers.Dense( units=y_train_onehot.shape[1],
                input_dim=54,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softmax'))

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)

adam_optimizer = keras.optimizers.Adam(
                    learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-06) #1e-06 gave better result than default value 1e-07

model.compile(optimizer=adam_optimizer,
                    loss='kullback_leibler_divergence')
                          
                            # cross-entropy: fancy name for logistic regression                        

# Train the MLP
history = model.fit(X_train_centered, y_train_onehot, batch_size=16, epochs=200, verbose=1, validation_split=0.1) # 90% training / 10% validation

y_train_pred = model.predict_classes(X_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

print('Test accuracy: %.2f%%' % (test_acc * 100))