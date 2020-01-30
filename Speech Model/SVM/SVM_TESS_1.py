#************************************************************************************
# Rezwan Matin
#  Speech Emotion Recognition using SVM and the TESS corpus
# Filename: SVM_TESS_1.py
# Date: 01/28/20
#
# Objective:
# 13 MFCCs (mean) and 13 MFCCs (standard deviation), ZCR.
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

# Save directory path in 'path'
path = r'C:\Books\Texas State Books\Fall 2019\Thesis A\Corpus\Simulated\TESS\All'

# Declare a dummy Numpy array (row vector)
result_array = np.empty([1,28])

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

i=0

for filename in file_list:
    
    # Read WAV file. 'rosa.core.load' returns sampling frequency in 'fs' and audio signal in 'sig'
    sig, fs = rosa.core.load(path + '\\' + file_list[i], sr=None)
    
    # 'rosa.feature.mfcc' extracts n_mfccs from signal and stores it into 'mfcc_feat'
    mfcc_feat = rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=13)
    
    # Calculate the average mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    avg_mfcc_feat = np.mean(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=13).T,axis=0)
    
    # Calculate the standard deviation of mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.std' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    std_mfcc_feat = np.std(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=13).T,axis=0)
    
    # Calculate the average zero crossing rate (utterance-level feature) using 'rosa.feat.zero_crossing_rate()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    zcross_feat = rosa.feature.zero_crossing_rate(sig)
    avg_zcross_feat = np.mean(rosa.feature.zero_crossing_rate(y=sig).T,axis=0)
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(avg_mfcc_feat, std_mfcc_feat, axis=0)
    
    feat1 = np.append(feat0, avg_zcross_feat, axis=0)
    
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '\\' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '\\' + file_list[i]))[0].split('_')[1]
    
    # Create a new Numpy array 'sample' to store features along with label
    sample = np.insert(feat1, obj=27, values=label)
    
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

# Transfer Numpy array data to a Pandas dataframe
df = pd.DataFrame(data=result_array)
# Label only the last (target) column
df = df.rename({27: "Emotion"}, axis='columns')
# Rename class labels
df['Emotion'].replace({1.0: "Neutral", 3.0: "Happy", 4.0: "Sad", 5.0: "Angry", 6.0: "Fearful", 7.0: "Disgust", 8.0: "Surprised"}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)
df.head(12)

df['Emotion'].value_counts()

# Extract target feature 'Emotion' in a vector y. Indexing from 0
y = df.iloc[0:2800, 27].values
# Extract features 'buying' and 'safety' in a vector X. Indexing from 0
X = df.iloc[0:2800, list(range(27))].values
print(y)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# Create an object 'sc'
sc = StandardScaler()

# Scale training data
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# Create SVM model with hyperparameters
#svm = SVC(kernel='sigmoid', C=10.0, random_state=1)

# Best settings from GridSearchCV
svm = SVC(kernel='rbf', C=10.0, gamma=0.1, random_state=1)

# This is training the model
svm.fit(X_train_std, y_train)

# Scale test data
sc.fit(X_test)
X_test_std = sc.transform(X_test)

# Test the model data
y_pred = svm.predict(X_test_std)

# Print out the performance metrics
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Training Accuracy: %.2f' % svm.score(X_train_std, y_train))
#print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Test Accuracy: %.2f' % svm.score(X_test_std, y_test))

# Print out the performance metrics in a text file.
#text_file = open("Emotion Classifier Performance Metrics.txt", "w")
#text_file.write('Misclassified samples: %d' % (y_test != y_pred).sum())
#text_file.write('Training Accuracy: %.2f' % svm.score(X_train_std, y_train))
#text_file.write('Test Accuracy: %.2f' % svm.score(X_test_std, y_test))
#text_file.close()