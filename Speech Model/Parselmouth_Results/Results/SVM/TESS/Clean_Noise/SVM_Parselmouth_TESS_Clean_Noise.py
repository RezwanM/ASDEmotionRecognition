#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: SVM_Parselmouth_TESS_Clean_Noise.py
# Date: 10/24/20
#
# Features:
# 36 features - Functionals on Pitch (f0), Loudness(Intensity), MFCC, formants, formants BW, HNR, Jitter, and Shimmer.
#
#*************************************************************************************

import pandas as pd
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa as rosa
import os
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.utils import resample


# Initialize timer
t_1 = time.perf_counter()

# Set the random seeds for replicating results over multiple runs.
np.random.seed(0)

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/TESS/TESS_Parselmouth_Clean_Noise.csv', header=0, index_col=0)

# Replace NaN values with zeroes
df = df.fillna(0)

# Rename target labels.
#df['Emotion'].replace({"Neutral" : 1.0, "Happy" : 2.0, "Sad" : 3.0, "Angry" : 4.0, "Fearful" : 5.0, "Disgust" : 6.0, "Surprised" : 7.0}, inplace=True)
df['Emotion'].replace({1.0 : 1.0, 3.0 : 2.0, 4.0 : 3.0, 5.0 : 4.0, 6.0 : 5.0, 7.0 : 6.0, 8.0 : 7.0}, inplace=True)

# Display class counts
print(df['Emotion'].value_counts())

# Extract target feature 'Emotion' in a vector y. Indexing from 0
y = df.iloc[0:5600, 36].values
# Extract features in a matrix X. Indexing from 0
X = df.iloc[0:5600, list(range(36))].values
print(y)

# Split data for training and testing. Stratifying y means that within each split, every class will have same number of samples. random_state=int means data is not shuffled randomly (i.e. same samples in each split every time!).
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=560, random_state=0, stratify=y) # training split = 90%, test split = 10%

# Further split training data for training and validating. Randomly shuffle training and validation data each time (i.e. no random seed!)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=560, random_state=0, stratify=y_train_temp) # training split = 80%, validation split = 10%


mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train, axis=0)

# Standardize the inputs
X_train_temp_centered = (X_train_temp - mean_vals)/std_val
X_train_centered = (X_train - mean_vals)/std_val
X_val_centered = (X_val - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_val, X_test

print(X_train_temp_centered.shape, y_train_temp.shape)
print(X_train_centered.shape, y_train.shape)
print(X_val_centered.shape, y_val.shape)
print(X_test_centered.shape, y_test.shape)



# Best settings from tuning
svm = SVC(kernel='rbf', C=10, gamma=0.01, random_state=0)

# This is training the model
svm.fit(X_train_centered, y_train)

# Test the model data using validation data
y_pred_val = svm.predict(X_val_centered)

# Test the model data using test data
y_pred_test = svm.predict(X_test_centered)

# Print out the performance metrics
print('Misclassified validation samples: %d' % (y_val != y_pred_val).sum())
print('Misclassified test samples: %d' % (y_test != y_pred_test).sum())
print('Training Accuracy: %.2f' % svm.score(X_train_centered, y_train))
print('Validation Accuracy: %.2f' % svm.score(X_val_centered, y_val))
print('Test Accuracy: %.2f' % svm.score(X_test_centered, y_test))

# Print out more performance metrics (Precision and Recall)
more_scores = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')

print('Precision: ', more_scores[0])
print('Recall: ', more_scores[1])

# Define a 10 fold CV with 11 % data of training set (train_temp) for validation
# 11 %, not 10 %,  because the validation split is being used instead of the test split.
cv = ShuffleSplit(n_splits=10, test_size=0.11, random_state=0)

# Plot learning curves with 10-fold CV
train_sizes, train_scores, test_scores = learning_curve(estimator=svm, X=X_train_temp_centered, y=y_train_temp, train_sizes=np.linspace(0.1, 1.0, 10), cv=cv, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig = plt.figure()
plt.plot(train_sizes, train_mean, color='tab:blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='tab:blue')
plt.plot(train_sizes, test_mean, color='tab:orange', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='tab:orange')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('SVM_P.GeMAPS_TESS_Clean_Noise')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
#plt.show()

# Save plot as PNG file
fig.savefig('Learning_Curves_SVM_Parselmouth_TESS_Clean_Noise.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))

