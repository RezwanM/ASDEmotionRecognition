#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: SVM_Librosa_CREMA-D_Clean_Noise.py
# Date: 10/24/20
#
# Features:
# 62 features - 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast (mean), 2 poly features (mean), and 1 RMS (mean).
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
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/CREMA-D/CREMA-D_Librosa_Clean_Noise.csv', header=0, index_col=0)

# Rename target labels.
df['Emotion'].replace({"Neutral" : 1.0, "Happy" : 2.0, "Sad" : 3.0, "Angry" : 4.0, "Fearful" : 5.0, "Disgust" : 6.0}, inplace=True)

# Take data samples of each class from dataframe into separate dataframes.
df_happy = df.loc[df.Emotion==2.0]
df_sad = df[df.Emotion==3.0]
df_angry = df[df.Emotion==4.0]
df_fearful = df[df.Emotion==5.0]
df_disgust = df[df.Emotion==6.0]
df_neutral = df[df.Emotion==1.0]

# Join only the majority classes, leaving out Neutral and Surprised.
df_less = pd.concat([df_happy, df_sad, df_angry, df_fearful, df_disgust])

# Extract labels of majority classes.
y_less = df_less.iloc[0:12710, 62].values
# Extract features of majority classes.
X_less = df_less.iloc[0:12710, list(range(62))].values
print(y_less)

# Split and stratify majority class samples for training and testing.
X_train_temp_less, X_test_less, y_train_temp_less, y_test_less = train_test_split(X_less, y_less, test_size=1271, random_state=0, stratify=y_less) # training split = 90%, test split = 10%

# Further split and stratify majority class training samples for training data for training and validating.
X_train_less, X_val_less, y_train_less, y_val_less = train_test_split(X_train_temp_less, y_train_temp_less, test_size=1271, random_state=0, stratify=y_train_temp_less) # training split = 80%, validation split = 10%

# Take minority data samples from dataframe to array
neutral_array = df_neutral.to_numpy()

# Shuffle the data samples of minority class
np.random.shuffle(neutral_array)

# Split minority class Neutral in 80:10:10 ratio.
train_neutral = neutral_array[0:1738, :]
val_neutral = neutral_array[1738:1956, :]
test_neutral = neutral_array[1956:2174, :]

# Resample Neutral data to match majority class samples.
train_neutral_resampled = resample(train_neutral, n_samples=2034, replace=True, random_state=0)
val_neutral_resampled = resample(val_neutral, n_samples=254, replace=True, random_state=0)
test_neutral_resampled = resample(test_neutral, n_samples=254, replace=True, random_state=0)

# Separate features and target labels for Neutral data.
X_train_neutral = train_neutral_resampled[:, 0:62]
X_val_neutral = val_neutral_resampled[:, 0:62]
X_test_neutral = test_neutral_resampled[:, 0:62]
y_train_neutral = train_neutral_resampled[:, 62]
y_val_neutral = val_neutral_resampled[:, 62]
y_test_neutral = test_neutral_resampled[:, 62]

# Join upsampled minority data samples with majority data samples.
X_train = np.concatenate((X_train_less, X_train_neutral), axis=0)
X_val = np.concatenate((X_val_less, X_val_neutral), axis=0)
X_test = np.concatenate((X_test_less, X_test_neutral), axis=0)
y_train = np.concatenate((y_train_less, y_train_neutral), axis=0)
y_val = np.concatenate((y_val_less, y_val_neutral), axis=0)
y_test = np.concatenate((y_test_less, y_test_neutral), axis=0)

# Join Training and Validation split to create CV results in Learning Curves below.
X_train_temp = np.concatenate((X_train, X_val), axis=0)
y_train_temp = np.concatenate((y_train, y_val), axis=0)

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train, axis=0)

# Standardize the inputs
X_train_temp_centered = (X_train_temp - mean_vals)/std_val
X_train_centered = (X_train - mean_vals)/std_val
X_val_centered = (X_val - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_val, X_test, X_train_temp_less, y_train_temp_less

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
plt.title('SVM_Custom_CREMA-D_Clean_Noise')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
#plt.show()

# Save plot as PNG file
fig.savefig('Learning_Curves_SVM_Librosa_CREMA-D_Clean_Noise.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))

