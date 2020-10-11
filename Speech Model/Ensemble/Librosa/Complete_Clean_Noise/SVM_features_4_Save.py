#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: SVM_features_4_Save.py
# Date: 10/10/20
#
# Features:
# 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast, 2 poly features, and 1 RMS.
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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
import joblib

#n_features = 62, num_classes = 7, samples_per_class = 200

# Initialize timer
t_1 = time.perf_counter()

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS+TESS+CREMA-D/RAVDESS+TESS+CREMA-D_Librosa_Clean_Noise.csv', header=0, index_col=0)

# Balance the dataset for equal number of samples for each class.
# Separate majority and minority classes
df_minority_1 = df[df.Emotion=="Neutral"]
df_majority3 = df[df.Emotion=="Happy"]
df_majority4 = df[df.Emotion=="Sad"]
df_majority5 = df[df.Emotion=="Angry"]
df_majority6 = df[df.Emotion=="Fearful"]
df_majority7 = df[df.Emotion=="Disgust"]
df_majority8 = df[df.Emotion=="Surprised"]
	 
# Upsample minority class
df_minority_upsampled = resample(df_minority_1, 
								 replace=True,     # sample with replacement
								 n_samples=3726,    # to match majority class
								 random_state=0) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled_1 = pd.concat([df_minority_upsampled, df_majority3, df_majority4, df_majority5, df_majority6, df_majority7, df_majority8])
 
# Display new class counts
df_upsampled_1.Emotion.value_counts()

# Reset row (sample) indexing
df_upsampled_1 = df_upsampled_1.reset_index(drop=True)

df_upsampled_1['Emotion'].value_counts()

df_minority_2 = df[df.Emotion=="Surprised"]

# Upsample minority class
df_minority_upsampled_2 = resample(df_minority_2, 
								 replace=True,     # sample with replacement
								 n_samples=2542,    # to match majority class
								 random_state=0) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled_2, df_upsampled_1])
 
# Display new class counts
df_upsampled.Emotion.value_counts()

# Reset row (sample) indexing
df_upsampled = df_upsampled.reset_index(drop=True)

df_upsampled['Emotion'].value_counts()

df_upsampled['Emotion'].replace({"Neutral" : 1.0, "Happy" : 2.0, "Sad" : 3.0, "Angry" : 4.0, "Fearful" : 5.0, "Disgust" : 6.0, "Surprised" : 7.0}, inplace=True)
# Reset row (sample) indexing
df = df_upsampled.reset_index(drop=True)

# Display new class counts
print(df['Emotion'].value_counts())

# Extract target feature 'Emotion' in a vector y. Indexing from 0
y = df.iloc[0:26082, 62].values
# Extract features 'buying' and 'safety' in a vector X. Indexing from 0
X = df.iloc[0:26082, list(range(62))].values
print(y)

# Split data for training and testing. Stratifying y means that within each split, every class will have same number of samples. random_state=int means data is not shuffled randomly (i.e. same samples in each split every time!).
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=2608, random_state=0, stratify=y) # training split = 90%, test split = 10%

# Further split training data for training and validating. Randomly shuffle training and validation data each time (i.e. no random seed!)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=2608, random_state=0, stratify=y_train_temp) # training split = 80%, validation split = 10%

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

np.random.seed(0)

# Make a pipeline with StandardScaler() included and best settings from GridSearchCV
pipe_svc = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma=0.01, random_state=0))

# This is training the model
pipe_svc.fit(X_train, y_train)

# Test the model data using validation data
y_pred_val = pipe_svc.predict(X_val)

# Test the model data using test data
y_pred_test = pipe_svc.predict(X_test)

# Print out the performance metrics
print('Misclassified validation samples: %d' % (y_val != y_pred_val).sum())
print('Misclassified test samples: %d' % (y_test != y_pred_test).sum())
print('Training Accuracy: %.2f' % pipe_svc.score(X_train, y_train))
print('Validation Accuracy: %.2f' % pipe_svc.score(X_val, y_val))
print('Test Accuracy: %.2f' % pipe_svc.score(X_test, y_test))

# Print out more performance metrics (Precision and Recall)
more_scores = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')

print('Precision: ', more_scores[0])
print('Recall: ', more_scores[1])


# Define a 10 fold CV with 11 % data of training set (train_temp) for validation
cv = ShuffleSplit(n_splits=10, test_size=0.11, random_state=0)

# Plot learning curves with 10-fold CV
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_svc, X=X_train_temp, y=y_train_temp, train_sizes=np.linspace(0.1, 1.0, 10), cv=cv, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig = plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='orange', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='orange')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Accuracies vs. Samples (SVM_Librosa_RAVDESS+TESS+CREMA-D_Clean_Noise))')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
#plt.show()

# Save plot as PNG file
fig.savefig('Accuracies vs. Samples (SVM_Librosa_RAVDESS+TESS+CREMA-D_Clean_Noise).png')

# Save the model as a pickle in a file 
joblib.dump(pipe_svc, 'SVM_Librosa.pkl')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))


