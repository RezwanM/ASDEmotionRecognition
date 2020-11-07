#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: All_Librosa_Comple_Clean_Noise_Load.py
# Date: 10/24/20
#
# Features:
# 62 features for SVM and MLP - 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast (mean), 2 poly features (mean), and 1 RMS (mean).
# 36 features for RNN - 26 MFCCs, 7 spectral contrast, 2 poly features, and 1 RMS.
#
#*************************************************************************************


import librosa as rosa
import numpy as np
import tensorflow as tf
import joblib
import statistics
from statistics import StatisticsError
import pandas as pd
import math
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.utils import resample

np.random.seed(0)

tf.random.set_seed(0)

# Load the SVM model from the pickle file 
svm_pkl = joblib.load(r'/home/r_m727/All_Files/Dep_Metrics/SVM_Librosa_Complete_Clean_Noise.pkl')  

# Load the MLP model from the pickle file 
mlp_h5 = tf.keras.models.load_model(r'/home/r_m727/All_Files/Dep_Metrics/MLP_Librosa_Complete_Clean_Noise.h5')

# Load the RNN model from the pickle file 
rnn_h5 = tf.keras.models.load_model(r'/home/r_m727/All_Files/Dep_Metrics/RNN_Librosa_Complete_Clean_Noise.h5') 


#SVM and MLP
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

mean_vals = np.mean(X_train_temp, axis=0)
std_val = np.std(X_train_temp, axis=0)

# Standardize the inputs
X_train_centered_mlp = (X_train - mean_vals)/std_val
X_val_centered_mlp = (X_val - mean_vals)/std_val
X_test_centered_mlp = (X_test - mean_vals)/std_val

del X_train, X_val, X_test, X_train_temp, y_train_temp

print(X_train_centered_mlp.shape, y_train.shape)
print(X_val_centered_mlp.shape, y_val.shape)
print(X_test_centered_mlp.shape, y_test.shape)

# One-Hot Encode the classes
y_train_onehot = keras.utils.to_categorical(y_train)
y_val_onehot = keras.utils.to_categorical(y_val)
y_test_onehot = keras.utils.to_categorical(y_test)

#RNN
# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS+TESS+CREMA-D/RAVDESS+TESS+CREMA-D_Librosa_Clean_Noise_RNN.csv', header=0, index_col=0)

median_num_frames = (df.shape[1]-1)//36

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
y = df.iloc[0:26082, 36*median_num_frames].values
# Extract features 'buying' and 'safety' in a vector X. Indexing from 0
X = df.iloc[0:26082, list(range(36*median_num_frames))].values
print(y)

# Split data for training and testing. Stratifying y means that within each split, every class will have same number of samples. random_state=int means data is not shuffled randomly (i.e. same samples in each split every time!).
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=2608, random_state=0, stratify=y) # training split = 90%, test split = 10%

# Further split training data for training and validating. Randomly shuffle training and validation data each time (i.e. no random seed!)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=2608, random_state=0, stratify=y_train_temp) # training split = 80%, validation split = 10%

mean_vals = np.mean(X_train_temp, axis=0)
std_val = np.std(X_train_temp, axis=0)

# Standardize the inputs
X_train_centered = (X_train - mean_vals)/std_val
X_val_centered = (X_val - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_val, X_test, X_train_temp, y_train_temp

print(X_train_centered.shape, y_train.shape)
print(X_val_centered.shape, y_val.shape)
print(X_test_centered.shape, y_test.shape)

# One-Hot Encode the classes
y_train_onehot = keras.utils.to_categorical(y_train)
y_val_onehot = keras.utils.to_categorical(y_val)
y_test_onehot = keras.utils.to_categorical(y_test)

# Reshaping X_train and X_test to 3D Numpy arrays for feeding into the RNN. RNNs require 3D array input.
# 3D dimensions are (layers, rows, columns).
X_train_3D = np.reshape(X_train_centered, (X_train_centered.shape[0], median_num_frames, 36))
X_val_3D = np.reshape(X_val_centered, (X_val_centered.shape[0], median_num_frames, 36))
X_test_3D = np.reshape(X_test_centered, (X_test_centered.shape[0], median_num_frames, 36))

print(X_train_3D.shape, y_train.shape)
print(X_val_3D.shape, y_val.shape)
print(X_test_3D.shape, y_test.shape)

# Transpose tensors so that rows=features and columns=frames.
X_train_3D_posed = tf.transpose(X_train_3D, perm=[0, 2, 1])
X_val_3D_posed = tf.transpose(X_val_3D, perm=[0, 2, 1])
X_test_3D_posed = tf.transpose(X_test_3D, perm=[0, 2, 1])

print(X_train_3D_posed.shape, y_train.shape)
print(X_val_3D_posed.shape, y_val.shape)
print(X_test_3D_posed.shape, y_test.shape)


pred_list=[]


for i in range(0, 2608):
	feat = np.reshape(X_test_centered_mlp[i], (1,-1)) 
	feat_rnn = np.reshape(X_test_3D_posed[i], (1, 36, median_num_frames))
	
	# Make prediction using SVM model
	pred_svm = svm_pkl.predict(feat)

	# Make prediction using MLP model
	pred_mlp = mlp_h5.predict(feat)

	# Convert One Hot label to integer label
	pred_mlp = np.argmax(pred_mlp,axis=1)

	# Make prediction using RNN model
	pred_rnn = rnn_h5.predict(feat_rnn)

	# Convert One Hot label to integer label
	pred_rnn = np.argmax(pred_rnn,axis=1)

	# Put all three predictions in a list
	preds = []
	preds.append(int(pred_svm))
	preds.append(int(pred_mlp))
	preds.append(int(pred_rnn))

	# Voting - final prediction is the class that was predicted the most
	try:
		pred = statistics.mode(preds) # calculate mode of predicitions
	except StatisticsError: 
		pred = int(pred_rnn) # if all unique predictions (no mode), select MLP's prediction
	
	pred_list.append(pred)

correct_preds = np.sum(y_test == pred_list, axis=0)
test_acc = correct_preds / y_test.shape[0]

print('Test accuracy: %.2f%%' % (test_acc * 100))

