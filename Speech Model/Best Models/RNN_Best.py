#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Best_RNN.py
# Date: 10/1/20
#
# Objective:
# 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast, 2 poly features, and 1 RMS.
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
from sklearn.utils import resample
from tensorflow.keras.callbacks import LearningRateScheduler

# Initialize timer
t_1 = time.perf_counter()

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
y = df_upsampled.iloc[0:26082, 36*median_num_frames].values
# Extract features 'buying' and 'safety' in a vector X. Indexing from 0
X = df_upsampled.iloc[0:26082, list(range(36*median_num_frames))].values
print(y)

# Split data for training and testing. Stratifying y means that within each split, every class will have same number of samples. random_state=int means data is not shuffled randomly (i.e. same samples in each split every time!).
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=2608, random_state=0, stratify=y) # training split = 90%, test split = 10%

# Further split training data for training and validating. Randomly shuffle training and validation data each time (i.e. no random seed!)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=2608, random_state=0, stratify=y_train_temp) # training split = 80%, validation split = 10%

mean_vals = np.mean(X_train_temp, axis=0)
std_val = np.std(X_train_temp)

# Standardize the inputs
X_train_centered = (X_train - mean_vals)/std_val
X_val_centered = (X_val - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_val, X_test, X_train_temp, y_train_temp

print(X_train_centered.shape, y_train.shape)
print(X_val_centered.shape, y_val.shape)
print(X_test_centered.shape, y_test.shape)

np.random.seed(0)

tf.random.set_seed(0)

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

# Create an object/instance 'model' for the 'Sequential()' class.
model = keras.models.Sequential()
	
model.add(
	keras.layers.GRU( units=36,
				input_shape=(36, median_num_frames),
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='relu',
				dropout=0.25,
				return_sequences=True))

model.add(
	keras.layers.GRU( units=36,
				input_shape=(36, median_num_frames),
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='relu'))

model.add(
	keras.layers.Dense( units=y_train_onehot.shape[1],
				input_dim=36,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='softmax'))

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)

adam_optimizer = keras.optimizers.Adam(
					learning_rate=lrate) #1e-06 gave better result than default value 1e-07

model.compile(optimizer=adam_optimizer,
					loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])

# Train the RNN
history = model.fit(X_train_3D_posed, y_train_onehot, batch_size=16, epochs=50, verbose=2, validation_data=(X_val_3D_posed, y_val_onehot)) # 80% training / 10% validation

print(history.history)

# Evaluate the model on the test data using `evaluate`
results = model.evaluate(X_test_3D_posed, y_test_onehot, batch_size=16)
print("test loss, test acc:", results)


# Plot the training and validation accuracies vs. epochs for the latest loop iteration
fig = plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.grid()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Save plot as PNG file
fig.savefig('Accuracies vs. Epochs_GRU_GRU_36.png')

# Plot the training and validation losses vs. epochs for the latest loop iteration
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Save plot as PNG file
fig.savefig('Losses vs. Epochs_GRU_GRU_36.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))
