#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: RNN_Librosa_RAVDESS_Clean.py
# Date: 10/24/20
#
# Features:
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
from sklearn.utils import resample
from tensorflow.keras.callbacks import LearningRateScheduler


# Initialize timer
t_1 = time.perf_counter()

# Set the random seeds for replicating results over multiple runs.
np.random.seed(0)
tf.random.set_seed(0)

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS/RAVDESS_Librosa_Clean_RNN.csv', header=0, index_col=0)

# column of dataframe represents the features (36*median_num_frames) and -1 to avoid considering label column.
median_num_frames = (df.shape[1]-1)//36

# Rename target labels.
df['Emotion'].replace({"Neutral" : 1.0, "Happy" : 2.0, "Sad" : 3.0, "Angry" : 4.0, "Fearful" : 5.0, "Disgust" : 6.0, "Surprised" : 7.0}, inplace=True)

# Take data samples of each class from dataframe into separate dataframes.
df_happy = df.loc[df.Emotion==2.0]
df_sad = df[df.Emotion==3.0]
df_angry = df[df.Emotion==4.0]
df_fearful = df[df.Emotion==5.0]
df_disgust = df[df.Emotion==6.0]
df_neutral = df[df.Emotion==1.0]
df_surprised = df[df.Emotion==7.0]

# Join only the majority classes, leaving out Neutral.
df_less = pd.concat([df_happy, df_sad, df_angry, df_fearful, df_disgust, df_surprised])

# Extract labels of majority classes.
y_less = df_less.iloc[0:1152, 36*median_num_frames].values
# Extract features of majority classes.
X_less = df_less.iloc[0:1152, list(range(36*median_num_frames))].values
print(y_less)

# Split and stratify majority class samples for training and testing.
X_train_temp_less, X_test_less, y_train_temp_less, y_test_less = train_test_split(X_less, y_less, test_size=115, random_state=0, stratify=y_less) # training split = 90%, test split = 10%

# Further split and stratify majority class training samples for training data for training and validating.
X_train_less, X_val_less, y_train_less, y_val_less = train_test_split(X_train_temp_less, y_train_temp_less, test_size=115, random_state=0, stratify=y_train_temp_less) # training split = 80%, validation split = 10%

# Take minority data samples from dataframe to array
neutral_array = df_neutral.to_numpy()

# Shuffle the data samples of minority class
np.random.shuffle(neutral_array)

# Split minority class Neutral in 80:10:10 ratio.
train_neutral = neutral_array[0:76, :]
val_neutral = neutral_array[76:86, :]
test_neutral = neutral_array[86:96, :]

# Resample Neutral data to match majority class samples.
train_neutral_resampled = resample(train_neutral, n_samples=154, replace=True, random_state=0)
val_neutral_resampled = resample(val_neutral, n_samples=19, replace=True, random_state=0)
test_neutral_resampled = resample(test_neutral, n_samples=19, replace=True, random_state=0)

# Separate features and target labels for Neutral data.
X_train_neutral = train_neutral_resampled[:, 0:36*median_num_frames]
X_val_neutral = val_neutral_resampled[:, 0:36*median_num_frames]
X_test_neutral = test_neutral_resampled[:, 0:36*median_num_frames]
y_train_neutral = train_neutral_resampled[:, 36*median_num_frames]
y_val_neutral = val_neutral_resampled[:, 36*median_num_frames]
y_test_neutral = test_neutral_resampled[:, 36*median_num_frames]

# Join upsampled minority data samples with majority data samples.
X_train = np.concatenate((X_train_less, X_train_neutral), axis=0)
X_val = np.concatenate((X_val_less, X_val_neutral), axis=0)
X_test = np.concatenate((X_test_less, X_test_neutral), axis=0)
y_train = np.concatenate((y_train_less, y_train_neutral), axis=0)
y_val = np.concatenate((y_val_less, y_val_neutral), axis=0)
y_test = np.concatenate((y_test_less, y_test_neutral), axis=0)


mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train, axis=0)

# Standardize the inputs
X_train_centered = (X_train - mean_vals)/std_val
X_val_centered = (X_val - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_val, X_test, X_train_temp_less, y_train_temp_less

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

# Create an object/instance 'model' for the 'Sequential()' class.
model = keras.models.Sequential()
	
model.add(
	keras.layers.LSTM( units=36,
				input_shape=(36, median_num_frames),
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='tanh',
				recurrent_activation='sigmoid',
				dropout=0.30,
				recurrent_dropout=0.30,
				return_sequences=True))

model.add(
	keras.layers.LSTM( units=12,
				input_shape=(36, median_num_frames),
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='tanh',
				recurrent_activation='sigmoid',
				dropout=0.30))

model.add(
	keras.layers.Dense( units=y_train_onehot.shape[1],
				input_dim=12,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='softmax'))

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)

adam_optimizer = keras.optimizers.Adam(
					learning_rate=lrate) #1e-06 gave better result than default value 1e-07

model.compile(optimizer=adam_optimizer,
					loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])

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
plt.title('RNN_Custom_RAVDESS_Clean')
plt.grid()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Save plot as PNG file
fig.savefig('Accuracy_Curves_RNN_Librosa_RAVDESS_Clean.png')

# Plot the training and validation losses vs. epochs for the latest loop iteration
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN_Custom_RAVDESS_Clean')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Save plot as PNG file
fig.savefig('Loss_Curves_RNN_Librosa_RAVDESS_Clean.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))

