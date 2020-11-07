#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: MLP_Librosa_TESS_Clean_Noise.py
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
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.utils import resample
from tensorflow.keras.callbacks import LearningRateScheduler


# Initialize timer
t_1 = time.perf_counter()

# Set the random seeds for replicating results over multiple runs.
np.random.seed(0)
tf.random.set_seed(0)

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/TESS/TESS_Librosa_Clean_Noise.csv', header=0, index_col=0)

df['Emotion'].replace({"Neutral" : 1.0, "Happy" : 2.0, "Sad" : 3.0, "Angry" : 4.0, "Fearful" : 5.0, "Disgust" : 6.0, "Surprised" : 7.0}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)

# Display class counts
print(df['Emotion'].value_counts())

# Extract target feature 'Emotion' in a vector y. Indexing from 0
y = df.iloc[0:5600, 62].values
# Extract features in a matrix X. Indexing from 0
X = df.iloc[0:5600, list(range(62))].values
print(y)

# Split data for training and testing. Stratifying y means that within each split, every class will have same number of samples. random_state=int means data is not shuffled randomly (i.e. same samples in each split every time!).
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=560, random_state=0, stratify=y) # training split = 90%, test split = 10%

# Further split training data for training and validating. Randomly shuffle training and validation data each time (i.e. no random seed!)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=560, random_state=0, stratify=y_train_temp) # training split = 80%, validation split = 10%


mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train, axis=0)

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

# Create an object/instance 'model' for the 'Sequential()' class.
model = keras.models.Sequential()
model.add(
	keras.layers.Dense( units=105,
				input_dim=X_train_centered.shape[1],
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros', 
				activation='relu'))

model.add(
    keras.layers.Dropout(0.30))

model.add(
	keras.layers.Dense( units=62,
				input_dim=105,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros', 
				activation='relu'))

model.add(
   keras.layers.Dropout(0.10))

model.add(
	keras.layers.Dense( units=y_train_onehot.shape[1],
				input_dim=62,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				activation='softmax'))

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)

adam_optimizer = keras.optimizers.Adam(
					learning_rate=lrate) #1e-06 gave better result than default value 1e-07

model.compile(optimizer=adam_optimizer,
					loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])
						  
							# cross-entropy: fancy name for logistic regression                        

# Train the MLP
history = model.fit(X_train_centered, y_train_onehot, batch_size=16, epochs=50, verbose=2, validation_data=(X_val_centered, y_val_onehot))

print(history.history)

# Evaluate the model on the test data using `evaluate`
results = model.evaluate(X_test_centered, y_test_onehot, batch_size=16)
print("test loss, test acc:", results)


# Plot the training and validation accuracies vs. epochs for the latest loop iteration
fig = plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('MLP_Custom_TESS_Clean_Noise')
plt.grid()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Save plot as PNG file
fig.savefig('Accuracy_Curves_MLP_Librosa_TESS_Clean_Noise.png')

# Plot the training and validation losses vs. epochs for the latest loop iteration
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MLP_Custom_TESS_Clean_Noise')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# Save plot as PNG file
fig.savefig('Loss_Curves_MLP_Librosa_TESS_Clean_Noise.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))
