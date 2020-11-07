#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: RNN_Librosa_TESS_Clean_Noise.py
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
import itertools
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import pipeline
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import statistics
from sklearn.utils import resample
from tensorflow.keras.callbacks import LearningRateScheduler


# Initialize timer
t_1 = time.perf_counter()

# Set the random seeds for replicating results over multiple runs.
np.random.seed(0)
tf.random.set_seed(0)

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/TESS/TESS_Librosa_Clean_Noise_RNN.csv', header=0, index_col=0)

median_num_frames = (df.shape[1]-1)//36

df['Emotion'].replace({"Neutral" : 1.0, "Happy" : 2.0, "Sad" : 3.0, "Angry" : 4.0, "Fearful" : 5.0, "Disgust" : 6.0, "Surprised" : 7.0}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)

# Display class counts
print(df['Emotion'].value_counts())

# Extract target feature 'Emotion' in a vector y. Indexing from 0
y = df.iloc[0:5600, 36*median_num_frames].values
# Extract features in a matrix X. Indexing from 0
X = df.iloc[0:5600, list(range(36*median_num_frames))].values
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

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)


adam_optimizer = keras.optimizers.Adam(
					learning_rate=lrate) #1e-06 gave better result than default value 1e-07
# Define the MLP model
def model():
    model = keras.models.Sequential([
        keras.layers.LSTM(units=36, input_shape=(36, median_num_frames), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh', recurrent_activation='sigmoid', dropout=0.30, recurrent_dropout=0.30, return_sequences=True),
        keras.layers.LSTM(units=12, input_shape=(36, median_num_frames), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh', recurrent_activation='sigmoid', dropout=0.30),
        keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=12, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax')
    ])
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    return model

# Make the pipeline
pipe_rnn = pipeline.Pipeline([
    ('nn', KerasClassifier(build_fn=model, nb_epoch=50, batch_size=16, verbose=2, validation_data=(X_val_3D_posed, y_val_onehot)))
])
             
pipe_rnn.fit(X_train_3D_posed, y_train_onehot)

y_pred = pipe_rnn.predict(X_test_3D_posed)
#class_names = df_upsampled['Emotion'].unique()

# Convert One Hot labels into integers
y_test_int = np.argmax(y_test_onehot, axis=1)

class_names = np.unique(y_test_int)

confmat = confusion_matrix(y_true=y_test_int, y_pred=y_pred, labels=class_names)

pd.DataFrame(confmat, index=class_names, columns=class_names)

# Convert label array to list
class_names = list(class_names)

# Rename classes for Confusion Matrix
class_names[0] = "Neutral"
class_names[1] = "Happy"
class_names[2] = "Sad"
class_names[3] = "Anger"
class_names[4] = "Fear"
class_names[5] = "Disgust"
class_names[6] = "Surprise"


# Define function for plotting the Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

fig = plt.figure()
plot_confusion_matrix(confmat, class_names, title='RNN_Custom_TESS_Clean_Noise')
#plt.show()

# Save plot as PNG file
fig.savefig('Confusion_Matrix_RNN_Librosa_TESS_Clean_Noise.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))

