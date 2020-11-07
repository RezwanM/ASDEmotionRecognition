#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: MLP_Librosa_CREMA-D_Clean_Conf.py
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
from sklearn import pipeline
#from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import resample
from tensorflow.keras.callbacks import LearningRateScheduler


# Initialize timer
t_1 = time.perf_counter()

# Set the random seeds for replicating results over multiple runs.
np.random.seed(0)
tf.random.set_seed(0)

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/CREMA-D/CREMA-D_Librosa_Clean.csv', header=0, index_col=0)

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
y_less = df_less.iloc[0:6355, 62].values
# Extract features of majority classes.
X_less = df_less.iloc[0:6355, list(range(62))].values
print(y_less)

# Split and stratify majority class samples for training and testing.
X_train_temp_less, X_test_less, y_train_temp_less, y_test_less = train_test_split(X_less, y_less, test_size=635, random_state=0, stratify=y_less) # training split = 90%, test split = 10%

# Further split and stratify majority class training samples for training data for training and validating.
X_train_less, X_val_less, y_train_less, y_val_less = train_test_split(X_train_temp_less, y_train_temp_less, test_size=635, random_state=0, stratify=y_train_temp_less) # training split = 80%, validation split = 10%

# Take minority data samples from dataframe to array
neutral_array = df_neutral.to_numpy()

# Shuffle the data samples of minority class
np.random.shuffle(neutral_array)

# Split minority class Neutral in 80:10:10 ratio.
train_neutral = neutral_array[0:869, :]
val_neutral = neutral_array[869:978, :]
test_neutral = neutral_array[978:1087, :]

# Resample Neutral data to match majority class samples.
train_neutral_resampled = resample(train_neutral, n_samples=1017, replace=True, random_state=0)
val_neutral_resampled = resample(val_neutral, n_samples=127, replace=True, random_state=0)
test_neutral_resampled = resample(test_neutral, n_samples=127, replace=True, random_state=0)

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

# Define the learning rate schedule. This can then be passed as the learning rate for the optimizer.
lrate = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.8)

adam_optimizer = keras.optimizers.Adam(
					learning_rate=lrate) #1e-06 gave better result than default value 1e-07
# Define the MLP model
def model():
    model = keras.models.Sequential([
        keras.layers.Dense(units=105, input_dim=X_train_centered.shape[1], kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(units=62, input_dim=105, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'),
        keras.layers.Dropout(0.10),
        keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=62, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax')
    ])
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    return model

# Make the pipeline
pipe_mlp = pipeline.Pipeline([
    ('nn', KerasClassifier(build_fn=model, nb_epoch=50, batch_size=16, verbose=2, validation_data=(X_val_centered, y_val_onehot)))
])
             
pipe_mlp.fit(X_train_centered, y_train_onehot)

y_pred = pipe_mlp.predict(X_test_centered)
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
plot_confusion_matrix(confmat, class_names, title='MLP_Custom_CREMA-D_Clean')
#plt.show()

# Save plot as PNG file
fig.savefig('Confusion_Matrix_MLP_Librosa_CREMA-D_Clean.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))

