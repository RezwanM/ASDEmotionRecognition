#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: SVM_Librosa_RAVDESS_Clean_Noise_Conf.py
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
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import resample


# Initialize timer
t_1 = time.perf_counter()

# Set the random seeds for replicating results over multiple runs.
np.random.seed(0)

# Import dataframe/dataset into an instance/object 'df' using Pandas. Use first row as column header and first column as row header!
df = pd.read_csv(r'/home/r_m727/All_Files/Corpus/Simulated/RAVDESS/RAVDESS_Librosa_Clean_Noise.csv', header=0, index_col=0)

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
y_less = df_less.iloc[0:2304, 62].values
# Extract features of majority classes.
X_less = df_less.iloc[0:2304, list(range(62))].values
print(y_less)

# Split and stratify majority class samples for training and testing.
X_train_temp_less, X_test_less, y_train_temp_less, y_test_less = train_test_split(X_less, y_less, test_size=230, random_state=0, stratify=y_less) # training split = 90%, test split = 10%

# Further split and stratify majority class training samples for training data for training and validating.
X_train_less, X_val_less, y_train_less, y_val_less = train_test_split(X_train_temp_less, y_train_temp_less, test_size=230, random_state=0, stratify=y_train_temp_less) # training split = 80%, validation split = 10%

# Take minority data samples from dataframe to array
neutral_array = df_neutral.to_numpy()

# Shuffle the data samples of minority class
np.random.shuffle(neutral_array)

# Split minority class Neutral in 80:10:10 ratio.
train_neutral = neutral_array[0:152, :]
val_neutral = neutral_array[152:172, :]
test_neutral = neutral_array[172:192, :]

# Resample Neutral data to match majority class samples.
train_neutral_resampled = resample(train_neutral, n_samples=308, replace=True, random_state=0)
val_neutral_resampled = resample(val_neutral, n_samples=38, replace=True, random_state=0)
test_neutral_resampled = resample(test_neutral, n_samples=38, replace=True, random_state=0)

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

#class_names = df_upsampled['Emotion'].unique()
class_names = np.unique(y_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_test, labels=class_names)
#confmat.shape
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
plot_confusion_matrix(confmat, class_names, title='SVM_Custom_RAVDESS_Clean_Noise')
#plt.show()

# Save plot as PNG file
fig.savefig('Confusion_Matrix_SVM_Librosa_RAVDESS_Clean_Noise.png')

t_2 = time.perf_counter()

print('Time taken to execute code: % seconds' % (t_2-t_1))

