#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: SVM_Best_Load.py
# Date: 10/1/20
#
# Features:
# 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast, 2 poly features, and 1 RMS.
# 
#*************************************************************************************

# This code records audio for 3 seconds on a loop and predicts emotion using SVM classifier

import librosa as rosa
import numpy as np
import joblib
import sounddevice as sd


# Load the SVM model from the pickle file 
svm_pkl = joblib.load('SVM_librosa.pkl')  

fs = 16000  # Record at 16000 samples per second
seconds = 3

def change_label(argument):
    switcher = {
        1:"Neutral",
        2:"Happy",
        3:"Sad",
        4:"Angry",
        5:"Fearful",
        6:"Disgust",
        7:"Surprised",
    }
    return switcher.get(argument, "Nothing")

print('Recording...')

while True:
    # In sounddevice, frames mean samples!
    # Blocksize is the number of samples per frame!
    
    # Store recorded signal into a Numpy array
    sig = sd.rec(frames=int(fs*seconds), samplerate=fs, channels=1, blocksize=512)
    
    sd.wait() # Wait until recording is finished
    
    sig = np.reshape(sig, (48000,))
    
    # Calculate the average mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    avg_mfcc_feat = np.mean(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True).T,axis=0)
    
    # Calculate the standard deviation of mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.std' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    std_mfcc_feat = np.std(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True).T,axis=0)
    
    spec_feat = np.mean(rosa.feature.spectral_contrast(y=sig, sr=fs, n_fft=512, hop_length=256).T, axis=0)
    
    poly_feat = np.mean(rosa.feature.poly_features(y=sig, sr=fs, n_fft=512, hop_length=256).T, axis=0)
    
    rms_feat = np.mean(rosa.feature.rms(y=sig, frame_length=512, hop_length=256).T, axis=0)
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(avg_mfcc_feat, std_mfcc_feat, axis=0)
    
    feat1 = np.append(feat0, spec_feat, axis=0)
    
    feat2 = np.append(feat1, poly_feat, axis=0)
    
    feat3 = np.append(feat2, rms_feat, axis=0)
    
    feat = np.reshape(feat3, (1,-1)) 
    
    mean_vals = np.mean(feat, axis=0)
    std_val = np.std(feat)

    # Standardize the inputs
    feat_centered = (feat - mean_vals)/std_val
    
    # Make prediction using SVM model
    pred = svm_pkl.predict(feat_centered)
    
    print(change_label(pred[0]))
    
    del sig