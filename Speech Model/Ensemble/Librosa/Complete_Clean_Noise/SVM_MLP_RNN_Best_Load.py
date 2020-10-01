#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: SVM_MLP_RNN_Best_Load.py
# Date: 10/1/20
#
# Features:
# 26 MFCCs (mean) and 26 MFCCs (standard deviation), 7 spectral contrast, 2 poly features, and 1 RMS (for SVM and MLP).
# 26 MFCCs, 7 spectral contrast, 2 poly features, and 1 RMS (for RNN).
# 
#*************************************************************************************

# This code records audio for 3 seconds on a loop and predicts emotion using SVM, MLP, and RNN classifier

import librosa as rosa
import numpy as np
import tensorflow as tf
import joblib
import sounddevice as sd
import statistics


# Load the SVM model from the pickle file 
svm_pkl = joblib.load('SVM_librosa.pkl')  

# Load the MLP model from the pickle file 
mlp_h5 = tf.keras.models.load_model('MLP_Librosa.h5')

# Load the RNN model from the pickle file 
rnn_h5 = tf.keras.models.load_model('RNN_Librosa.h5') 

fs = 16000  # Record at 16000 samples per second
seconds = 3
median_num_frames = 153 # 16000*3/512

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
    
    # SVM and MLP feature extraction
    # Calculate the average mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.mean' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    avg_mfcc_feat = np.mean(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True).T,axis=0)
    
    # Calculate the standard deviation of mfcc (utterance-level features) using 'rosa.feat.mfcc()' and 'np.std' method. '.T' transposes the rows and columns. 'axis=0' indicates average is calculated column-wise
    std_mfcc_feat = np.std(rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True).T,axis=0)
    
    avg_spec_feat = np.mean(rosa.feature.spectral_contrast(y=sig, sr=fs, n_fft=512, hop_length=256).T, axis=0)
    
    avg_poly_feat = np.mean(rosa.feature.poly_features(y=sig, sr=fs, n_fft=512, hop_length=256).T, axis=0)
    
    avg_rms_feat = np.mean(rosa.feature.rms(y=sig, frame_length=512, hop_length=256).T, axis=0)
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(avg_mfcc_feat, std_mfcc_feat, axis=0)
    
    feat1 = np.append(feat0, avg_spec_feat, axis=0)
    
    feat2 = np.append(feat1, avg_poly_feat, axis=0)
    
    feat3 = np.append(feat2, avg_rms_feat, axis=0)
    
    feat = np.reshape(feat3, (1,-1)) 
    
    mean_vals = np.mean(feat, axis=0)
    std_val = np.std(feat)

    # Standardize the inputs
    feat_centered = (feat - mean_vals)/std_val
    
    # RNN feature extraction
    # 'rosa.feature.mfcc' extracts n_mfccs from signal and stores it into 'mfcc_feat'
    mfcc_feat = rosa.feature.mfcc(y=sig, sr=fs, n_mfcc=26, n_fft=512, hop_length=256, htk=True)
    
    spec_feat = rosa.feature.spectral_contrast(y=sig, sr=fs, n_fft=512, hop_length=256)
	
    poly_feat = rosa.feature.poly_features(y=sig, sr=fs, n_fft=512, hop_length=256)
	
    rms_feat = rosa.feature.rms(y=sig, frame_length=512, hop_length=256)
    
    # Append the three 1D arrays into a single 1D array called 'feat'.
    feat0 = np.append(mfcc_feat, spec_feat, axis=0)
    
    feat1 = np.append(feat0, poly_feat, axis=0)
	
    feat2 = np.append(feat1, rms_feat, axis=0)
    
    # Transpose the array to flip the rows and columns. This is done so that the features become column parameters, making each row an audio frame.
    transp_feat = feat2.T
    
    # Note: The 'cap frame number' is basically the limit we set for the number of frames for each sample, so that all samples have equal dimensions.
    if transp_feat.shape[0]<median_num_frames:

        # If number of frames is smaller than the cap frame number, we pad the array in order to reach our desired dimensions.

        # Pad the array so that it matches the cap frame number. The second value in the argument contains two tuples which indicate which way to pad how much.  
        transp_feat = np.pad(transp_feat, ((0, median_num_frames-transp_feat.shape[0]), (0,0)), 'mean')

    elif transp_feat.shape[0]>median_num_frames:

        # If number of frames is larger than the cap frame number, we delete rows (frames) which exceed the cap frame number in order to reach our desired dimensions.

        # Define a tuple which contains the range of the row indices to delete.
        row_del_index = (range(median_num_frames, transp_feat.shape[0], 1))

        transp_feat = np.delete(transp_feat, row_del_index, axis=0)

    else:
        # If number of frames match the cap frame length, perfect!
        transp_feat = transp_feat
    
    # Transpose again to flip the rows and columns. This is done so that the features become row parameters, making each column an audio frame.
    transp2_feat = transp_feat.T
    
    # Flatten the entire 2D Numpy array into 1D Numpy array. So, the first 36 values of the 1D array represent the features for first frame, the second 36 represent the features for second frame, and so on till the final (cap) frame.
    # 'C' means row-major ordered flattening.
    feat_rnn = transp2_feat.flatten('C')
    
    feat_rnn = np.reshape(feat_rnn, (1,-1)) 
    
    mean_vals = np.mean(feat_rnn, axis=0)
    std_val = np.std(feat_rnn)

    # Standardize the inputs
    feat_centered_rnn = (feat_rnn - mean_vals)/std_val
    
    # Reshaping feat_centered to 3D Numpy array for feeding into the RNN. RNNs require 3D array input.
    # 3D dimensions are (layers, rows, columns).
    feat_3D = np.reshape(feat_centered_rnn, (feat_centered_rnn.shape[0], median_num_frames, 36))
    
    # Transpose tensors so that rows=features and columns=frames.
    feat_3D_posed = tf.transpose(feat_3D, perm=[0, 2, 1])
    
    # Make prediction using SVM model
    pred_svm = svm_pkl.predict(feat_centered)
    
    # Make prediction using MLP model
    pred_mlp = mlp_h5.predict(feat_centered)
    
    # Convert One Hot label to integer label
    pred_mlp = np.argmax(pred_mlp,axis=1)
    
    # Make prediction using RNN model
    pred_rnn = rnn_h5.predict(feat_3D_posed)
    
    # Convert One Hot label to integer label
    pred_rnn = np.argmax(pred_rnn,axis=1)
    
    # Put all three predictions in a list
    preds = []
    preds.append(int(pred_svm))
    preds.append(int(pred_mlp))
    preds.append(int(pred_rnn))
    
    # Voting - final prediction is the class that was predicted the most
    pred = statistics.mode(preds)
    
    print(change_label(pred))
    
    del sig