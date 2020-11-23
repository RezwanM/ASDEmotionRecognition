# ASDSpeechEmotionRecognition

This repository contains files from a speech emotion recognition system which was created using machine learning and deep learning techniques. Ensemble learning was used, which involves joining multiple machine learning algorithms using voting to classify speech recordings in real-time. A support vector machine (SVM), a multilayer perceptron (MLP), and a recurrent neural network (RNN) model was trained on the Ryerson Audio-Visual Database of Emotional Speech and Songs (RAVDESS), the Toronto Emotional Speech Set (TESS), the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D), and a custom dataset which contains utterances from the three datasets with added background noise. Three background noise samples were added (Playground Noise, Shopping Mall Ambiance, and Streets) at three different SNR values (0 dB, 5 dB, and 10 dB). MATLAB was used for noise addition. Two separate audio feature sets were used, and their performances were compared. One of them was a custom feature set and the other one contained features from a popular speech emotion feature set. The custom feature set was extracted using the *Librosa* library, and contains 36 low-level descriptors: MFCCs (26), spectral contrast (7), polynomial coefficients (2), and RMS energy (1). The partial GeMAPS feature set was extracted using the *Parselmouth* library, and contains fifteen low-level descriptors: Pitch (1), Loudness (1), MFCCs (4), Formants (3), Formant bandwidths (3), HNR (1), Jitter (1), and Shimmer (1).

*Python Libraries:*
tensorflow 2.1.0; scikit-learn	0.23.2; numpy	1.18.5; pandas	1.1.1; matplotlib	3.1.3; praat-parselmouth 0.3.3; librosa	0.8.0; tensorflow-estimator	2.1.0; joblib	0.14.1; h5py	2.10.0; numba 0.48.0



