# Description

"10x_CV_Results" directory contains the results of 10-fold cross-validation. More specifically, it contains files of the experiments involving both feature sets (Custom and Partial GeMAPS) and all three classifiers (SVM, MLP, and RNN).

"Best Models" directory contains the files of the best models. More specifically, it contains files of the best models created using both feature sets (Custom and Partial GeMAPS) and all three classifiers (SVM, MLP, and RNN).

"Ensemble_Performance_Metrics" directory contains the performance measures of the Ensemble model (SVM+MLP+RNN) using the Custom feature set. The directory includes individual saved model files and other files of the experiment.

"Save_Features_CSVs" directory contains files used to extract features from the datasets and save them as CSV files for later use.

"Librosa_Results" directory contains all experiment files for the Custom feature set and all three classifiers (SVM, MLP, and RNN). This directory has two sub-directories. The "Confusion_Matrices" sub-directory contains experiment files for plotting the confusion matrices, and the "Results" sub-directory contains experiment files for printing the performance metrics.

"Parselmouth_Results" directory contains all experiment files for the Partial GeMAPS feature set and all three classifiers (SVM, MLP, and RNN). This directory has two sub-directories. The "Confusion_Matrices" sub-directory contains experiment files for plotting the confusion matrices, and the "Results" sub-directory contains experiment files for printing the performance metrics.

"SVM" directory contains all experiment files involving the SVM classifier and the individual datasets (RAVDESS, TESS, and CREMA-D). These are mainly Jupyter Notebook files from the inital work.

"MLP" directory contains all experiment files involving the MLP classifier and the individual datasets (RAVDESS, TESS, and CREMA-D). These are mainly Jupyter Notebook files from the inital work.

"RNN" directory contains all experiment files involving the RNN classifier and the individual datasets (RAVDESS, TESS, and CREMA-D). These are mainly Jupyter Notebook files from the inital work.

"Noise" directory contains the noise samples (audio files) selected for the noise addition part and also the MATLAB files that were used for the experiments involving noise addition. Three noise samples (Mall, Playground, and Sreets) were added to clean speech files from all three datasets (RAVDESS, TESS, and CREMA-D) using three SNR values (0 dB, 5 dB, and 10 dB).






