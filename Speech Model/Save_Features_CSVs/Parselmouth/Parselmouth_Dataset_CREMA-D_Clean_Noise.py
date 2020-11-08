#************************************************************************************
# Rezwan Matin
# Thesis B
# Filename: Parselmouth_Dataset_CREMA-D_Clean_Noise.py
# Date: 6/22/20
#
# Objective:
# 36 features - Functionals on Pitch (f0), Loudness(Intensity), MFCC, formants, formants BW, HNR, Jitter, and Shimmer.
#
#*************************************************************************************

import parselmouth
from parselmouth.praat import call
import glob
import os
import pandas as pd
import numpy as np

# All features together!
# Pitch, Loudness (Intensity), MFCCs, Formants, Jitter, Shimmer, and HNR
def extractFeatures(voiceID, f0min, f0max, unit):
    
    sound = parselmouth.Sound(voiceID) # read the sound
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
	
    # Pitch (f0)
    pitch = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max, time_step=0.016) # 21 values per frame
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit) # get standard deviation
    p20F0 = call(pitch, "Get quantile", 0, 0, 0.20, unit) # 20th percentile
    p50F0 = call(pitch, "Get quantile", 0, 0, 0.50, unit) # 50th percentile
    p80F0 = call(pitch, "Get quantile", 0, 0, 0.80, unit) # 80th percentile
    rng2080F0 = p80F0 - p20F0 # Range, from 20th percentile to 80th percentile
    
    # Loudness (Intensity)
    loudness = sound.to_intensity(minimum_pitch=f0min, time_step=0.016) # 1 value per frame
    meanloud = call(loudness, "Get mean", 0, 0) # get mean loudness
    stdevloud = call(loudness, "Get standard deviation", 0, 0) # get standard deviation
    p20loud = call(loudness, "Get quantile", 0, 0, 0.20) # 20th percentile
    p50loud = call(loudness, "Get quantile", 0, 0, 0.50) # 50th percentile
    p80loud = call(loudness, "Get quantile", 0, 0, 0.80) # 80th percentile
    rng2080loud = p80loud - p20loud  # Range, from 20th percentile to 80th percentile
    
    # MFCC (1-4)
    mfcc = sound.to_mfcc(number_of_coefficients=4, window_length=0.032, time_step=0.016, firstFilterFreqency=100.0, distance_between_filters=100.0) # 4 values per frame
    mfcc_array = mfcc.to_array() # take everything into a Numpy array
    meanmfcc = np.mean(mfcc_array.T, axis=0)
    sdmfcc = np.std(mfcc_array.T, axis=0)
    
    # Formants (f1, f2, and f3) & their Bandwidths
    formants = sound.to_formant_burg(time_step=0.016, max_number_of_formants=4, maximum_formant=5500.0, window_length=0.032, pre_emphasis_from=50.0)
    numPoints = call(pointProcess, "Get number of points")
    
    file_list = []
    f1_list = []
    f2_list = []
    f3_list = []
    f1_bw_list = []
    f2_bw_list = []
    f3_bw_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_bw = call(formants, "Get bandwidth at time", 1, t, 'Hertz', 'Linear')
        f2_bw = call(formants, "Get bandwidth at time", 2, t, 'Hertz', 'Linear')
        f3_bw = call(formants, "Get bandwidth at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f1_bw_list.append(f1_bw)
        f2_bw_list.append(f2_bw)
        f3_bw_list.append(f3_bw)
        
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f1_bw_list = [f1_bw for f1_bw in f1_bw_list if str(f1_bw) != 'nan']
    f2_bw_list = [f2_bw for f2_bw in f2_bw_list if str(f2_bw) != 'nan']
    f3_bw_list = [f3_bw for f3_bw in f3_bw_list if str(f3_bw) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = np.mean(f1_list)
    f2_mean = np.mean(f2_list)
    f3_mean = np.mean(f3_list)
    f1_bw_mean = np.mean(f1_bw_list)
    f2_bw_mean = np.mean(f2_bw_list)
    f3_bw_mean = np.mean(f3_bw_list)
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_sd = np.std(f1_list)
    f2_sd = np.std(f2_list)
    f3_sd = np.std(f3_list)
    f1_bw_sd = np.std(f1_bw_list)
    f2_bw_sd = np.std(f2_bw_list)
    f3_bw_sd = np.std(f3_bw_list)
    
    # Jitter and Shimmer
    jitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.032, 1.3) # local absolute jitter
    shimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.032, 1.3, 1.6) # local dB shimmer
    
    # HNR
    hnr = sound.to_harmonicity_cc(time_step=0.016, minimum_pitch=f0min, silence_threshold=0.1, periods_per_window=4.5)
    meanhnr = call(hnr, "Get mean", 0, 0)
    sdhnr = call(hnr, "Get standard deviation", 0, 0)
	
    return meanF0, stdevF0, p20F0, p50F0, p80F0, rng2080F0, meanloud, stdevloud, p20loud, p50loud, p80loud, rng2080loud, meanmfcc, sdmfcc, f1_mean, f2_mean, f3_mean, f1_sd, f2_sd, f3_sd, f1_bw_mean, f2_bw_mean, f3_bw_mean, f1_bw_sd, f2_bw_sd, f3_bw_sd, jitter, shimmer, meanhnr, sdhnr



# create lists to put the results
label_list = []
mean_F0_list = []
sd_F0_list = []
p20_F0_list = []
p50_F0_list = []
p80_F0_list = []
rng2080_F0_list = []
mean_loud_list = []
sd_loud_list = []
p20_loud_list = []
p50_loud_list = []
p80_loud_list = []
rng2080_loud_list = []
mean_mfcc1_list = []
mean_mfcc2_list = []
mean_mfcc3_list = []
mean_mfcc4_list = []
sd_mfcc1_list = []
sd_mfcc2_list = []
sd_mfcc3_list = []
sd_mfcc4_list = []
f1_mean_list = []
f2_mean_list = []
f3_mean_list = []
f1_bw_mean_list = []
f2_bw_mean_list = []
f3_bw_mean_list = []
f1_sd_list = []
f2_sd_list = []
f3_sd_list = []
f1_bw_sd_list = []
f2_bw_sd_list = []
f3_bw_sd_list = []
jitter_list = []
shimmer_list = []
hnr_mean_list = []
hnr_sd_list = []


# Save directory path in 'path'
path = r'/home/r_m727/All_Files/Corpus/Simulated/CREMA-D/Clean_Noise'

# Create a list of audio file names 'file_list'
file_list = os.listdir(path)

i=0

# Go through all the wave files in the folder and measure pitch
for filename in file_list:
    sound = parselmouth.Sound(path + '/' + file_list[i])
    sound = sound.resample(new_frequency=16000)
    # Save emotion label from file name. 'path' contains directory's address, 'file_list' contains file name, and '/' joins the two to form file's address
    label = os.path.splitext(os.path.basename(path + '/' + file_list[i]))[0].split('-')[2]
    
    # Get audio and apply feature extraction
    (meanF0, stdevF0, p20F0, p50F0, p80F0, rng2080F0, meanloud, stdevloud, p20loud, p50loud, p80loud, rng2080loud, meanmfcc, sdmfcc, f1_mean, f2_mean, f3_mean, f1_sd, f2_sd, f3_sd, f1_bw_mean, f2_bw_mean, f3_bw_mean, f1_bw_sd, f2_bw_sd, f3_bw_sd, jitter, shimmer, meanhnr, sdhnr) = extractFeatures(sound, 94, 600, "LogHertz") # pitch floor, pitch ceiling, unit

    # Append features to lists...
    label_list.append(label)
    
    #Pitch (f0)
    mean_F0_list.append(meanF0)
    sd_F0_list.append(stdevF0)
    p20_F0_list.append(p20F0)
    p50_F0_list.append(p50F0)
    p80_F0_list.append(p80F0)
    rng2080_F0_list.append(rng2080F0)

    # Loudness (Intensity)
    mean_loud_list.append(meanloud)
    sd_loud_list.append(stdevloud)
    p20_loud_list.append(p20loud)
    p50_loud_list.append(p50loud)
    p80_loud_list.append(p80loud)
    rng2080_loud_list.append(rng2080loud)

    # MFCC (1-4)
    mean_mfcc1_list.append(meanmfcc[1])
    mean_mfcc2_list.append(meanmfcc[2])
    mean_mfcc3_list.append(meanmfcc[3])
    mean_mfcc4_list.append(meanmfcc[4])
    sd_mfcc1_list.append(sdmfcc[1])
    sd_mfcc2_list.append(sdmfcc[2])
    sd_mfcc3_list.append(sdmfcc[3])
    sd_mfcc4_list.append(sdmfcc[4])

    # Formants (f1, f2, and f3) & their Bandwidths
    f1_mean_list.append(f1_mean)
    f2_mean_list.append(f2_mean)
    f3_mean_list.append(f3_mean)
    f1_bw_mean_list.append(f1_bw_mean)
    f2_bw_mean_list.append(f2_bw_mean)
    f3_bw_mean_list.append(f3_bw_mean)
    f1_sd_list.append(f1_sd)
    f2_sd_list.append(f2_sd)
    f3_sd_list.append(f3_sd)
    f1_bw_sd_list.append(f1_bw_sd)
    f2_bw_sd_list.append(f2_bw_sd)
    f3_bw_sd_list.append(f3_bw_sd)

    # Jitter and Shimmer
    jitter_list.append(jitter)
    shimmer_list.append(shimmer)

    # HNR
    hnr_mean_list.append(meanhnr)
    hnr_sd_list.append(sdhnr)
    
    i+=1

# Arrange all feature lists into a Pandas dataframe
# Save data into a Pandas dataframe
df = pd.DataFrame(np.column_stack([mean_F0_list, sd_F0_list, p20_F0_list, p50_F0_list, p80_F0_list, rng2080_F0_list, mean_loud_list, sd_loud_list, p20_loud_list, p50_loud_list, p80_loud_list, rng2080_loud_list, mean_mfcc1_list, mean_mfcc2_list, mean_mfcc3_list, mean_mfcc4_list, sd_mfcc1_list, sd_mfcc2_list, sd_mfcc3_list, sd_mfcc4_list, f1_mean_list, f2_mean_list, f3_mean_list, f1_sd_list, f2_sd_list, f3_sd_list, f1_bw_mean_list, f2_bw_mean_list, f3_bw_mean_list, f1_bw_sd_list, f2_bw_sd_list, f3_bw_sd_list, jitter_list, shimmer_list, hnr_mean_list, hnr_sd_list, label_list]), 
                               columns=['f0_mean_Hz', 'f0_sd_Hz', 'f0_p20_Hz', 'f0_p50_Hz', 'f0_p80_Hz', 'f0_rng2080_Hz', 'Loud_mean_dB', 'Loud_sd_dB', 'Loud_p20_dB', 'Loud_p50_dB', 'Loud_p80_dB', 'Loud_rng2080_dB', 'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'MFCC4_mean', 'MFCC1_sd', 'MFCC2_sd', 'MFCC3_sd', 'MFCC4_sd', 'f1_mean', 'f2_mean', 'f3_mean', 'f1_sd', 'f2_sd', 'f3_sd', 'f1_bw_mean', 'f2_bw_mean', 'f3_bw_mean', 'f1_bw_sd', 'f2_bw_sd', 'f3_bw_sd', 'Jitter_loc_abs', 'Shimmer_loc_dB', 'HNR_mean', 'HNR_sd', 'Emotion'])
df

# Delete unnecessary emotion data (calm)
df.drop(df[df['Emotion'] == 2.0].index, inplace = True)
df['Emotion'].replace({1.0: "Neutral", 3.0: "Happy", 4.0: "Sad", 5.0: "Angry", 6.0: "Fearful", 7.0: "Disgust", 8.0: "Surprised"}, inplace=True)
# Reset row (sample) indexing
df = df.reset_index(drop=True)
                     
df.to_csv('CREMA-D_Parselmouth_Clean_Noise.csv')

print("Done!")                   
