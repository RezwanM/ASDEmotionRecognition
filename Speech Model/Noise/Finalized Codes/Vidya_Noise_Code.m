clc;
clear;
close all;


%CLEAN SPEECH
[signal,fs1] = audioread('Clean_Speech.wav'); %fs1 = 48kHz

%NOISE FILE
[city_noise,fs4] = audioread('Small_Crowd-Mike_Koenig-2101396541.mp3'); %fs4 = 44.1kHz

%Use just single (mono) channel for noise signal.
city_noise = city_noise(:,1);

%Resample noise file to 48kHz
city_noise = resample(city_noise,160,147);

%SIGNAL POWER
sig_power = sum(signal.^2)/length(signal);


%CITY TRAFFIC NOISE POWER
city_power = sum(city_noise(1:length(signal)).^2)/length(signal);

%DEFINE SNR LEVEL
SNR = 10;

%NOISE VARIANCE
k = (sig_power/city_power)*10^(-(SNR/10));

%REDUCE NOISE POWER BEFORE ADDING
power_reduced_city = sqrt(k)*city_noise(1:length(signal));

%REDUCED NOISE POWER
power_reduced_city_power = sum(power_reduced_city.^2)/length(signal);

%NOISY SIGNAL
sig_city = signal + power_reduced_city(1:length(signal));

%PLAY NOISY FILE AUDIO
sound(sig_city,fs4);
%pause(2);

%WRITE NOISY SIGNAL INTO FILE
audiowrite('C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Playground_Noisy_Signal_10dB.wav',sig_city,fs1);

%CHECK SNR USING MATLAB FUNCTION
ss3 = snr((signal),(power_reduced_city))

%CHECK SNR MANUALLY
snr_result3 = 10*log10(sig_power/power_reduced_city_power)