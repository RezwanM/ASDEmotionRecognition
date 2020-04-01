%Load noisy file. Store default sampling rate in fs.
[s, fss] = audioread('C:/Books/Texas State Books/Fall 2019/Thesis A/Corpus/Simulated/RAVDESS/Classroom_Ambience_5dB/Classroom_Ambience_5dB.wav');

%Load clean speech file.
[c, fsc] = audioread('C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/All_RAVDESS/03-01-01-01-01-01-01.wav');

%Load noise file. Store default sampling rate in fs.
[n, fsn] = audioread('C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Classroom_Ambiance.wav');
%Use just single (mono) channel of noise signal.
nMono = n(:,1);

%Find the no. of rows of clean speech array.
len = size(c, 1);
%Resize noise signal to length of clean speech.
nMonoResize = nMono(1:len,:);

%Calculate Power of clean speech.
powClean = bandpower(c);

%Calculate Power of noise (mono).
powNoise = bandpower(nMonoResize);

%Calculate Noise Variance (var) for a given SNR.
SNR = 15;
var = (powClean/powNoise)*10^(-SNR/10);

%Noise Standard Deviation (std).
std = sqrt(var);
  
%New adjusted noise signal.
nNew = std.*nMonoResize;
  
%Add noise signal to clean speech to create noisy signal.
s = c + nNew;

SnR = snr(c, s-c);
SnR