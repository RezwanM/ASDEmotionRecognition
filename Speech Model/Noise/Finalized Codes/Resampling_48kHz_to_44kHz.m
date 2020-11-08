%Load noise file. Store default sampling rate in fs.
[n, fsn] = audioread('C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Classroom_Ambiance.wav');

% Specify the folder where the files live.
myFolder = 'C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/All_RAVDESS';
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.wav'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  
  % Now do whatever you want with this file name,
  
  %Load clean speech file.
  [c, fsc] = audioread(fullFileName);
  
  %Resample original signal from 48kHz to 44.1kHz to match noise SR.
  c1 = resample(c,147,160);
  
  %Normalize signal.
  c2 = c1 / max(abs(c1)); 
  
  %Save noisy signal as WAV file.
  audiowrite(fullFileName,c2,fsn);
end