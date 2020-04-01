% Specify the folder where the files live.
myFolder = 'C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/All_RAVDESS';
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.wav'); % Change to whatever pattern you need.
theFiles = dir(filePattern);

%Load noise file. Store default sampling rate in fs.
[n, fsn] = audioread('C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Party_Crowd.wav');
%Use just single (mono) channel of noise signal.
nMono = n(:,1);

for k = 1 : length(theFiles)
  
  %Clear previous warning.
  lastwarn('');
  
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  
  % Now do whatever you want with this file name,
  
  %Load clean speech file.
  [c, fsc] = audioread(fullFileName);
  
  %Calculate Power of clean speech.
  powClean = bandpower(c);
  
  %Find the no. of rows of clean speech array.
  len = size(c, 1);
  %Resize noise signal to length of clean speech.
  nMonoResize = nMono(1:len,:);
  
  %Calculate Power of noise (mono).
  powNoise = bandpower(nMonoResize);
  
  %Calculate Noise Variance (var) for a given SNR.
  SNR = -25;
  var = (powClean/powNoise)*10^(-SNR/10);
  
  %Normalize the signals.
  %cNorm = c / max(abs(c));
  %nNorm = nMonoResize / max(abs(nMonoResize));
  
  %Noise Standard Deviation (std).
  std = sqrt(var);
  
  %New adjusted noise signal.
  nNew = std.*nMonoResize;
  
  %Add noise signal to clean speech to create noisy signal.
  s = c + nNew;
  
  %Save noisy signal as WAV file.
  fprintf(1, 'Good, Now reading %s\n', fullFileName);
  audiowrite(fullFileName,s,fsc);
  
  %Check which warning occured (if any)
  [msgstr, msgid] = lastwarn;
  %This is when the clipping warning occurs. Fix: normalize the signal.
  switch msgid
   case 'MATLAB:audiovideo:audiowrite:dataClipped'
      %Normalize the output signal.
      s1 = s / max(abs(s));
      %Save noisy signal as WAV file.
      audiowrite(fullFileName,s1,fsc);
      fprintf(1, 'Warning, Now reading %s\n', fullFileName);
  end
end