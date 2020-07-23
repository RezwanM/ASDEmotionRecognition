%Specify the folder where the clean files live.
myFolder = 'C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/All_RAVDESS';
%'fullfile()' joins directory path string with '*.wav' extension.
filePattern = fullfile(myFolder, '*.wav'); % Change to whatever pattern you need.
%'theFiles' is an array containing info of each file, such as name, directory path, date created, size, etc.
theFiles = dir(filePattern);

%Specify the folder where the noise files live.
myNoiseFolder = 'C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Finalized Noise Samples';
%'fullfile()' joins directory path string with '*.wav' extension.
noiseFilePattern = fullfile(myNoiseFolder, '*.wav'); % Change to whatever pattern you need.
%'theNoiseFiles' is an array containing info of all three files, such as name, directory path, date created, size, etc.
theNoiseFiles = dir(noiseFilePattern);


for k = 1 : length(theFiles)
  
  %Clear previous warning.
  lastwarn('');
  
  %'.name' only extracts the file names (with extensions) from string.
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  
  %'.name' only extracts the file names (with extensions) from string.
  %Randomly select one out of the three noise files.
  baseNoiseFileName = theNoiseFiles(randi(length(theNoiseFiles))).name;
  fullNoiseFileName = fullfile(myNoiseFolder, baseNoiseFileName);
  
  %Load clean speech file and downsample from 48kHz to 16kHz.
  [cl, fsc] = audioread(fullFileName);
  c = resample(cl, 1, 3);
  
  %Load noise file and downsample from 44.1kHz to 16kHz.
  [no, fsn] = audioread(fullNoiseFileName);
  n = resample(no, 160, 441);
  %Use just single (mono) channel of noise signal.
  nMono = n(:,1);
  
  %Randomly select a section of the noise file equal to clean speech size.
  x = 1; %Number of values picked randomly.
  sMin = 1; %Minimum allowed starting value.
  sMax = length(nMono) - length(c); %Maximum allowed starting value.
  s = randi([sMin,sMax], x);
  %Resize noise signal to length of clean speech.
  nMonoResize = nMono(s:s+length(c)-1,:);
  
  %Calculate Power of clean speech.
  powClean = sum(c.^2)/length(c);
  
  %Calculate Power of noise (mono).
  powNoise = sum(nMonoResize.^2)/length(nMonoResize);
  
  %Randomly select one out of the three SNR values, -10dB, -5dB or 0dB.
  snrOptions = [10, 5];
  SNR = snrOptions(randi(length(snrOptions)));
  
  %Calculate Noise Variance (var) for a given SNR.
  var = (powClean/powNoise)*10^(-SNR/10);
  
  %Noise Standard Deviation (std).
  std = sqrt(var);
  
  %New adjusted noise signal.
  nNew = std.*nMonoResize;
  
  %Add noise signal to clean speech to create noisy signal.
  p = c + nNew;
  
  %Update file name according to noise file and SNR used.
  [filepath,name,ext] = fileparts(baseFileName);
  [filepathNoise,nameNoise,extNoise] = fileparts(baseNoiseFileName);
  oldName = name;
  newName = oldName + string('_') + string(nameNoise) + string('_') + string(SNR);
  newNameExt = newName + string('.wav');
  newFileName = fullfile(myFolder, newNameExt);
  
  %Save noisy signal as WAV file and export at 16kHz.
  fprintf(1, 'Good, Now reading %s\n', newFileName);
  audiowrite(newFileName,p,16000);
  
  %Check which warning occured (if any)
  [msgstr, msgid] = lastwarn;
  %This is when the clipping warning occurs. Fix: normalize the signal.
  switch msgid
   case 'MATLAB:audiovideo:audiowrite:dataClipped'
      %Normalize the output signal.
      p1 = p / max(abs(p));
      %Save noisy signal as WAV file.
      audiowrite(newFileName,p1,16000);
      fprintf(1, 'Warning, Now reading %s\n', newFileName);
  end
  
end