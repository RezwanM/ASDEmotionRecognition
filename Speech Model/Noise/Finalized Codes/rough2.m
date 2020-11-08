%Specify the folder where the clean files live.
myFolder = 'C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples';
%'fullfile()' joins directory path string with '*.wav' extension.
filePattern = fullfile(myFolder, '*.wav'); % Change to whatever pattern you need.
%'theFiles' is an array containing info of each file, such as name, directory path, date created, size, etc.
theFiles = dir(filePattern);
theFiles(2);

%Specify the folder where the noise files live.
myNoiseFolder = 'C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Finalized Noise Samples';
%'fullfile()' joins directory path string with '*.wav' extension.
noiseFilePattern = fullfile(myNoiseFolder, '*.wav'); % Change to whatever pattern you need.
%'theNoiseFiles' is an array containing info of all three files, such as name, directory path, date created, size, etc.
theNoiseFiles = dir(noiseFilePattern);

length(theNoiseFiles);
theNoiseFiles(1).name;
[filepath,name,ext] = fileparts(theNoiseFiles(1).name);
oldName = name;
newName = oldName+string('_')+string(1);
newNameExt = newName + string('.wav');