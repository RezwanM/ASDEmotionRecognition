%Load noise file. Store default sampling rate in fs.
[n, fsn] = audioread('C:/Books/Texas State Books/Spring 2020/Thesis B/Noise Samples/Classroom_Ambiance.wav');
%Use just single (mono) channel of noise signal.
nMono = n(:,1);

nPowMono = bandpower(nMono);
nPowMono;

Npts = length(nMono);
Noise_Power = sum(abs(nMono).*abs(nMono))/Npts;
Noise_Power;


a = [1,2,3,4,5,6];
asize = length(a);
a(4:6);


%Generate a random no. from a range.
xmin=0.8;
xmax=4;
n=1;
x=xmin+rand(1,n)*(xmax-xmin);

%Generate random integers within a range.
randi([-10 10], 1, 6);

%Pick a number randomly from a list.
a = [1,2,3,4,5,6];
num = a(randi(length(a)))