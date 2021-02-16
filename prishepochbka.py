import librosa
import numpy as np
import math 

n = 1024
buffer= np.ndarray([n], dtype=float)
PI = 3.1416
nu = 100

for x in range(n):
    buffer[x] = math.sin(2.0*PI*nu*x / 8192.0)

fur = np.fft.fft(buffer)
imagine = [x.imag for x in fur]
real = [x.real for x in fur]

#print(imagine)
#print(real)
col=10
print([el for el in fur[:col]])

mfcc =[x[0] for x in librosa.feature.mfcc(buffer,n_mfcc=20,win_length=n,hop_length=n*2)]

print(mfcc)