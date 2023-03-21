import numpy as np
import scipy
import matplotlib.pyplot as plt

data = np.squeeze(scipy.io.loadmat('CP2_SoundClip.mat')['y'])
Fs = 44100

fig = plt.figure()
plt.plot(np.arange(0, len(data))/Fs, data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Raw Frequency Data')

w = int(len(data)/4)

S1 = data[(1-1)*w:1*w]
S2 = data[(2-1)*w:2*w]
S3 = data[(3-1)*w:3*w]
S4 = data[(4-1)*w:4*w]

L = len(S1)/Fs
n = len(S1)
t = np.arange(0, L, 1/Fs)
tau = np.arange(0, L, 0.1)
k = (2*np.pi/(2*L))*np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)), axis=0)
k_shifted = scipy.fft.fftshift(k)

Sgt_spec1 = np.zeros([k_shifted.size, tau.size])
Sgt_spec2 = np.zeros([k_shifted.size, tau.size])
Sgt_spec3 = np.zeros([k_shifted.size, tau.size])
Sgt_spec4 = np.zeros([k_shifted.size, tau.size])
a = 400
ranges = np.arange(0, 1800)

S = data
L = len(data)/Fs
n = len(data)
t = np.arange(0, L, 1/Fs)
k = (1/L)*np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)), axis=0)

S_freq = scipy.fft.fftn(S)
S_freq[abs(S_freq) > 250] = 0
A5 = scipy.fft.ifft(S_freq).reshape(-1, 1)

S_freq = scipy.fft.fftn(S)
S_freq[((abs(S_freq) > 800) & (abs(S_freq) < 250)).any()] = 0
A6 = scipy.fft.ifft(S_freq).reshape(-1, 1)

# print("A5:", A5)
# print("A6:", A6)
# plt.show()