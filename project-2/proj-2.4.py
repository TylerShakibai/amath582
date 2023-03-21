import numpy as np
import scipy
import matplotlib.pyplot as plt

data = np.squeeze(scipy.io.loadmat('CP2_SoundClip.mat')['y'])
Fs = 44100
w = int(len(data)/4)
S4 = data[(4-1)*w:4*w]

L = len(S4)/Fs
n = len(S4)
t = np.arange(0, L, 1/Fs)
tau = np.arange(0, L, 0.1)
k = (2*np.pi/(2*L))*np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)), axis=0)
k_shifted = scipy.fft.fftshift(k)

Sgt_spec4 = np.zeros([k_shifted.size, tau.size])
a = 400
ranges = np.arange(0, 1800)

for i in range(tau.size):
    g = np.exp(-a*(t - tau[i])**2)
    Sg = g*S4
    Sgt = scipy.fft.fft(Sg)
    peak_index = np.argmax(abs(Sgt[ranges]))
    filter_center = k[peak_index]
    g = np.exp(-(1/L)*(abs(k) - filter_center)**2)
    Sgt_filtered = g*Sgt
    Sgt_spec4[:, i] = scipy.fft.fftshift(abs(Sgt_filtered))

A4 = Sgt_spec4
TAU, KS = np.meshgrid(tau, k_shifted)

fig4 = plt.figure()
plt.pcolormesh(TAU, KS, np.log(Sgt_spec4 + 1))
plt.ylim(0, 800)
plt.title('Spectrogram for Window #4')

# print("A4:", A4)
# plt.show()