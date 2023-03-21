import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

data = scipy.io.loadmat('Kraken.mat')['Kraken']
L = 10
n = 64

x2 = np.linspace(-L, L, n + 1)
x = x2[0:n]
y = x
z = x

k = (2*np.pi/(2*L))*np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)), axis=0)
k_shifted = scipy.fft.fftshift(k)

[Y, X, Z] = np.meshgrid(k, k, k)
[Ky, Kx, Kz] = np.meshgrid(k_shifted, k_shifted, k_shifted)

Uk = np.zeros((n, n, n))
for i in range(49):
    Un = data[:, i].reshape(n, n, n, order='f')
    M = np.amax(abs(Un))
    Uk = Uk + scipy.fft.fftn(Un)

A1 = Uk
A2 = Uk/49

argMax = np.argmax(Uk)
index = np.unravel_index(argMax, Uk.shape)

A3 = X[index]
A4 = Y[index]
A5 = Z[index]

gaussian = scipy.fft.ifftshift(np.exp(-(1/L)*((Kx - X[index])**2 + (Ky - Y[index])**2 +(Kz - Z[index])**2)))
A6 = gaussian

x_pos = np.zeros(49); y_pos = np.zeros(49); z_pos = np.zeros(49)
for t in range(49):
    Un = data[:, t].reshape(n, n, n, order='f')
    Uf = scipy.fft.ifftn(scipy.fft.fftn(Un) * gaussian)
    modulus = np.argmax(np.abs(Uf))
    filterIndex = np.unravel_index(modulus, Uf.shape)
    x_pos[t] = x[filterIndex[0]]
    y_pos[t] = y[filterIndex[1]]
    z_pos[t] = z[filterIndex[2]]

A7 = x_pos.reshape(1, -1)
A8 = y_pos.reshape(1, -1)
A9 = z_pos.reshape(1, -1)

table = pd.DataFrame(data = {'x': x_pos, 'y': y_pos, 'z': z_pos}, index = range(1, 50))

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Kraken Location in 3D')
ax.plot3D(x_pos, y_pos, z_pos)

fig = plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Projection of Kraken Location ')
plt.plot(x_pos, y_pos)

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)
# print("A6:", A6)
# print("A7:", A7)
# print("A8:", A8)
# print("A9:", A9)
# print(table)
# plt.show()