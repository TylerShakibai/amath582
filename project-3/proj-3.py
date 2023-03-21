import numpy as np
import scipy
import matplotlib.pyplot as plt

Xt1_1 = scipy.io.loadmat('Xt1_1.mat')['Xt1_1']
Xt2_1 = scipy.io.loadmat('Xt2_1.mat')['Xt2_1']
Xt3_1 = scipy.io.loadmat('Xt3_1.mat')['Xt3_1']

Xt1_1 = Xt1_1[:, :len(Xt1_1[0])]
Xt2_1 = Xt2_1[:, :len(Xt1_1[0])]
Xt3_1 = Xt3_1[:, :len(Xt1_1[0])]

X1 = np.concatenate((Xt1_1, Xt2_1, Xt3_1))
X1_mean = np.mean(X1, axis=1)
X1_n = X1 - np.matrix(X1_mean).T

svd1 = np.linalg.svd(X1_n, full_matrices=False)
u1 = svd1[0]
s1 = svd1[1]
vh1 = svd1[2]
A1 = vh1

nrgs1 = (s1**2)/(np.linalg.norm(s1)**2)
A2 = nrgs1.reshape(-1, 1)

fig1 = plt.figure()
plt.semilogy(nrgs1, '.')
plt.xlabel('Index')
plt.ylabel('Energy')
plt.title('Log-Scaled Energies for Test #1')

s1_diag = np.diag(s1)
r = 2
X1_rank2 = (u1[:, 0:r] @ s1_diag[0:r, 0:r]) @ vh1[0:r, :]
A3 = X1_rank2

fig2 = plt.figure()
for r in range(1, 4):
    X1_rankr = (u1[:, 0:r] @ s1_diag[0:r, 0:r]) @ vh1[0:r, :]
    plt.plot(X1_rankr[0].T, label=f'n = {r}')

plt.xlabel('Time')
plt.ylabel('Height')
plt.title('Rank-n Approximations for Test #1')
plt.legend()


Xt1_2 = scipy.io.loadmat('Xt1_2.mat')['Xt1_2']
Xt2_2 = scipy.io.loadmat('Xt2_2.mat')['Xt2_2']
Xt3_2 = scipy.io.loadmat('Xt3_2.mat')['Xt3_2']

Xt1_2 = Xt1_2[:, :len(Xt1_2[0])]
Xt2_2 = Xt2_2[:, :len(Xt1_2[0])]
Xt3_2 = Xt3_2[:, :len(Xt1_2[0])]

X2 = np.concatenate((Xt1_2, Xt2_2, Xt3_2))
X2_mean = np.mean(X2, axis=1)
X2_n = X2 - np.matrix(X2_mean).T

svd2 = np.linalg.svd(X2_n, full_matrices=False)
u2 = svd2[0]
s2 = svd2[1]
vh2 = svd2[2]
A4 = vh2

nrgs2 = (s2**2)/(np.linalg.norm(s2)**2)
A5 = nrgs2.reshape(-1, 1)

fig3 = plt.figure()
plt.semilogy(nrgs2, ".")
plt.xlabel("Index")
plt.ylabel("Energy")
plt.title('Log-Scaled Energies for Test #2')

s2_diag = np.diag(s2)
r = 3
X2_rank3 = (u2[:, 0:r] @ s2_diag[0:r, 0:r]) @ vh2[0:r, :]
A6 = X2_rank3

fig4 = plt.figure()
for r in range(1, 5):
    X2_rankr = (u2[:, 0:r] @ s2_diag[0:r, 0:r]) @ vh2[0:r, :]
    plt.plot(X2_rankr[0].T, label=f'n = {r}')

plt.xlabel('Time')
plt.ylabel('Height')
plt.title('Rank-n Approximations for Test #2')
plt.legend()


Xt1_3 = scipy.io.loadmat('Xt1_3.mat')['Xt1_3']
Xt2_3 = scipy.io.loadmat('Xt2_3.mat')['Xt2_3']
Xt3_3 = scipy.io.loadmat('Xt3_3.mat')['Xt3_3']

Xt1_3 = Xt1_3[:, :len(Xt3_3[0])]
Xt2_3 = Xt2_3[:, :len(Xt3_3[0])]
Xt3_3 = Xt3_3[:, :len(Xt3_3[0])]

X3 = np.concatenate((Xt1_3, Xt2_3, Xt3_3))
X3_mean = np.mean(X3, axis=1)
X3_n = X3 - np.matrix(X3_mean).T

svd3 = np.linalg.svd(X3_n, full_matrices=False)
u3 = svd3[0]
s3 = svd3[1]
vh3 = svd3[2]
A7 = vh3

nrgs3 = (s3**2)/(np.linalg.norm(s3)**2)
A8 = nrgs3.reshape(-1, 1)

fig5 = plt.figure()
plt.semilogy(nrgs3, ".")
plt.xlabel("Index")
plt.ylabel("Energy")
plt.title('Log-Scaled Energies for Test #3')

s3_diag = np.diag(s3)
r = 3
X3_rank3 = (u3[:, 0:r] @ s3_diag[0:r, 0:r]) @ vh3[0:r, :]
A9 = X3_rank3

fig6 = plt.figure()
for r in range(1, 5):
    X3_rankr = (u3[:, 0:r] @ s3_diag[0:r, 0:r]) @ vh3[0:r, :]
    plt.plot(X3_rankr[0].T, label=f'n = {r}')

plt.xlabel('Time')
plt.ylabel('Height')
plt.title('Rank-n Approximations for Test #3')
plt.legend()


Xt1_4 = scipy.io.loadmat('Xt1_4.mat')['Xt1_4']
Xt2_4 = scipy.io.loadmat('Xt2_4.mat')['Xt2_4']
Xt3_4 = scipy.io.loadmat('Xt3_4.mat')['Xt3_4']

Xt1_4 = Xt1_4[:, :len(Xt1_4[0])]
Xt2_4 = Xt2_4[:, :len(Xt1_4[0])]
Xt3_4 = Xt3_4[:, :len(Xt1_4[0])]

X4 = np.concatenate((Xt1_4, Xt2_4, Xt3_4))
X4_mean = np.mean(X4, axis=1)
X4_n = X4 - np.matrix(X4_mean).T

svd4 = np.linalg.svd(X4_n, full_matrices=False)
u4 = svd4[0]
s4 = svd4[1]
vh4 = svd4[2]
A10 = vh4

nrgs4 = (s4**2)/(np.linalg.norm(s4)**2)
A11 = nrgs4.reshape(-1, 1)

fig7 = plt.figure()
plt.semilogy(nrgs4, ".")
plt.xlabel("Index")
plt.ylabel("Energy")
plt.title('Log-Scaled Energies for Test #4')

s4_diag = np.diag(s4)
r = 2
X4_rank2 = (u4[:, 0:r] @ s4_diag[0:r, 0:r]) @ vh4[0:r, :]
A12 = X4_rank2

fig8 = plt.figure()
for r in range(1, 4):
    X4_rankr = (u4[:, 0:r] @ s4_diag[0:r, 0:r]) @ vh4[0:r, :]
    plt.plot(X4_rankr[0].T, label=f'n = {r}')

plt.xlabel('Time')
plt.ylabel('Height')
plt.title('Rank-n Approximations for Test #4')
plt.legend()

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)
# print("A6:", A6)
# print("A7:", A7)
# print("A8:", A8)
# print("A9:", A9)
# print("A10:", A10)
# print("A11:", A11)
# print("A12:", A12)
# plt.show()