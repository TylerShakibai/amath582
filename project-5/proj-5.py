import numpy as np
import scipy

data = scipy.io.loadmat('CP5_data.mat')['data']
X1 = data[:, :-1]
X2 = data[:, 1:]

u, sig, v = np.linalg.svd(X1, full_matrices=False)
S = u.T @ X2 @ v.T @ np.diag(1/sig)
A1 = S

slices = 676
t = np.linspace(0, 2*np.pi, slices)
dt = t[1] - t[0]

mu, vec = np.linalg.eig(S)
omega = np.log(mu)/dt
phi = u @ vec

m = np.min(np.abs(omega))
index = np.argmin(np.abs(omega))
mv = vec[:, index]
phi = (u @ mv).reshape(-1, 1)

y0 = (np.linalg.pinv(phi).dot(X1[:, 0])).real
u_modes = np.zeros((len(y0), len(t)))
for i in range(len(t)):
    u_modes[:, i] = y0 * np.exp(m * t[i])

u_dmd = phi * u_modes
A2 = m
A3 = u_dmd[:, 338].reshape(-1, 1)

back = u_dmd[:, :slices]
fore = data - back

A4 = np.abs(back[:, 337]).reshape(-1, 1)
A5 = np.abs(fore[:, 338]).reshape(-1, 1)

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)