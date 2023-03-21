import numpy as np
import scipy
import pywt
import matplotlib.pyplot as plt

training_images = scipy.io.loadmat('CP4_training_images.mat')['training_images']
training_labels = scipy.io.loadmat('CP4_training_labels.mat')['training_labels']
Training_DWT = scipy.io.loadmat('Training_DWT.mat')['Training_DWT']

test_images = scipy.io.loadmat('CP4_test_images.mat')['test_images']
test_labels = scipy.io.loadmat('CP4_test_labels.mat')['test_labels']

training_images = training_images.reshape(784, 30000, order ='f')

def dc_wavelet(dcfile):
    m, n = dcfile.shape
    pxl = int(np.sqrt(m))
    nw = m // 4
    dcData = np.zeros((nw, n))
    for k in range(n):
        X = dcfile[:, k].reshape(pxl, pxl).astype('float')
        cA, (cH, cV, cD) = pywt.dwt2(X, 'haar')
        cod_cH1 = np.abs(cH).astype('float')
        cod_cV1 = np.abs(cV).astype('float')
        cod_edge = (cod_cH1 + cod_cV1)/2
        cod_edge = cod_edge.reshape(nw, 1)
        dcData[:, k] = cod_edge.flatten()
    return dcData

u, s, v = np.linalg.svd(Training_DWT, full_matrices=False)

fig1 = plt.figure()
plt.semilogy(s[0:20], ".")
plt.xlabel("Index")
plt.ylabel("Singular Values")
plt.title('Singular Values of Training Data')

A1 = 15
A2 = u[:, 0:15]

train_0 = []
train_1 = []
# y = u @ Training_DWT
y = np.diag(s) @ v

for i in range(len(training_labels)):
    if training_labels[i] == 0:
        train_0.append(y[0:15, i])
    elif training_labels[i] == 1:
        train_1.append(y[0:15, i])

train_0 = np.asarray(train_0).T
train_1 = np.asarray(train_1).T

nd = train_0.shape[1]
nc = train_1.shape[1]
md = np.mean(train_0, axis=1)
mc = np.mean(train_1, axis=1)

Sw = np.zeros((train_0.shape[0], train_0.shape[0]))
for k in range(nd):
    Sw += np.outer(train_0[:, k] - md, train_0[:, k] - md)
for k in range(nc):
    Sw += np.outer(train_1[:, k] - mc, train_1[:, k] - mc)

Sb = np.outer(md - mc, md - mc)

A3 = Sw
A4 = Sb

vals, vecs = scipy.linalg.eig(Sb, Sw)
index = np.argmax(np.abs(vals))
w = vecs[:, index]

w = w/np.linalg.norm(w, ord=2)
A5 = w.reshape(15, 1)
w = -w

vtrain_0 = w.T @ train_0
vtrain_1 = w.T @ train_1

sort_0 = np.sort(vtrain_0)
sort_1 = np.sort(vtrain_1)

t1 = len(sort_0) - 1
t2 = 0

while sort_0[t1] > sort_1[t2]:
    t1 -= 1
    t2 += 1

thresh = (sort_0[t1] + sort_1[t2])/2
A6 = thresh

test_images = test_images.reshape(784, 5000)
images_filtered = []
for i in range(5000):
    if (test_labels[i] == 0) or (test_labels[i] == 1):
        images_filtered.append(test_images[:, i])

images_filtered = np.asarray(images_filtered).T
testSet = images_filtered
testNum = testSet.shape[1]

testWave = dc_wavelet(testSet)
testMatrix = A2.T @ testWave
pVal = np.dot(w, testMatrix)

ResVecBool = pVal > thresh
ResVec = ResVecBool.astype(int)
A7 = ResVec.reshape(1, -1)

hidden_labels = []

for i in range (len(test_labels)):
    if (test_labels[i] == 0) or (test_labels[i] == 1):
        hidden_labels.append(test_labels[i])
        
hidden_labels = np.asarray(hidden_labels)

error = np.abs(ResVec.reshape(-1, 1) - hidden_labels)
errNum = np.sum(error)
sucRate = 1 - errNum/testNum

fig2 = plt.figure()
fig2.suptitle('First 4 Principal Components')
for k in range(1, 5):
    ax = fig2.add_subplot(2, 2, k)
    u1 = np.reshape(u[:, k-1], (14, 14))
    u2 = np.interp(u1, (u1.min(), u1.max()), (0, 1))
    ax.imshow(u2, cmap='gray')

fig3 = plt.figure()
fig3.suptitle('Training Numbers Grid')
for k in range(1, 10):
    ax = fig3.add_subplot(3, 3, k)
    training_images1 = np.reshape(training_images[:, k-1], (28, 28)).T
    ax.imshow(training_images1)

fig3.tight_layout(pad=1)

fig4 = plt.figure()
fig4.suptitle('Test Numbers Grid')
for k in range(1, 10):
    ax = fig4.add_subplot(3, 3, k)
    test_images1 = np.reshape(test_images[:, k-1], (28, 28))
    ax.imshow(test_images1)

fig4.tight_layout(pad=1)

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)
# print("A6:", A6)
# print("A7:", A7)
# plt.show()