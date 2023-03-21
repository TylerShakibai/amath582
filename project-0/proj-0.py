import numpy as np

# Problem 1

A1 = np.matrix([[34, 45], [17, 6]])


# Problem 2
# Initialize A, B, C, D, x, y, and z.  Then do the operations to calculate the variables A2 to A10

A = np.matrix([[1, 2], [-1, 1]])
B = np.matrix([[2, 0], [0, 2]])
C = np.matrix([[2, 0, -3], [0, 0, -1]])
D = np.matrix([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0]).reshape(-1, 1)
y = np.array([0, 1]).reshape(-1, 1)
z = np.array([1, 2, -1]).reshape(-1, 1)

A2 = A + B
A3 = (3*x - 4*y)
A4 = np.matmul(A, x)
A5 = np.matmul(B, (x - y))
A6 = np.matmul(D, x)
A7 = np.matmul(D, y) + z
A8 = np.matmul(A, B)
A9 = np.matmul(B, C)
A10 = np.matmul(C, D)