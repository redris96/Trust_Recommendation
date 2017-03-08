import numpy as np

def norm(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				R[i][j] = (R[i][j] - 1.0)/4.0
	return R

def bound(x):
	return 1/(1 + np.exp(-x))

def dbound(x):
	y = np.exp(x)
	return y/pow((1 + y),2)

def bou(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			a = bound(R[i][j])
			R[i][j] = a
	return R

def get(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				R[i][j] = (R[i][j] * 4.0) + 1
	return R

def matrix_factorize(R, C, U, V, Z, K, steps=10000, alpha=0.1, beta=0.001, gamma = 10):
	V = V.T
	Z = Z.T
	e = 0
	ne = 0
	for step in xrange(steps):
		ne = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					y = np.dot(U[i,:], V[:,j])
					a = bound(y)
					b = dbound(y)
					eij = (R[i][j] - a)
					ne += eij * eij
					U[i,:] = U[i,:] + alpha * (b * eij * V[:,j] - beta * U[i,:])
					V[:,j] = V[:,j] + alpha * (b * eij * U[i,:] - beta * V[:,j])
					ne += beta * (np.dot(U[i,:],U[i,:]) + np.dot(V[:,j],V[:,j]))
			for j in xrange(len(C[i])):
				if C[i][j] > 0:
					y = np.dot(U[i,:], Z[:,j])
					a = bound(y)
					b = dbound(y)
					jminus = np.count_nonzero(C[:,j])
					iplus = np.count_nonzero(C[i,:])
					weight = np.sqrt(jminus/ (iplus+jminus + 0.0))
					weight = 1
					eij = (C[i][j] * weight - a)
					ne += gamma * eij * eij
					U[i,:] = U[i,:] + alpha * (gamma * b * eij * Z[:,j]) 
					Z[:,j] = Z[:,j] + alpha * (gamma * b * eij * U[i,:] - beta * Z[:,j])
					ne += beta * np.dot(Z[:,j] , Z[:,j])
		ne = ne * 0.5
		if ne < 0.001:
			break
		else:
			print ne
			pass
	return U, V.T , Z.T, ne


R = [
	 [5.0,2,0,3,0,4,0,0],
	 [4,3,0,0,5.0,0,0,0],
	 [4,0,2,0,0,0,2,4],
	 [0,0,0,0,0,0,0,0],
	 [5,1,2,0,4,3,0,0],
	 [4,3,0,2,4,0,3,5],
	]

C = [
	 [0,0,0,0,0,0],
	 [0,0,0,1,0,0],
	 [0.8,0,0,0,0,0],
	 [0.8,1,0,0,0.6,0],
	 [0,0,0.4,0,0,0.8],
	 [0,0,0,0,0,0],
	]

R = np.array(R)
R = norm(R)
C = np.array(C)

N = len(R)
M = len(R[0])

K = 5

U = np.random.rand(N,K)
V = np.random.rand(M,K)
Z = np.random.rand(N,K)

nU, nV, nZ, e = matrix_factorize(R, C, U, V, Z, K)
nR = np.dot(nU,nV.T)
nnR = bou(nR)
aR = get(nnR)
print aR
nC = np.dot(nU,nZ.T)
nnC = bou(nC)
print "error", e