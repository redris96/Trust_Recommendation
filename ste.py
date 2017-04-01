import numpy as np

def norm(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				R[i][j] = (R[i][j])/5.0
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
				R[i][j] = (R[i][j] * 5.0)
	return R

def matrix_factorize(R, U, V, C, K, steps=200, alpha=0.1, beta=0.001,w=0.4):
	V = V.T
	ra = np.zeros(R.shape)
	ne = 0
	for step in xrange(steps):
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					v = np.dot(U[i,:], V[:,j]) * w
					tot = 0
					for k in xrange(len(R)):
						if C[i][k] > 0:
							tot += np.dot(U[k,:], V[:,j]) * C[i][k]
					v += tot * (1-w)
					ra[i][j] = v
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					v = ra[i][j]
					a = bound(v)
					b = dbound(v)
					eij = a - R[i][j]
					ne += eij * eij
					tot = np.zeros(K)
					for k in xrange(len(R)):
						if C[k][i] > 0:
							v = ra[k][j]
							ak = bound(v)
							bk = dbound(v)
							keij = ak - R[k][j]
							tot += keij*bk*C[k][i]*V[:,j]
					U[i,:] = U[i,:] - alpha*(w*b*eij*V[:,j] + (1-w)*tot + beta*U[i,:])
					tot = 0
					for k in xrange(len(R)):
						if C[i][k] > 0:
							tot += C[i][k] * U[k,:]
					# tot = np.dot(C[i,:], U)
					V[:,j] = V[:,j] - alpha*(eij*b*(w*U[i,:] + (1-w)*tot) + beta*V[:,j])
					ne += beta * (np.dot(U[i,:],U[i,:]) + np.dot(V[:,j], V[:,j]))
		if ne < 0.001 or ne > 50:
			break
		else:
			print ne
	return U, V.T 


R = [
	 [5.0,2,0,3,0,4,0,0],
	 [4,3,0,0,5,0,0,0],
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

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nU, nV = matrix_factorize(R, P, Q, C, K)
nR = np.dot(nU,nV.T)
nnR = bou(nR)
aR = get(nnR)
print aR