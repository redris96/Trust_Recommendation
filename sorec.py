import numpy

def norm(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				R[i][j] = (R[i][j] - 1.0)/4.0
	return R

def bound(x):
	return 1/(1 + numpy.exp(-x))

def dbound(x):
	y = numpy.exp(x)
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
	# ne = 0
	for step in xrange(steps):
		ne = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					y = numpy.dot(U[i,:], V[:,j])
					a = bound(y)
					b = dbound(y)
					eij = (R[i][j] - a)
					# ne += eij * eij
					for k in xrange(K):
						U[i][k] = U[i][k] + alpha * (b * eij * V[k][j] - beta * U[i][k])
						V[k][j] = V[k][j] + alpha * (b * eij * U[i][k] - beta * V[k][j])
						# ne += beta * (U[i][k]*U[i][k] + V[k][j]*V[k][j])
			for j in xrange(len(C[i])):
				if C[i][j] > 0:
					y = numpy.dot(U[i,:], Z[:,j])
					a = bound(y)
					b = dbound(y)
					jminus = numpy.count_nonzero(C[:,j])
					iplus = numpy.count_nonzero(C[i,:])
					weight = numpy.sqrt(jminus/ (iplus+jminus + 0.0))
					weight = 1
					eij = (C[i][j] * weight - a)
					# ne += gamma * eij * eij
					for k in xrange(K):
						U[i][k] = U[i][k] + alpha * (gamma * b * eij * Z[k][j]) 
						Z[k][j] = Z[k][j] + alpha * (gamma * b * eij * U[i][k] - beta * Z[k][j])
						# ne += gamma * Z[k][j] * Z[k][j]
		# ne = ne * 0.5
		e = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					y = numpy.dot(U[i,:], V[:,j])
					a = bound(y)
					e = e + pow(R[i][j] - a, 2)
					for k in xrange(K):
						e = e + (beta/2) * (pow(U[i][k],2) + pow(V[k][j],2))
		for i in xrange(len(C)):
			for j in xrange(len(C[i])):
				if C[i][j] > 0:
					y = numpy.dot(U[i,:], Z[:,j])
					a = bound(y)
					jminus = numpy.count_nonzero(C[:,j])
					iplus = numpy.count_nonzero(C[i,:])
					weight = numpy.sqrt(jminus/ (iplus+jminus + 0.0))
					weight = 1
					e = e + (gamma/2)*pow(C[i][j] * weight - a, 2)
					for k in xrange(K):
						e = e + (beta/2) * (pow(Z[k][j],2))
		if e < 0.001:
			# print e
			break
		else:
			# print e
			pass
	return U, V.T , Z.T, e


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

R = numpy.array(R)
R = norm(R)
C = numpy.array(C)

N = len(R)
M = len(R[0])

K = 5

U = numpy.random.rand(N,K)
V = numpy.random.rand(M,K)
Z = numpy.random.rand(N,K)

nU, nV, nZ, e = matrix_factorize(R, C, U, V, Z, K)
# print nU, "\n", nV
nR = numpy.dot(nU,nV.T)
nnR = bou(nR)
aR = get(nnR)
# print nnR, "\n", R
print aR
nC = numpy.dot(nU,nZ.T)
nnC = bou(nC)
# print nnC
# print nC, "\n", C
print "error", e