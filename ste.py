import numpy

def norm(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				R[i][j] = (R[i][j])/5.0
	return R

def bound(x):
	return 1/(1 + numpy.exp(-x))

def dbound(x):
	y = numpy.exp(x)
	return y/pow((1 + y),2)

def matrix_factorize(R, P, Q, C, K, steps=5000, alpha=0.0002, beta=0.001,w=0.4):
	Q = Q.T
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				v = numpy.dot(P[i,:], Q[:,j]) * w
				tot = 0
				for k in xrange(len(R)):
					if C[i][j] > 0:
						tot += numpy.dot(P[k,:], Q[:,j]) * C[i][j]
				v += tot * (1-w)
	for step in xrange(steps):
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					v = numpy.dot(P[i,:], Q[:,j]) * w
					tot = 0
					for k in xrange(len(R)):
						if C[i][j] > 0:
							tot += numpy.dot(P[k,:], Q[:,j]) * C[i][j]
					v += (1-w)*tot
					a = bound(v)
					b = dbound(v)
					eij = R[i][j] - a 
					for k in xrange(K):
						P[i][k] = P[i][k] + alpha * (w* b * eij * Q[k][j] + (1-w)*eij* - beta * P[i][k])
						dtot = 0
						for l in xrange(len(R)):
							if C[i][j] > 0:
								dtot += P[l][k] * C[i][j]
						Q[k][j] = Q[k][j] + alpha * (b * eij * (w*P[i][k] + (1-w)*dtot) - beta * Q[k][j])
		e = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
					for k in xrange(K):
						e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break
	return P, Q.T 


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

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

nP, nQ = matrix_factorize(R, P, Q, C, K)
nR = numpy.dot(nP,nQ.T)
print nR, "\n", R