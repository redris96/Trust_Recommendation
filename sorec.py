import numpy as np
import sys
from sklearn.model_selection import train_test_split

def norm(R):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				R[i][j] = (R[i][j] - 0)/5.0
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
				R[i][j] = (R[i][j] * 5.0) + 0
	return R

def mae(U, V, test, u, itm):
	e = 0.0
	ke = 0.0
	for i in test:
		try:
			val = np.dot(U[u[i[0]],:],V[itm[i[1]],:].T)
			val = bound(val) * 5
			e += np.absolute(val - i[2])
		except KeyError:
			ke += 1
	if ke > 0:
		print "KeyErrors", ke
	return e, e/len(test)

def matrix_factorize(R, C, U, V, Z, K, steps=200, alpha=0.1, beta=0.001, gamma = 10):
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
					# print "w for ", i, weight		
					# weight = 1
					eij = (C[i][j] * weight - a)
					ne += gamma * eij * eij
					U[i,:] = U[i,:] + alpha * (gamma * b * eij * Z[:,j]) 
					Z[:,j] = Z[:,j] + alpha * (gamma * b * eij * U[i,:] - beta * Z[:,j])
					ne += beta * np.dot(Z[:,j] , Z[:,j])
		ne = ne * 0.5
		if ne < 0.001:
			break
		else:
			# print ne
			pass
		if step % 10 == 0:
			print step, "iterations done."
			print "process error", ne
			print "calculating MAE"
			global r_test
			t, e = mae(U, V.T, r_test, ud, itm)
			print "total", t, "MAE", e
	return U, V.T , Z.T, ne



# R = [
# 	 [5.0,2,1,3,0,4,0,0],
# 	 [4,3,0,0,5.0,0,0,0],
# 	 [4,0,2,0,0,0,2,4],
# 	 [0,0,0,0,0,0,0,0],
# 	 [5,1,2,0,4,3,0,0],
# 	 [4,3,0,2,4,0,3,5],
# 	]

# C = [
# 	 [0,0,0,0,0,0],
# 	 [0,0,0,1,0,0],
# 	 [0.8,0,0,0,0,0],
# 	 [0.8,1,0,0,0.6,0],
# 	 [0,0,0.4,0,0,0.8],
# 	 [0,0,0,0,0,0],
# 	]


def data(f,sh, ud, itm,flag):
	# print len(f[:,2]), len(f[:,[0]])
	# a = sp.csr_matrix((f[:,2], (f[:,0],f[:,1])), shape=sh)
	a = np.zeros(sh)
	# itm = {}
	# if flag == 0:
	# 	u = {}
	# else:
	u = ud
	itm = itm
	j = 0
	k = 0
	for i in f:
		if u.has_key(i[0]) == False:
			u[i[0]] = j
			j += 1
		if flag == 1:
			if u.has_key(i[1]) == False:
				u[i[1]] = j
				j += 1
			a[u[i[0]]][u[i[1]]] = i[2]
		else:
			if itm.has_key(i[1]) == False:
				itm[i[1]] = k
				k += 1
			a[u[i[0]]][itm[i[1]]] = i[2]
		# print i[2]
		# sys.exit()
	return a, u, itm

def create_dic(r):
	u = {}
	itm = {}
	j = 0
	k = 0
	for i in r:
		if u.has_key(i[0]) == False:
			u[i[0]] = j
			j += 1
		if itm.has_key(i[1]) == False:
			itm[i[1]] = k
			k += 1
	return u, itm

#data
r_data = np.genfromtxt('rating_short_1_3.txt', dtype=int, delimiter=' ')
t_data = np.genfromtxt('trust_short_1_3.txt', dtype=int, delimiter=' ')
r_train, r_test = train_test_split(r_data, test_size=0.3, random_state=42)

ud, itm = create_dic(r_data)

R, ud, itm = data(r_train, (1000,3000), ud, itm, 0)
C, ud, itm = data(t_data, (1000,1000), ud, itm, 1)

# print type(C)
# quit()


R = np.array(R)
R = norm(R)
C = np.array(C)

N = len(R)
M = len(R[0])

K = 5

U = np.random.rand(N,K)
V = np.random.rand(M,K)
Z = np.random.rand(N,K)

print "finished data pre-processing"

nU, nV, nZ, em = matrix_factorize(R, C, U, V, Z, K)
nR = np.dot(nU,nV.T)
print nR
nnR = bou(nR)
aR = get(nnR)
print aR

# nC = np.dot(nU,nZ.T)
# nnC = bou(nC)


print "process error", em

print "calculating MAE"

t, e = maen(nU, nV, r_test, ud, itm)
print "test len", r_test.shape
print "total", t, "MAE", e