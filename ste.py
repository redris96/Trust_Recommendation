import numpy as np
import sys
from sklearn.model_selection import train_test_split

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

def matrix_factorize(R, U, V, C, K, steps=200, alpha=0.1, beta=0.001,w=0.4):
	V = V.T
	ra = np.zeros(R.shape)
	ne = 0
	for step in xrange(steps):
		ne = 0
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
					# v = np.dot(U[i,:], V[:,j]) * w
					# tot = 0
					# for k in xrange(len(R)):
					# 	if C[i][k] > 0:
					# 		tot += np.dot(U[k,:], V[:,j]) * C[i][k]
					# v += tot * (1-w)
					a = bound(v)
					b = dbound(v)
					eij = a - R[i][j]
					ne += eij * eij
					tot = np.zeros(K)
					for k in xrange(len(R)):
						if C[k][i] > 0:
							if R[k][j] > 0:
								v = ra[k][j]
								# v = np.dot(U[k,:], V[:,j]) * w
								# tota = 0
								# for p in xrange(len(R)):
								# 	if C[k][p] > 0:
								# 		tota += np.dot(U[p,:], V[:,j]) * C[k][p]
								# v += tota * (1-w)
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
		ne *= 0.5
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

	return U, V.T, ne 

# R = [
# 	 [5.0,2,0,3,0,4,0,0],
# 	 [4,3,0,0,5,0,0,0],
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
n_u = 1
r_data = np.genfromtxt('rating_short_'+ str(n_u)+'_'+ str(3*n_u)+'.txt', dtype=int, delimiter=' ')
t_data = np.genfromtxt('trust_short_'+ str(n_u)+'_'+ str(3*n_u)+'.txt', dtype=int, delimiter=' ')
r_train, r_test = train_test_split(r_data, test_size=0.3, random_state=42)

ud, itm = create_dic(r_data)

R, ud, itm = data(r_train, (n_u * 1000,n_u * 3000), ud, itm, 0)
C, ud, itm = data(t_data, (n_u * 1000,n_u * 1000), ud, itm, 1)
print "for",n_u*1000, "users and", n_u*3000, "items"

R = np.array(R)
R = norm(R)
C = np.array(C)

N = len(R)
M = len(R[0])

K = 5

U = np.random.rand(N,K)
V = np.random.rand(M,K)

print "finished data pre-processing"

nU, nV, em = matrix_factorize(R, U, V, C, K)
nR = np.dot(nU,nV.T)
nnR = bou(nR)
aR = get(nnR)
print aR

print "process error", em

print "calculating MAE"

t, e = mae(nU, nV, r_test, ud, itm)
print "test len", r_test.shape
print "total", t, "MAE", e