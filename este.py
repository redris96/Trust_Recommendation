import numpy as np
import sys
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, lil_matrix

def norm(R):
	for i in xrange(len(R)):
		R[i] = (R[i] - 0.0)/5.0
		# for j in xrange(len(R[i])):
		# 	if R[i][j] > 0:
		# 		R[i][j] = (R[i][j] - 0)/5.0
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

def matrix_factorize(R, U, V, C, K, steps=400, alpha=0.1, beta=0.001,w=0.4):
	V = V.T
	ra = lil_matrix(R.shape)
	Rl = R.tolil()
	ne = 0
	pre_e = 10
	for step in xrange(steps):
		ne = 0
		for i,j,val in zip(R.row, R.col, R.data):
			v = np.dot(U[i,:], V[:,j]) * w
			tot = 0
			# for k in xrange(len(R)):
			# 	if C[i][k] > 0:
			nzk = np.nonzero(C.getrow(i))[0]
			for k in nzk:
				tot += np.dot(U[k,:], V[:,j]) * C[i,k]
			v += tot * (1-w)
			ra[i,j] = v
		for i,j,val in zip(R.row, R.col, R.data):
			v = ra[i,j]
			# v = np.dot(U[i,:], V[:,j]) * w
			# tot = 0
			# for k in xrange(len(R)):
			# 	if C[i][k] > 0:
			# 		tot += np.dot(U[k,:], V[:,j]) * C[i][k]
			# v += tot * (1-w)
			a = bound(v)
			b = dbound(v)
			eij = a - val
			ne += eij * eij
			tot = np.zeros(K)
			# for k in xrange(len(R)):
			# 	if C[k][i] > 0:
			nzk = np.nonzero(C.getcol(i))[0]
			for k in nzk:
				if Rl[k,j] > 0:
					v = ra[k,j]
					# v = np.dot(U[k,:], V[:,j]) * w
					# tota = 0
					# for p in xrange(len(R)):
					# 	if C[k][p] > 0:
					# 		tota += np.dot(U[p,:], V[:,j]) * C[k][p]
					# v += tota * (1-w)
					ak = bound(v)
					bk = dbound(v)
					keij = ak - Rl[k,j]
					tot += keij*bk*C[k,i]*V[:,j]
			U[i,:] = U[i,:] - alpha*(w*b*eij*V[:,j] + (1-w)*tot + beta*U[i,:])
			tot = 0
			# for k in xrange(len(R)):
			# 	if C[i][k] > 0:
			nzk = np.nonzero(C[i,:])[0]
			for k in nzk:
				tot += C[i,k] * U[k,:]
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
			if pre_e < e:
				print pre_e, e
				break
			pre_e = e

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
print "for",n_u*1000, "users and", n_u*3000, "items"
r_data = np.genfromtxt('rating_short_'+ str(n_u)+'_'+ str(3*n_u)+'.txt', dtype=float, delimiter=' ')
t_data = np.genfromtxt('trust_short_'+ str(n_u)+'_'+ str(3*n_u)+'.txt', dtype=float, delimiter=' ')

user = np.unique(np.append(r_data[:,0],[t_data[:,0], t_data[:,1]]))
items = np.unique(r_data[:,1])
# print items
# if 46465 not in user and 46465  in t_data[:,1]:
# 	print "NO"
N = user.shape[0]
M = items.shape[0]
# print "N,M", N,M
# sys.exit()
ud = dict(zip(user, np.arange(N)))
itm = dict(zip(items, np.arange(M)))
# print len(ud), len(itm)
# print ud
# sys.exit()
# i,j,rdata = np.hsplit(r_train, 3)
# i = i.flatten
# j = j.flatten
# rdata = rdata.flatten

r_train, r_test = train_test_split(r_data, test_size=0.2, random_state=42)

# ud, itm = create_dic(r_data)

# R, ud, itm = data(r_train, (n_u * 1000,n_u * 3000), ud, itm, 0)
# C, ud, itm = data(t_data, (n_u * 1000,n_u * 1000), ud, itm, 1)
# print "for",n_u*1000, "users and", n_u*3000, "items"

# M = 49290
# N = 139738
# r_data = np.genfromtxt('dataset/ratings_data.txt', dtype=int, delimiter=' ')
# t_data = np.genfromtxt('dataset/trust_data.txt', dtype=int, delimiter=' ')
# r_train, r_test = train_test_split(r_data, test_size=0.3, random_state=42)

# R = np.array(R)
# i,j,rdata = np.hsplit(r_train, 3)
# i = i.flatten
# j = j.flatten
# rdata = rdata.flatten
r_train[:,2] = norm(r_train[:,2])
# print r_train[:,0][:5]
# sys.exit()
x = r_train[:,0]
y = r_train[:,1]
p = t_data[:,0]
q = t_data[:,1]

# print "ah", itm[19793]
# print x.shape
for k,v in ud.iteritems():
	x[x == k] = v
	p[p == k] = v
	q[q == k] = v
for k,v in itm.iteritems():
	y[y == k] = v
# print np.max(p), np.max(q)
R = coo_matrix((r_train[:,2], (x,y)) , shape = (n_u*1000, n_u*3000))
C = coo_matrix((t_data[:,2], (p,q)) , shape = (n_u*1000, n_u*3000))
# R = R.tolil()
C = C.tolil()
# N = len(R)
# M = len(R[0])
s = R.shape
N = s[0]
M = s[1]

K = 5

U = np.random.rand(N,K)
V = np.random.rand(M,K)

print "finished data pre-processing"

nU, nV, em = matrix_factorize(R, U, V, C, K)
# nR = np.dot(nU,nV.T)
# nnR = bou(nR)
# aR = get(nnR)
# print aR

print "process error", em

print "calculating MAE"

t, e = mae(nU, nV, r_test, ud, itm)
print "test len", r_test.shape
print "total", t, "MAE", e