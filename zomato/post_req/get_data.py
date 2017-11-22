import numpy as np
import sys
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import scipy.sparse
import os

#initial users list
init = set()
f = open('initial.txt', 'r')
for line in f:
	line = line.strip()
	init.add(line)


check = 0
flag = 1
if flag == 1:
	ratings_data = open("ratings_data2.txt","w")
	trust_data = open("trust_data2.txt","w")
else:
	ratings_data = open("unmapped_ratings_data.txt","w")
	trust_data = open("unmapped_trust_data.txt","w")

# user-user-trust
# c_x = []
# c_y = []
# count=1
# user_map = {}
# user_count = 1
# for filename in os.listdir('users'):
# 	# print filename
# 	name = filename.split("_")
# 	user_id = name[0]
# 	# if count == 1:
# 	# 	print user_id
# 	if not user_map.has_key(user_id):
# 		user_map[user_id] = user_count
# 		user_count += 1
# 	# print user_id
# 	file = open('users/'+ filename)
# 	if name[1] == "follows":
# 		# print "in trusts"
# 		for line in file:
# 			line = line.strip()
# 			user_other = line
# 			# c_x.append(user_id)
# 			# c_y.append(usr_other)
# 			if user_other in init:
# 				if not user_map.has_key(user_other):
# 					user_map[user_other] = user_count
# 					user_count += 1
# 				if flag == 1:
# 					trust_data.write(str(user_map[user_id]) + " " + str(user_map[user_other]) + " 1\n")
# 				else:
# 					trust_data.write(str(user_id) + " " + str(user_other) + " 1\n")
# 			# print usr_other
# 	else:
# 		# print "in trustedby"
# 		# print name[1]
# 		for line in file:
# 			line = line.strip()
# 			user_other = line
# 			# c_x.append(usr_other)
# 			# c_y.append(user_id)
# 			if user_other in init:
# 				if not user_map.has_key(user_other):
# 					user_map[user_other] = user_count
# 					user_count += 1
# 				if flag == 1:
# 					trust_data.write(str(user_map[user_other]) + " " + str(user_map[user_id]) + " 1\n")
# 				else:
# 					trust_data.write(str(user_other) + " " + str(user_id) + " 1\n")

# 			# print usr_other
# 	if check == 1:
# 		count += 1
# 		if count == 4:
# 			break
# # c_data = np.ones(np.size(c_x))
# print "users: ", user_count

#user-item-rating
item_map = {}
itm_count = 1
r_x = []
r_y = []
r_data = []
count=1
for filename in os.listdir('reviews'):
	# print filename
	user_id = int(filename.split("_")[0])
	# print user_id
	file = open('reviews/'+ filename)
	for line in file:
		line = line.strip()
		col = line.split(" ")
		itm_id = col[0].strip()
		#map it to integer
		if not item_map.has_key(itm_id):
			item_map[itm_id] = itm_count
			itm_count += 1
		itm_rating = col[1].strip()
		# try:
		if not user_map.has_key(user_id):
			user_map[user_id] = user_count
			user_count += 1
			if count == 1:
				print user_id
				count += 1
		if flag == 1:
			ratings_data.write(str(user_map[user_id]) + " " + str(item_map[itm_id]) + " "+ str(float(itm_rating)) +"\n")
		else:
			ratings_data.write(str(user_id) + " " + str(itm_id) + " "+ str(float(itm_rating)) +"\n")
		# except:
		# 	continue
		# r_x.append(user_id)
		# r_y.append(item_map[itm_id])
		# r_data.append(itm_rating)
	if check == 1:
		count += 1
		if count == 4:
			break

print "users: ", user_count
print "items: ", itm_count
#buffer
b = 10
# print c_x, c_y, c_data
# R = coo_matrix((r_data, (r_x, r_y)))  #, shape=(users+b, items+b))
# C = coo_matrix((c_data, (c_x, c_y)))  #, shape=(users+b, users+b))

# scipy.sparse.save_npz('ratings_sparse.npz', R)
# scipy.sparse.save_npz('trust_sparse.npz', C)

print "done"

# if "31826510" in init:
# 	print "yeas"
# print type(init[0])