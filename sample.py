import random
import sys

# users_index = random.sample(range(1,49290), 7000)
# items_index = random.sample(range(1,139738), 21000)

# ratings_file = "dataset/ratings_data.txt"
# trust_file = "dataset/trust_data.txt"

users = 75887
items = 112476

users_index = random.sample(range(1,users), 3000)
items_index = random.sample(range(1,items),9000)

users_index = set(users_index)
items_index = set(items_index)

ratings_file = "dataset3/ratings_data.txt"
trust_file = "dataset3/trust_data.txt"

r_data = open(ratings_file, 'r')
t_data = open(trust_file, 'r')
r_short = open('rating3_short.txt', 'w')
t_short = open('trust3_short.txt', 'w')

# print users_index[:10], items_index[:10]
# sys.exit(0)

usr_idx = []
for line in r_data:
	u = line.split(" ")
	# print int(u[0])
	# sys.exit()
	if int(u[0]) in users_index and int(u[1]) in items_index:
	# if int(u[1]) in items_index:
		# if u[0] not in usr_idx:
		# 	usr_idx.append(u[0])
		r_short.write(line)

for i, line in enumerate(t_data):
	u = line.split(" ")
	# print u
	# sys.exit()
	try:
		if int(u[0]) in users_index and int(u[1]) in users_index:
		# if int(u[1]) in usr_idx and int(u[2]) in usr_idx:
			t_short.write(line)
	except ValueError:
		print i
		sys.exit()
