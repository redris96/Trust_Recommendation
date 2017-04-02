import random
import sys

users_index = random.sample(range(1,49290), 3000)
items_index = random.sample(range(1,139738), 9000)

ratings_file = "dataset/ratings_data.txt"
trust_file = "dataset/trust_data.txt"

r_data = open(ratings_file, 'r')
t_data = open(trust_file, 'r')
r_short = open('rating_short.txt', 'w')
t_short = open('trust_short.txt', 'w')

# print users_index[:10], items_index[:10]
# sys.exit(0)


for line in r_data:
	u = line.split(" ")
	# print int(u[0])
	# sys.exit()
	if int(u[0]) in users_index and int(u[1]) in items_index:
		r_short.write(line)

for i, line in enumerate(t_data):
	u = line.split(" ")
	# print u
	# sys.exit()
	try:
		if int(u[1]) in users_index and int(u[2]) in users_index:
			t_short.write(line[1:])
	except ValueError:
		print i
		sys.exit()
