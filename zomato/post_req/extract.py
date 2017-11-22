import json
from bs4 import BeautifulSoup

with open('output.json') as data_file:
	data = json.load(data_file)

html = data["html"]
# print html
soup = BeautifulSoup(html, 'html.parser')
ids = soup.find_all('a',{"data-entity_id":True})
divs = soup.find_all('div',{"class":"rev-text"})
for n,div in zip(ids, divs):
	rat_div = div.contents[1]
	rating = float(rat_div['aria-label'].split()[1])
	print n['data-entity_id'],rating
# for div in divs:
# 	rat = div.contents[1]
# 	tex = float(rat['aria-label'].split()[1])
# 	print tex