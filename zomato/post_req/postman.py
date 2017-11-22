import requests
import json
from bs4 import BeautifulSoup

url = "https://www.zomato.com/php/social_load_more.php"

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i 

def get_data(user_id, type_of):
	payload = {'entity_id': str(user_id), 'profile_action':str(type_of), 'page':'0', 'limit':'9000'}
	# payload = ""
	cookies = {'Cookie':'dpr=1; fbtrack=4c8f6ef3eb528ca8a65a4ec14f09952d; zl=en; fbcity=6; o2menub=seen; PHPSESSID=511f15a3bc4873b6788acd54525803afacda7e11; session_id=null; csrf=7e4a5022efe5a3adefa96001e68d8753; ak_bmsc=DEB95B9D516522E98850486470B3187C17201D5D6C59000062550B5AEC27FE26~plSzp3yiOXsM0AAzIGywhXNTa0SPV9StoGiHdjZ90Q6ZdsLF5SdSXoJrTT3TVjDWTZtFTEWeOoiQs3QyDwZhfyg5A63B9pR8AkbbGYag+GW+1fKD+OLZ/c6Q8l6C6FeWXov8gal+uTKfKrCp5zwY35Lm/r8vcmhc5US9ppkXD+29O2dcAPPVHpRiaG6/yfgBlukT7OxncRh1MbXYGWf9YH8wCnT9dG6gFHi4OxX7dG9/0=; bm_sv=CD4D573466A0F0D84C3613E8BC3970FE~bj5XahNv1nDOcGf9kDTYdrJdz5V/Uak4LoQMTbM4DduSp+xhsXPNsUYJtJKy9ThvEQmBA+2e2ORHUOiyeAeVwSxf/N8OMUFf6SmueGoz6D95C2blNPHTvilQFs6t/wxpK3oq3ZzSwvVbYNhGkIkmfn1F0VnhRsx/KY9vdT3RDzY='}
	headers = {
	    'content-type': "application/x-www-form-urlencoded",
	    # 'cache-control': "no-cache",
	    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36',
	    }

	response = requests.post(url, data=payload, headers=headers,cookies=cookies)
	return response.text

def extract_data(user_id,data, type_of):
	data = json.loads(data)
	html = data["html"]
	soup = BeautifulSoup(html, 'html.parser')
	if type_of != "user-reviews":
		ids = soup.find_all('a',{"data-entity_id":True})
		# data_e = []
		name = str(user_id)+'_'+str(type_of)
		f = open('users/'+name,'w')
		for n in ids:
			# data_e.append(int(n.get('data-entity_id')))
			f.write(str(n.get('data-entity_id'))+'\n')
	else:
		ids = soup.find_all('a',{"data-entity_id":True})
		divs = soup.find_all('div',{"class":"rev-text"})
		name = str(user_id)+'_reviews'
		f = open('reviews/'+name,'w')
		for n,div in zip(ids, divs):
			rat_div = div.contents[1]
			try:
				rating = float(rat_div['aria-label'].split()[1])
				f.write(str(n['data-entity_id'])+' '+str(rating)+'\n')
			except:
				continue


types = ["follows", "followedby","user-reviews"]
flag=2	
if flag == 1:
	type_now = ["user-reviews"]
else:
	type_now = ["follows", "followedby"]

def main():
	f = open('initial.txt','r')
	if flag != 1:
		log_file = 'done.txt'
	else:
		log_file = 'done_review.txt'
	done = open(log_file,'a')
	count = file_len(log_file)
	for i in xrange(count-1):
		f.next()
	for line in f:
		user_id = int(line)
		for type_of in type_now:
			output = get_data(user_id, type_of)
			# print output
			extract_data(user_id, output, type_of)
		done.write(str(user_id)+'\n')


if __name__ == "__main__": main()