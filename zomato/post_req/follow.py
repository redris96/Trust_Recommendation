import requests

url = "https://www.zomato.com/php/social_load_more.php"


headers = {
    'User-Agent': 'Mozilla/5.0',
}

user_id = 27807550
params = {"entity_id": "27807550", "profile_action":"followedby", "page":"0", "limit":"9"}
r = requests.post(url = url, data = params)

data = r.json()
print data