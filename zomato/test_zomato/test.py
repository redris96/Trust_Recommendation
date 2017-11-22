import requests
import urllib2

url = 'https://www.zomato.com/hyderabad'
url2 = "http://quotes.toscrape.com/"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

# page = requests.get(url).text

# opener = urllib2.build_opener()
# opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0')]
# response = opener.open(url)

req = urllib2.Request(url, data=None, headers=headers)
response = urllib2.urlopen(req)

page = response.read()
print page