import scrapy
import requests
# from scrapy.http import FormRequest
from bs4 import BeautifulSoup
import json

cookies = {'Cookie':'dpr=1; fbtrack=4c8f6ef3eb528ca8a65a4ec14f09952d; zl=en; fbcity=6; o2menub=seen; PHPSESSID=511f15a3bc4873b6788acd54525803afacda7e11; csrf=751fda4e08ba9f3aa0a0e58345e68ae6; ak_bmsc=B9AD2F46E76C8732B0F459698D73469317201D9526490000BE73095A2465B373~pl5TVwpI7MT8VHcwlrL9a83t7BHoohNJk4+66GOrDSlWBYf5zVcV/bA3UyXHg4Ru+WEEbIRgyKTauIM4ANJGGVnLz5C6YyWgg6YFvbkeMTUASu5RTmWA3l1p4+0rRY6A05+FQaLw3SdCQcyOhQOMvc+e/r0H8IYoIsM3UifZXyBRroWO9I4JOqPnOLs/nyffSwEeDCArcaJKqmE5w8Wpw39hO5Fy0XegA0pT9C4sHlalI=; session_id=655b56ab9553b-207c-4ed0-a172-3e5696ca904d; bm_sv=08CBF21873DC52DB7456F455DB6E3975~EuIIkDgPNzp8lE6pxCbN6VQbettgXPEqLeeNXENX7R4fMkReTjztymYI3/TCbx5NcxnQsJ2VY47VLHqtePTBOjdiIhd7gyOQoohzGDMwJLtbWcp2ASq8YFxAjsJWBoUZoFbxJs8rR/dwCUNRSS03j/oivwV3IvUZk/Bg8lS66uI='}

class ProfileSpider(scrapy.Spider):
    name = "profile"

    def start_requests(self):
        urls = [
        	'https://www.zomato.com/hyderabad'
        	'https://www.zomato.com/hyderabad/restaurants?page=1',
            'https://www.zomato.com/Foodpsyco',
            'https://www.zomato.com/users/mohit-painuly-38215952',
        ]
        # for url in urls:
        #     yield scrapy.Request(url=url, callback=self.parse)

        url = "https://www.zomato.com/php/social_load_more.php"
        user_id = 27807550
        data = {"entity_id": str(user_id), "profile_action":"followedby", "page":'0', "limit":'9'}
        # yield requests.post(url, data=data,cookies=cookies)
        yield scrapy.Request(url=url, method='POST', body=json.dumps(data), callback=self.parse, headers={'Content-Type': 'application/json; charset=UTF-8'}, cookies=cookies)
        # yield scrapy.FormRequest(url=url, callback=self.parse, formdata=params)



    def parse(self, response):
        # profile_id = response.url.split("/")[-1]
        user_id = 27807550
        filename = 'followers_%s.html' % profile_id
        # filename = "followers.html"
        with open(filename, 'wb') as f:
            f.write(response.text)
        self.log('Saved file %s' % filename)