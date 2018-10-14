import requests
import json
url = ('https://newsapi.org/v2/everything?'
       'q=Delhi Crime&'
       'from=2018-09-14&'
       'sortBy=popularity&'
       'apiKey=PlaceYourAPIKey')

response = requests.get(url)

with open('data.json', 'w') as outfile:
    json.dump(response.json(), outfile)
