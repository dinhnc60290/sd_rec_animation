import requests

import os
from urllib.parse import urlparse
import urllib.request
import json

url = "https://www.youtube.com/shorts/WX3rhcAjbQM?list=PL78sHffDgI2IQrpvyeVY8aUbZ4ZWTVuJ3"

# remove the query string from the URL
url_no_query = urlparse(url)._replace(query=None).geturl()

# get the last path component of the URL
video_id = os.path.basename(urlparse(url_no_query).path)
print(video_id)

url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
querystring = {"videoId":video_id}
headers = {
	"X-RapidAPI-Key": "ed93aabe90msh5e506ae335eb5f9p1b2bc7jsna523689aded7",
	"X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
}
response = requests.get(url, headers=headers, params=querystring)
json_data = json.loads(response.text)
caption = json_data["title"]
list_video = json_data["videos"]["items"]
# Filter the items based on the criteria
filtered_items = [item for item in list_video if item['extension'] == 'mp4' and item['hasAudio'] and item['width'] == max([item['width'] for item in list_video if item['extension'] == 'mp4' and item['hasAudio']]) and item['height'] == max([item['height'] for item in list_video if item['extension'] == 'mp4' and item['hasAudio']])]

# Get the item with the highest resolution
highest_resolution_item = max(filtered_items, key=lambda x: x['width'] * x['height'])

# Print the result
video_url = highest_resolution_item["url"]

if not os.path.exists("videos"):
        os.makedirs("videos")
urllib.request.urlretrieve(
    video_url, os.path.join("videos", "video.mp4"))
