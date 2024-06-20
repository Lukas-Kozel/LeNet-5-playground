import requests
filepath = "data/5.jpg"
ip_address = "http://127.0.0.1:5000/upload"
with open(filepath, 'rb') as image_file:
    response = requests.post(ip_address,files={'file':image_file})