import requests
from os.path import join, expanduser

with open(join(expanduser('~'), '.api')) as f:
    API = f.readline()

queue = []

while queue:
    ...

print(requests.get(f'{API}/stop').text)