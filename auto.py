import logging
logging.basicConfig(
    level=logging.INFO, 
    filename='auto.log',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')
logging.info('Get started')

import os
os.chdir('codev2-4')

queue = ['20200202_H14_NSS62471', '20200204_H14_NSS62471']
while queue:
    os.system(f'python VHIT_test.py {" ".join(queue)}')
    queue = []

import requests
with open('../.url') as f:
    api = f.readline()
logging.info(requests.get(f'{api}/stop').text.replace('\n', '  ') + '\n==============================================================')