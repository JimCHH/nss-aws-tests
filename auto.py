import logging
logging.basicConfig(
    level=logging.INFO, 
    filename='auto.py.log',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')
logging.info('Get started')

import os
os.chdir('codev2-4')

import time
from api_lambda_sqs import sqs
while True:
    try:
        # print(sqs.cases())
        # time.sleep(10)
        os.system(f'python VHIT_test.py {" ".join(sqs.cases())}')
    except:
        break

import requests
with open('../.url') as f:
    api = f.readline()
logging.info(requests.get(f'{api}/stop').text.replace('\n', '  ') + '\n==============================================================')