import os
os.chdir(os.path.join(os.path.expanduser('~'), 'nss-aws-tests', 'codev2_4'))

import logging
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join(os.path.expanduser('~'), 'nss-aws-tests', 'auto.py.log'),
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')

from api_lambda_sqs import sqs
import time
idle = 0
while idle < 60:
    try:
        command = f'python VHIT_test.py {" ".join(sqs.cases())}'
        os.system(command)
        logging.info(command)
        idle = 0
        # logging.info(sqs.cases())
        # time.sleep(3)
    except KeyError:
        logging.info('Messages available: 0')
        time.sleep(60)
        idle += 1
    except Exception as f_ck:
        logging.error(f_ck)
        time.sleep(60)
        idle += 1

import requests
logging.info(requests.get('https://deqg3un8ha.execute-api.eu-central-1.amazonaws.com/stop').text.replace('\n', ' '))