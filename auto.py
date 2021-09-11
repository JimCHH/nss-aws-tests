import logging, os
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join('/home/ubuntu', 'nss-aws-tests', 'auto.py.log'),
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')

os.chdir(os.path.join('/home/ubuntu', 'nss-aws-tests'))
from api_lambda_sqs import sqs_uploading_cases
from csv_corrector import correct
os.chdir(os.path.join('/home/ubuntu', 'nss-aws-tests', 'codev2_4'))

import time
idle = 0
while idle < 60:
    cases = sqs_uploading_cases.cases()
    if cases:
        correct(cases)
        test4 = 'python VHIT_test.py ' + ' '.join(cases)
        os.system(test4)
        logging.info(test4)
        idle = 0
    else:
        time.sleep(60)
        idle += 1

import requests
logging.info(requests.get('https://deqg3un8ha.execute-api.eu-central-1.amazonaws.com/stop').text.replace('\n', ' '))
