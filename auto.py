import logging, os
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join('/home/ubuntu', 'nss-aws-tests', 'auto.py.log'),
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')

os.chdir(os.path.join('/home/ubuntu', 'nss-aws-tests'))
from api_lambda_sqs import sqs_uploaded_cases
from csv_corrector import correct
os.chdir(os.path.join('/home/ubuntu', 'nss-aws-tests', 'codev2_4'))

import time
idle = 0
while idle < 60:
    cases = sqs_uploaded_cases.cases()
    if cases:
        for case in cases:
            test4 = f'python VHIT_test.py {case}'
            logging.info(test4)
            try:
                os.system('sudo umount /home/ubuntu/S3')
                os.system('s3fs neurobit-asg /home/ubuntu/S3 -o passwd_file=/home/ubuntu/.aws/credentials-s3fs -o uid=1000 -o gid=1000')
                correct(case)
                os.system(test4)
            except Exception as e:
                logging.info(e)
        idle = 0
    else:
        time.sleep(60)
        idle += 1

import requests
logging.info(requests.get('https://deqg3un8ha.execute-api.eu-central-1.amazonaws.com/stop').text.replace('\n', ' '))