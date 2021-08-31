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
while True:
    try:
        command = f'python VHIT_test.py {" ".join(sqs.cases())}'
        os.system(command)
        logging.info(command)
        # logging.info(sqs.cases())
        # time.sleep(3)
    except Exception as f_ck:
        logging.error(f_ck)
        break