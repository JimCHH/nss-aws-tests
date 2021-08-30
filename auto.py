import logging
logging.basicConfig(
    level=logging.INFO, 
    filename='auto.py.log',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')
logging.info('開始')

import os
os.chdir('codev2_4')

from api_lambda_sqs import sqs
import time
while True:
    try:
        command = f'python VHIT_test.py {" ".join(sqs.cases())}'
        os.system(command)
        logging.info(command)
        # print(sqs.cases())
        # time.sleep(10)
    except:
        break

logging.info('結束\n')