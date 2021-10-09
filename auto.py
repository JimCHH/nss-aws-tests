import logging, os
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join('/home/ubuntu', 'nss-aws-tests', 'auto.py.log'),
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')

os.chdir(os.path.join('/home/ubuntu', 'nss-aws-tests'))
with open('aws/api') as f:
    url = f.readline()
from aws import sqs_uploaded_cases
from csv_corrector import correct
os.chdir(os.path.join('/home/ubuntu', 'nss-aws-tests', 'tests'))

import time
idle = 0
while idle < 10:
    cases = sqs_uploaded_cases.cases()
    if cases:
        for case in cases:

            try:
                os.system('sudo umount /home/ubuntu/S3')
                os.system('s3fs neurobit-asg /home/ubuntu/S3 -o passwd_file=/home/ubuntu/.aws/credentials-s3fs -o uid=1000 -o gid=1000')
                date_site_patient = case
                source = f'/home/ubuntu/S3/{date_site_patient.split("_")[1]}/Result/{date_site_patient}'
            except Exception as e:
                logging.info(e)

            test123 = f'python Test123_mp4_segmentation_extraction_pkl_visualization_pdf.py {case}'
            logging.info(test123)
            try:
                os.system(test123)
                os.system(f'rm {source}/*.pkl')
            except Exception as e:
                logging.info(e)

            test4 = f'python Test4_VHIT_test.py {case}'
            logging.info(test4)
            try:
                correct(case)
                os.system(test4)
                os.system(f'rm {source}/*.CSV')
            except Exception as e:
                logging.info(e)

        idle = 0
    else:
        time.sleep(60)
        idle += 1

import requests
requests.get(url)