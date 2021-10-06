import logging, os
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join('/home/ubuntu', 'nss-aws-tests', 'auto.py.log'),
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')

import boto3
def cases():
    client = boto3.client('sqs')
    url = 'https://sqs.eu-central-1.amazonaws.com/960602048864/uploaded_cases'
    msg = client.receive_message(QueueUrl=url)
    logging.info('sqs_uploaded_cases.py: client.receive_message')
    if msg.get('Messages'):
        client.delete_message(QueueUrl=url, ReceiptHandle=msg['Messages'][0]['ReceiptHandle'])
        logging.info('sqs_uploaded_cases.py: client.delete_message')
        payload = eval(msg['Messages'][0]['Body'])
        site = payload['site']
        cases = payload['cases'].split(',')
    else:
        cases = None
    return cases

if __name__ == '__main__':
    print(cases())
