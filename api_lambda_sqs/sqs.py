import boto3

def cases():
    sqs = boto3.client('sqs')
    url = 'https://sqs.eu-central-1.amazonaws.com/960602048864/uploading_cases.fifo'
    msg = sqs.receive_message(QueueUrl=url)
    payload = eval(msg['Messages'][0]['Body'])
    site = payload['site']
    cases = payload['cases'].split(',')
    return cases

if __name__ == '__main__':
    print(cases())