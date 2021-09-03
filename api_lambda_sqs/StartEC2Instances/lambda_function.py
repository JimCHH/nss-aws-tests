import boto3
region = 'eu-central-1'
instances = ['i-03a49fe156346239a', 'i-0fb06f7d468a63de1']
custom_name = dict(zip(instances, ['Storage Gateway', 'p2.xlarge']))
ec2_client = boto3.client('ec2', region_name=region)
sqs_client = boto3.client('sqs', region_name=region)
url = 'https://sqs.eu-central-1.amazonaws.com/960602048864/uploading_cases'

from linebot import LineBotApi
from linebot.models import TextSendMessage
with open('token.txt') as f:
    channel_access_token = f.readline()
line_bot_api = LineBotApi(channel_access_token)
group_id = 'C8e0c9458a13b4dad203fcc224190a6f8'
# group_id = 'U41dda7349d30c51c503127901df2f27a' # push myself to debug

import urllib.parse
import time

def lambda_handler(event, context):
    try:
        # query = urllib.parse.parse_qs(event['queryStringParameters'])
        query = event['queryStringParameters'] # ALREADY a <class 'dict'> parsed by Chrome and Python-requests
        sqs_client.send_message(QueueUrl=url, MessageBody=str(query))
        site  = query['site']
        cases = query['cases']
    except:
        query = None
    else:
        print(query)

    if query:
        # line_bot_api.push_message(group_id, TextSendMessage(text=f'{site}上傳{cases}'))
        result = [f'{site}上傳{cases}']#[str(query)]
    else:
        result = []

    for instance in instances:
        for t in range(14):
            try:
                ec2_client.start_instances(InstanceIds=[instance])
                text = f'{custom_name[instance]} 開機'
                print(text)
                # line_bot_api.push_message(group_id, TextSendMessage(text=text))
                break
            except:
                text += f'失敗！60秒後重試第{t+1}次'
                print(text)
                # line_bot_api.push_message(group_id, TextSendMessage(text=text))
                time.sleep(60)
            finally:
                result.append(text)

    response = {}
    response['statusCode'] = 200
    response['headers'] = {}
    response['headers']['Content-Type'] = 'text/plain; charset=UTF-8'
    response['body'] = '\n'.join(result)
    line_bot_api.push_message(group_id, TextSendMessage(text=response['body']))
    return response