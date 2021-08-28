import urllib.parse

import boto3
region = 'eu-central-1'
instances = ['i-03a49fe156346239a', 'i-0fb06f7d468a63de1']
ec2 = boto3.client('ec2', region_name=region)
EC2 = dict(zip(instances, ['Storage Gateway', 'p2.xlarge']))

from linebot import LineBotApi
from linebot.models import TextSendMessage
with open('token.txt') as f:
    channel_access_token = f.readline()
line_bot_api = LineBotApi(channel_access_token)
group_id = 'C8e0c9458a13b4dad203fcc224190a6f8'

import time

def lambda_handler(event, context):
    try:
        querys = urllib.parse.parse_qs(event['queryStringParameters'])
    except:
        print('No querys.')
    else:
        print(f'querys = {querys}')

    result = []
    for instance in instances:
        for t in range(10):
            try:
                text = f'{EC2[instance]} 關機'
                ec2.stop_instances(InstanceIds=[instance])
                print(text)
                # line_bot_api.push_message(group_id, TextSendMessage(text=text))
                break
            except:
                text += f'失敗\n60秒後重試第{t+1}次'
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
    return response