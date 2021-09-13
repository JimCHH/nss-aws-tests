import boto3
region = 'eu-central-1'
instances = ['i-03a49fe156346239a', 'i-0fb06f7d468a63de1']
custom_name = dict(zip(instances, ['Storage Gateway', 'p2.xlarge']))
ec2_client = boto3.client('ec2', region_name=region)
sqs_client = boto3.client('sqs', region_name=region)
url = 'https://sqs.eu-central-1.amazonaws.com/960602048864/uploading_cases'

with open('tokens') as f:
    channel_access_token, access_token = f.readline().split()
# LINE Messaging API
from linebot import LineBotApi
from linebot.models import TextSendMessage
line_bot_api = LineBotApi(channel_access_token)
group_id = 'C8e0c9458a13b4dad203fcc224190a6f8'
group_id = 'U41dda7349d30c51c503127901df2f27a' # uncomment to push to myself
# LINE Notify API
import requests
headers = {'Content-Type': 'application/x-www-form-urlencoded',
           'Authorization': f'Bearer {access_token}'}

# import urllib.parse
import time

def lambda_handler(event, context):
    try:
        # query = urllib.parse.parse_qs(event['queryStringParameters'])
        query = event['queryStringParameters'] # ALREADY a <class 'dict'> parsed by Chrome and Python-requests
        site  = query['site']
        cases = query['cases']
        sqs_client.send_message(QueueUrl=url, MessageBody=str(query))
    except:
        cases = None
    else:
        print(query)

    if cases:
        cases = cases.replace(',', '\n')
        # line_bot_api.push_message(group_id, TextSendMessage(text=f'{site}上傳{cases}'))
        # payload = {'message': f'{site}上傳{cases}'}
        # requests.post('https://notify-api.line.me/api/notify', headers=headers, data=payload)
        result = [f'{site} 上傳\n{cases}']#[str(query)]
    else:
        result = [f'{site} 登入 NSS']

    for instance in instances:
        for t in range(14):
            try:
                text = f'{custom_name[instance]} 關機'
                ec2_client.stop_instances(InstanceIds=[instance])
                print(text)
                # line_bot_api.push_message(group_id, TextSendMessage(text=text))
                # payload = {'message': text}
                # requests.post('https://notify-api.line.me/api/notify', headers=headers, data=payload)
                break
            except:
                text += f'失敗60秒後重試第{t+1}次'
                print(text)
                # line_bot_api.push_message(group_id, TextSendMessage(text=text))
                # payload = {'message': text}
                # requests.post('https://notify-api.line.me/api/notify', headers=headers, data=payload)
                time.sleep(60)
            finally:
                result.append(text)

    # line_bot_api.push_message(group_id, TextSendMessage(text='\n'.join(result)))
    payload = {'message': '\n'+'\n'.join(result)}
    requests.post('https://notify-api.line.me/api/notify', headers=headers, data=payload)
    response = {}
    response['statusCode'] = 200
    response['headers'] = {}
    response['headers']['Content-Type'] = 'text/plain; charset=UTF-8'
    response['body'] = '\n'.join(result)
    return response
