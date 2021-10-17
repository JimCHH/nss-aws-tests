import json
import boto3
import re

s3 = boto3.resource('s3')
bucket = s3.Bucket('neurobit-asg')

def lambda_handler(event, context):
    mp4s, Mp4s, months = [], {}, []
    for obj in bucket.objects.all():
        if 'H14' not in obj.key and 'Reports' not in obj.key:
            m = re.search('\d{8}_\d{6}_H\d{2}_NSS\d{5}_Test\d.mp4', obj.key)
            if m:
                mp4s.append(m.group(0))
            m = re.search('(\d{8}_H\d{2}_NSS\d{5})/$', obj.key)
            if m:
                Mp4s[m.group(1)] = []
    for mp4 in mp4s:
        Mp4s[mp4[:8]+mp4[15:28]].append(mp4)
    cases = sorted(Mp4s)
    month = ['\n'.join(Mp4s[cases[0]]) + '\n']
    for i in range(1, len(cases)):
        if cases[i][4:6] == cases[i-1][4:6]:
            month.append('\n'.join(Mp4s[cases[i]]) + '\n')
        else:
            month.append(f'{cases[i-1][:6]} 月份共 {len(month)} 收案')
            months.append('\n'.join(month))
            month = ['\n'.join(Mp4s[cases[i]]) + '\n']
    month.append(f'{cases[i][:6]} 至今共 {len(month)} 收案')
    months.append('\n'.join(month))

    return {
        'statusCode': 200,
        'body': ('\n'+'-'*40+'\n').join(months)#json.dumps('Hello from Lambda!')
    }
