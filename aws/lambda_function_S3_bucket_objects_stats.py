import json
import boto3
import re

s3 = boto3.resource('s3')
bucket = s3.Bucket('neurobit-asg')

def lambda_handler(event, context):
    cases, months = [], []
    for obj in bucket.objects.all():
        m = re.search('(\d{8}_H\d{2}_NSS\d{5})/$', obj.key)
        if m and 'H14' not in m.group(1) and 'Reports' not in obj.key:
            cases.append(m.group(1))
    cases = sorted(cases)
    month = [cases[0]]
    for i in range(1, len(cases)):
        if cases[i][4:6] == cases[i-1][4:6]:
            month.append(cases[i])
        else:
            month.append(f'{cases[i-1][:6]} 月份共 {len(month)} 收案')
            months.append('\n'.join(month))
            month = [cases[i]]
    month.append(f'{cases[i][:6]} 至今共 {len(month)} 收案')
    months.append('\n'.join(month))

    return {
        'statusCode': 200,
        'body': '\n\n'.join(months)#json.dumps(cases)
    }
