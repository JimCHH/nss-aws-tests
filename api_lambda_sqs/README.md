## 1 API Gateway
set maximum timeout 30 sec
1. hit and run a Lambda function to start or stop EC2 instances

## 2 Lambda Functions
set maximum timeout 15 min
1. StartEC2Instances
2. StopEC2Instances

## 1 Simple Queue Service
https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html
- Thus, the consumer must delete the message from the queue after receiving and processing it.
- The default visibility timeout for a message is 30 seconds. The minimum is 0 seconds. The maximum is 12 hours.
- The default retention period is 4 days.
