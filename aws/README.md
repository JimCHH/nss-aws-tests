## 1 API Gateway
1. NSS backend API
> Timeout must be a number from 50 to 30000 milliseconds.

## 3 Lambda Functions
1. StartEC2Instances
2. StopEC2Instances
3. S3_bucket_objects_stats
> The maximum timeout is 15 minutes.

## 1 Simple Queue Service
1. uploaded_cases
> The default retention period is 4 days.

> [Thus, the consumer must delete the message from the queue after receiving and processing it.](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
> 
> [The default visibility timeout for a message is 30 seconds. The minimum is 0 seconds. The maximum is 12 hours.](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
