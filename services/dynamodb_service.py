import os
import boto3
import random
import time

session = boto3.Session(
    aws_access_key_id=os.getenv('aws_access_key_id_3'),
    aws_secret_access_key=os.getenv('aws_secret_access_key_3'),
    region_name='us-west-2'
)

dynamodb_client = session.client('dynamodb', region_name='us-west-2')

def insert_feedback_to_dynamodb(feedback_text):
    feedback_id = random.randint(1, 1000000)
    timestamp = int(time.time())

    try:
        response = dynamodb_client.put_item(
            TableName='GreeneryPulseFeedbackTable',
            Item={
                'id': {'N': str(feedback_id)},
                'Timestamp': {'N': str(timestamp)},
                'FeedbackText': {'S': feedback_text}
            }
        )
        return response
    except Exception as e:
        print(f"Error inserting feedback into DynamoDB: {e}")
