import boto3
import os

session = boto3.Session(
    aws_access_key_id=os.getenv('aws_access_key_id_3'),
    aws_secret_access_key=os.getenv('aws_secret_access_key_3'),
    region_name='us-west-2'
)

rekognition_client = session.client('rekognition', region_name='us-west-2')
s3_client = session.client('s3', region_name='us-west-2')

def analyze_images_from_s3_folder(bucket, folder):
    response = s3_client.list_objects_v2(Bucket='greenerypulseplanning', Prefix='Images')
    labels_per_image = {}
    
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            
            if key.endswith(('.png', '.jpg', '.jpeg')):
                rekognition_response = rekognition_client.detect_labels(
                    Image={'S3Object': {'Bucket': bucket, 'Name': key}},
                    MaxLabels=10
                )
                labels = rekognition_response['Labels']
                labels_per_image[key] = labels
                print(f"Image {key} labels:", labels)
            else:
                print(f"Skipped non-image file {key}")
    
    return labels_per_image
'''
# Example usage
bucket = 'greenerypulseplanning'
folder = 'Images/'  # Folder containing images
labels = analyze_images_from_s3_folder(bucket, folder)
print("Image Analysis Labels:", labels)
'''
