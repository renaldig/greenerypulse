import os
import json
import boto3

session = boto3.Session(
    aws_access_key_id=os.getenv('aws_access_key_id_3'),
    aws_secret_access_key=os.getenv('aws_secret_access_key_3'),
    region_name='us-west-2'
)

bedrock_client = session.client('bedrock-runtime', region_name='us-west-2')

def invoke_claude_model(prompt, image_data=None):
    payload = {
        "max_tokens": 1024,
        "system": "You are an urban planning assistant...",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "anthropic_version": "bedrock-2023-05-31"
    }

    if image_data:
        payload["messages"][0]["content"].append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data
            }
        })

    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json"
        )
        raw_response = response['body'].read().decode('utf-8')
        response_body = json.loads(raw_response)
        if 'content' in response_body:
            return response_body['content'][0].get('text', '')
    except Exception as e:
        print(f"Error invoking Claude model: {e}")
        return "Error invoking model."
