import boto3
import json
import base64

client = boto3.client("bedrock-runtime", region_name="us-west-2")

def invoke_detailed_claude_model(prompt):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read().decode('utf-8'))
        print("Response Body:", response_body)

        if 'messages' in response_body:
            return response_body['messages'][0].get('content', [{}])[0].get('text', '')
        elif 'content' in response_body:
            return response_body['content'][0].get('text', '')
        else:
            raise KeyError("Unexpected response structure: 'messages' or 'content' field not found.")
    except Exception as e:
        print(f"Error invoking model: {str(e)}")
        return None

prompt = "Generate a detailed urban planning simulation considering green spaces, energy efficiency, and transportation for Jakarta."
simulation_result = invoke_detailed_claude_model(prompt)
print(simulation_result)

def invoke_claude_with_image(image_path, prompt):
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
        }

        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read().decode('utf-8'))
        print("Response Body:", response_body)

        if 'messages' in response_body:
            return response_body['messages'][0].get('content', [{}])[0].get('text', '')
        elif 'content' in response_body:
            return response_body['content'][0].get('text', '')
        else:
            raise KeyError("Unexpected response structure: 'messages' or 'content' field not found.")
    except FileNotFoundError:
        print(f"Error: File not found at path {image_path}")
        return None
    except Exception as e:
        print(f"Error invoking model with image: {str(e)}")
        return None

# Example usage with image
prompt = "Analyze this image for green space optimization."
image_path = "Images/kalimantan1.png"
simulation_result_with_image = invoke_claude_with_image(image_path, prompt)
print(simulation_result_with_image)
