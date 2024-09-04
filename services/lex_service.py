import os
from services.dynamodb_service import insert_feedback_to_dynamodb
import boto3

session = boto3.Session(
    aws_access_key_id=os.getenv('aws_access_key_id_3'),
    aws_secret_access_key=os.getenv('aws_secret_access_key_3'),
    region_name='us-west-2'
)

lex_client = session.client('lexv2-runtime', region_name='us-west-2')

def send_to_lex_bot(bot_id, bot_alias_id, locale_id, user_id, text):
    response = lex_client.recognize_text(
        botId=bot_id,
        botAliasId=bot_alias_id,
        localeId=locale_id,
        sessionId=user_id,
        text=text
    )
    
    if 'interpretations' in response and response['interpretations']:
        for interpretation in response['interpretations']:
            if 'intent' in interpretation and interpretation['intent']['name'] == 'ProvideFeedback':
                intent_state = interpretation['intent']['state']
                slots = interpretation['intent'].get('slots', {})
                feedback_slot = slots.get('FeedbackText')

                if intent_state == 'ReadyForFulfillment' and feedback_slot and feedback_slot.get('value'):
                    feedback_text = feedback_slot['value'].get('interpretedValue')
                    if feedback_text:
                        insert_feedback_to_dynamodb(feedback_text)
                        return "Thank you for your feedback!"
                elif feedback_slot is None or feedback_slot.get('value') is None:
                    return "Please provide your feedback."
                else:
                    return "Processing your feedback, please wait."
            elif 'intent' in interpretation and interpretation['intent']['name'] == 'FallbackIntent':
                return "I'm having trouble understanding. Can you please rephrase?"

    if 'messages' in response and response['messages']:
        return response['messages'][0]['content']
    else:
        return "No valid response from the bot."
