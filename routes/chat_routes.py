import os
from flask import Blueprint, request, jsonify
from services.lex_service import send_to_lex_bot

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('text', '')
    bot_response = send_to_lex_bot(
        bot_id=os.getenv('bot_id'),
        bot_alias_id=os.getenv('bot_alias'),
        locale_id='en_US',
        user_id='greenerypulsebot',
        text=user_message
    )
    return jsonify(response=bot_response)
