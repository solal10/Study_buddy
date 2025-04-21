from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# System prompt for the AI
SYSTEM_PROMPT = """You are Study Buddy, a personal AI tutor helping students learn Artificial Intelligence and Machine Learning.

Start by asking the user what their current knowledge level is (beginner, intermediate, or advanced), and whether they've studied any AI/ML topics before.

Once they answer, build a personalized learning path based on their response.

Be warm, engaging, and clear — like a real tutor. Use analogies or examples when needed. Offer quizzes or checkpoints after each topic if the user wants.

You must act like a proactive guide, not a passive assistant. Always move the learning forward."""

# Initial greeting message
INITIAL_GREETING = """Hi! 👋 I'm Study Buddy, your personal AI tutor.  
Before we begin, could you tell me how familiar you are with AI or machine learning?"""

# Track conversation state
user_conversations = {}

# Error messages
ERROR_MESSAGES = {
    'missing_message': 'Please provide a message to continue our conversation.',
    'empty_message': 'I noticed your message was empty. Please ask me something about AI or machine learning!',
    'api_error': 'I encountered a temporary issue. Please try asking your question again.',
    'no_api_key': 'The AI assistant is not properly configured. Please make sure the OPENAI_API_KEY is set.',
    'general_error': 'I apologize, but I encountered an unexpected error. Please try again in a moment.'
}

def get_openai_client():
    """Initialize and return the OpenAI client with error handling."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment variables')
    return OpenAI(api_key=api_key)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Initialize OpenAI client
        client = get_openai_client()

        # Validate request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'response': ERROR_MESSAGES['missing_message']}), 400
        
        message = data['message'].strip()
        
        # Get or initialize conversation history
        conversation_id = request.headers.get('X-Conversation-Id', 'default')
        
        # For new conversations, return the greeting
        if conversation_id not in user_conversations:
            user_conversations[conversation_id] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": INITIAL_GREETING}
            ]
            return jsonify({"response": INITIAL_GREETING})
            
        # For existing conversations, require a non-empty message
        if not message:
            return jsonify({'response': ERROR_MESSAGES['empty_message']}), 400
        
        # Add user message to conversation history
        user_conversations[conversation_id].append({"role": "user", "content": message})
        
        # Add user message to conversation history
        user_conversations[conversation_id].append({"role": "user", "content": message})
        
        # Create chat completion with full conversation history
        chat_completion = client.chat.completions.create(
            model="gpt-4.1",  # Using GPT-4.1 as requested
            messages=user_conversations[conversation_id],
            temperature=0.7,  # Balanced between creativity and consistency
            max_tokens=1000   # Reasonable length for educational responses
        )
        
        # Add assistant's response to conversation history
        assistant_response = chat_completion.choices[0].message.content
        user_conversations[conversation_id].append({"role": "assistant", "content": assistant_response})
        
        # Limit conversation history to last 10 messages to prevent token limit issues
        if len(user_conversations[conversation_id]) > 12:  # system prompt + 10 messages
            user_conversations[conversation_id] = [
                user_conversations[conversation_id][0],  # Keep system prompt
                *user_conversations[conversation_id][-10:]  # Keep last 10 messages
            ]
        
        # Extract and return the response
        ai_response = chat_completion.choices[0].message.content
        return jsonify({"response": ai_response})

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return jsonify({"response": ERROR_MESSAGES['no_api_key']}), 500

    except APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({"response": ERROR_MESSAGES['api_error']}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"response": ERROR_MESSAGES['general_error']}), 500

if __name__ == '__main__':
    try:
        # Verify OpenAI API key is available
        get_openai_client()
        app.run(debug=True, port=5001)
    except ValueError as e:
        logger.error(str(e))
        exit(1)
