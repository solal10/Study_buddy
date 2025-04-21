from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError
from pymongo import MongoClient
import os
import logging
import re

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

# Constants
WELCOME_MESSAGE = """Hi! 👋 I'm Study Buddy, your personal AI tutor.
Would you like me to build a program tailored to your level and guide you step by step?"""

BASE_PROMPT = """
You are Study Buddy, a friendly and patient AI tutor helping students learn Artificial Intelligence and Machine Learning.
You will guide them step by step by building a personalized learning program based on their level and preferences.

Ask them if they want to build a personalized program. If yes, ask a few quick questions to understand their current knowledge (beginner, intermediate, advanced), their goals, and preferred learning pace.

Then, begin with the first topic and explain it clearly with examples. At the end of each topic, check if they want to go deeper, practice, or continue.

Keep track of what they’ve learned so far and adapt the program as they progress. Be encouraging, supportive, and flexible.

If they ask questions at any time, answer helpfully and link back to the learning program when possible.
"""


# Initialize MongoDB client
mongodb_uri = os.getenv('MONGODB_URI')
if not mongodb_uri:
    raise ValueError('MONGODB_URI not found in environment variables')

mongo_client = MongoClient(mongodb_uri)
db = mongo_client.study_buddy
users_collection = db.users

# Topic detection patterns
TOPIC_PATTERNS = {
    'ai_basics': r'what is AI|artificial intelligence basics|AI fundamentals',
    'ml_basics': r'what is machine learning|ML basics|machine learning fundamentals',
    'supervised_learning': r'supervised learning|classification|regression',
    'unsupervised_learning': r'unsupervised learning|clustering|dimensionality reduction',
    'neural_networks': r'neural network|deep learning|backpropagation',
    'nlp': r'natural language processing|NLP|text processing',
    'computer_vision': r'computer vision|image processing|CV',
    'reinforcement_learning': r'reinforcement learning|RL|reward system'
}

def detect_topics(message):
    """Detect topics in the user's message using regex patterns."""
    detected = []
    message = message.lower()
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, message, re.IGNORECASE):
            detected.append(topic)
    return detected

def get_user_context(session_id):
    """Get or create user context from MongoDB"""
    if not session_id:
        return {'onboarded': False}
        
    user = users_collection.find_one({'session_id': session_id})
    if not user:
        user = {
            'session_id': session_id,
            'level': '',
            'preference': '',
            'interests': [],
            'onboarded': False
        }
        users_collection.insert_one(user)
    return user

def update_user_context(session_id, message, response):
    """Update user context based on conversation."""
    # Detect level if not set
    message_lower = message.lower()
    
    # Level detection with more variations
    if any(word in message_lower for word in ['beginner', 'basic', 'start', 'new']):
        users_collection.update_one(
            {'session_id': session_id},
            {'$set': {'level': 'beginner'}}
        )
    elif any(word in message_lower for word in ['intermediate', 'mid', 'middle', 'moderate']):
        users_collection.update_one(
            {'session_id': session_id},
            {'$set': {'level': 'intermediate'}}
        )
    elif any(word in message_lower for word in ['advanced', 'expert', 'experienced', 'proficient']):
        users_collection.update_one(
            {'session_id': session_id},
            {'$set': {'level': 'advanced'}}
        )
    
    # Detect and add new topics
    new_topics = detect_topics(message)
    if new_topics:
        users_collection.update_one(
            {'session_id': session_id},
            {'$addToSet': {'topics': {'$each': new_topics}}}
        )

def get_system_prompt(user_context):
    """Generate system prompt based on user context"""
    prompt = BASE_PROMPT

    if user_context.get('level'):
        prompt += f"\n\nThe user is a {user_context['level']} learner who prefers {user_context['preference']}."
        
    return prompt

# Error messages
ERROR_MESSAGES = {
    'missing_message': 'Please provide a message to continue our conversation.',
    'empty_message': 'I noticed your message was empty. Please ask me something about AI or machine learning!',
    'openai_error': 'Sorry, there was an error generating the response. Please try again.',
    'server_error': 'An unexpected error occurred. Please try again.',
    'missing_session': 'Session ID is required.',
    'missing_level': 'Level is required for onboarding.',
    'missing_preference': 'Learning preference is required for onboarding.',
    'missing_interests': 'At least one interest is required for onboarding.',
    'no_api_key': 'The AI assistant is not properly configured. Please make sure the OPENAI_API_KEY is set.',
    'general_error': 'I apologize, but I encountered an unexpected error. Please try again in a moment.'
}

def get_openai_client():
    """Initialize and return the OpenAI client with error handling."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment variables')
    return OpenAI(api_key=api_key)

@app.route('/welcome-message', methods=['GET'])
def welcome_message():
    try:
        return jsonify({'message': WELCOME_MESSAGE})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        message = data.get('message', '')

        # Get user context
        try:
            user_context = get_user_context(session_id)
        except Exception as e:
            return jsonify({'error': f'Failed to get user context: {str(e)}'}), 500
        
        if not user_context.get('onboarded'):
            return jsonify({'response': 'Please complete the onboarding form first.'})

        # Generate system prompt based on user context
        system_prompt = get_system_prompt(user_context)
        
        # Create chat completion
        try:
            client = get_openai_client()
            chat_completion = client.chat.completions.create(
                model="gpt-4",  # Using GPT-4
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,  # Balanced between creativity and consistency
                max_tokens=1000   # Reasonable length for educational responses
            )
        
        # Get assistant's response
        except APIError as e:
            return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500
        
        assistant_response = chat_completion.choices[0].message.content
        
        # Update user context with new topics
        update_user_context(session_id, message, assistant_response)
        
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

@app.route('/onboarding', methods=['POST'])
def onboarding():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Validate required fields
        required_fields = ['level', 'preference']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Update user context in MongoDB
        users_collection.update_one(
            {'session_id': session_id},
            {
                '$set': {
                    'level': data.get('level'),
                    'preference': data.get('preference'),
                    'onboarded': True
                }
            },
            upsert=True
        )
        
        return jsonify({
            'success': True,
            'message': 'Onboarding completed successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Verify OpenAI API key is available
        get_openai_client()
        app.run(port=5001, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        exit(1)
