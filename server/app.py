from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError
from pymongo import MongoClient
from bson import ObjectId
import os
import logging
import re
from datetime import datetime
from auth import hash_password, check_password, generate_token, token_required

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
try:
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError('MONGODB_URI environment variable not set')
    
    mongo_client = MongoClient(mongodb_uri)
    mongo_db = mongo_client.study_buddy  # Explicitly specify the database name
    # Test the connection
    mongo_client.server_info()
    logger.info('Successfully connected to MongoDB')
except Exception as e:
    logger.error(f'Failed to connect to MongoDB: {e}')
    raise

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})

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
    # 📚 Foundations
    'What is AI': r'\b(what is AI|artificial intelligence basics|AI fundamentals|introduction to AI)\b',
    'AI Applications': r'\b(applications of AI|real-world AI|AI use cases|AI in healthcare|AI in finance|AI deployment)\b',
    'Ethics in AI': r'\b(AI ethics|bias|fairness|responsible AI|AI safety|ethical considerations)\b',

    # 🧠 Machine Learning Core
    'Machine Learning Basics': r'\b(what is machine learning|ML basics|machine learning fundamentals|types of ML)\b',
    'Supervised Learning': r'\b(supervised learning|classification|regression|training data|labeled data)\b',
    'Unsupervised Learning': r'\b(unsupervised learning|clustering|unlabeled data)\b',
    'Reinforcement Learning': r'\b(reinforcement learning|RL|reward system|agent|environment|policy)\b',
    'Overfitting and Underfitting': r'\b(overfitting|underfitting|bias[- ]variance tradeoff|generalization)\b',
    'Hyperparameter Tuning': r'\b(hyperparameter tuning|grid search|random search|parameter optimization|model tuning)\b',

    # 🧮 Algorithms & Techniques
    'Neural Networks': r'\b(neural network|deep learning|backpropagation|layers|neurons|activation functions)\b',
    'Decision Trees & Random Forests': r'\b(decision tree|random forest|tree[- ]based model|gini impurity|entropy split)\b',
    'K-Nearest Neighbors': r'\b(k[- ]nearest neighbors|knn|instance[- ]based learning|similarity[- ]based learning)\b',
    'Support Vector Machines': r'\b(support vector machine|svm|maximum margin classifier|kernel trick)\b',

    # 🛠 Workflow & Utilities
    'Data Preprocessing': r'\b(data preprocessing|cleaning|feature engineering|normalization|data transformation)\b',
    'Feature Selection': r'\b(feature selection|selecting features|redundant features|dimensionality reduction techniques)\b',
    'Dimensionality Reduction': r'\b(dimensionality reduction|PCA|principal component analysis|t[- ]?SNE|feature compression)\b',
    'Model Evaluation': r'\b(model evaluation|evaluation metrics|model metrics|accuracy|precision|recall|confusion matrix|cross[- ]?validation)\b',

    # 🧪 Specializations
    'Natural Language Processing': r'\b(natural language processing|NLP|text processing|tokenization|text analysis)\b',
    'Computer Vision': r'\b(computer vision|image processing|CV|image recognition|object detection)\b',

    # 🔄 Fallback
    'Other': r'.*'
}

def extract_topic_from_response(response):
    """Extract the main topic from the assistant's response."""
    # First try explicit topic indicators
    topic_indicators = [
        r"Let's talk about ([^.]+)",
        r"In this lesson, we'll cover ([^.]+)",
        r"Today we'll learn about ([^.]+)",
        r"Let's explore ([^.]+)",
        r"Understanding ([^:]+):",
        r"Now we'll discuss ([^.]+)",
        r"Let's dive into ([^.]+)",
        r"Moving on to ([^,]+),"
    ]
    
    # Try to find explicit topic mentions first
    for pattern in topic_indicators:
        match = re.search(pattern, response)
        if match:
            topic = match.group(1).strip()
            # Find the closest matching predefined topic
            for defined_topic in TOPIC_PATTERNS.keys():
                if defined_topic != 'Other' and (topic.lower() in defined_topic.lower() or defined_topic.lower() in topic.lower()):
                    return defined_topic
    
    # If no explicit topic found, try pattern matching
    for topic, pattern in TOPIC_PATTERNS.items():
        if topic != 'Other' and re.search(pattern, response, re.IGNORECASE):
            return topic
    
    # If still no match, return Unknown
    return "Unknown"

def detect_topics_from_text(text):
    """Detect topics in text using regex patterns."""
    text = text.lower()
    for topic, pattern in TOPIC_PATTERNS.items():
        if topic != 'Other' and re.search(pattern, text, re.IGNORECASE):
            return topic
    return "Unknown"

def get_user_context(session_id):
    """Get or create user context from MongoDB"""
    if not session_id:
        return {'onboarded': False}
    
    # Use upsert to either find or create the user
    user = mongo_db.users.find_one_and_update(
        {'session_id': session_id},
        {'$setOnInsert': {
            'session_id': session_id,
            'level': '',
            'preference': '',
            'interests': [],
            'onboarded': False
        }},
        upsert=True,
        return_document=True
    )
    return user

def update_user_context(session_id, message, response):
    """Update user context based on conversation."""
    # Detect level if not set
    message_lower = message.lower()
    
    # Level detection with more variations
    if any(word in message_lower for word in ['beginner', 'basic', 'start', 'new']):
        mongo_db.users.update_one(
            {'session_id': session_id},
            {'$set': {'level': 'beginner'}}
        )
    elif any(word in message_lower for word in ['intermediate', 'mid', 'middle', 'moderate']):
        mongo_db.users.update_one(
            {'session_id': session_id},
            {'$set': {'level': 'intermediate'}}
        )
    elif any(word in message_lower for word in ['advanced', 'expert', 'experienced', 'proficient']):
        mongo_db.users.update_one(
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
    
    # Add covered topics to the prompt
    if user_context.get('topics'):
        covered_topics = ', '.join(user_context['topics'])
        prompt += f"\n\nThe user has already learned about: {covered_topics}. "
        prompt += "Avoid repeating those topics unless explicitly asked. "
        prompt += "Continue progressing based on their level and preference."
    
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

@app.route('/welcome-message')
@token_required
def welcome_message():
    try:
        user_id = request.user_id
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({'message': WELCOME_MESSAGE})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
@token_required
def ask():
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        message = data.get('message', '')
        user_id = request.user_id

        # Get user context
        try:
            user = users_collection.find_one({'_id': ObjectId(user_id)})
            if not user:
                return jsonify({'error': 'User not found'}), 404
            user_context = user.get('context', {})
        except Exception as e:
            return jsonify({'error': f'Failed to get user context: {str(e)}'}), 500
        
        if not user.get('onboarded'):
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
        
        # Extract and update topics from assistant's response
        try:
            # Extract main topic from response
            current_topic = extract_topic_from_response(assistant_response)
            
            if current_topic and current_topic != 'Unknown':
                # Get current user data to check if topic exists
                existing_topics = user.get('topics', [])
                
                if current_topic not in existing_topics:
                    # Update MongoDB with new topic and timestamp
                    current_time = datetime.utcnow().isoformat() + 'Z'
                    users_collection.update_one(
                        {'_id': ObjectId(user_id)},
                        {
                            '$addToSet': {'topics': current_topic},
                            '$set': {f'topics_timestamps.{current_topic}': current_time}
                        }
                    )
                    print(f'Added new topic: {current_topic} at {current_time}')
        except Exception as e:
            print(f'Error updating topics: {e}')

        return jsonify({
            'response': assistant_response,
            'detected_topic': current_topic if current_topic != 'Unknown' else None
        })

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
@token_required
def onboarding():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        user_id = request.user_id
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Update user context with onboarding data
        knowledge_level = data.get('knowledge_level')
        goals = data.get('goals')
        pace = data.get('pace')
        interests = data.get('interests', [])

        if not all([knowledge_level, goals, pace]):
            return jsonify({'error': 'Missing required onboarding information'}), 400

        update_data = {
            'onboarded': True,
            'context': {
                'knowledge_level': knowledge_level,
                'goals': goals,
                'pace': pace,
                'interests': interests
            }
        }

        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )

        return jsonify({'message': 'Onboarding completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/topics-status', methods=['GET'])
@token_required
def get_topics_status():
    """Get the completion status of all topics for a user."""
    try:
        user_id = request.user_id
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Return completed topics and their timestamps
        return jsonify({
            'completed_topics': user.get('topics', []),
            'topics_timestamps': user.get('topics_timestamps', {})
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mark-topic', methods=['PATCH'])
@token_required
def mark_topic():
    """Mark a topic as completed or uncompleted."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        topic = data.get('topic')
        completed = data.get('completed', True)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400

        user_id = request.user_id
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        if completed:
            # Mark topic as completed with timestamp
            current_time = datetime.utcnow().isoformat() + 'Z'
            users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {
                    '$addToSet': {'topics': topic},
                    '$set': {f'topics_timestamps.{topic}': current_time}
                }
            )
        else:
            # Remove topic from completed list
            users_collection.update_one(
                {'session_id': session_id},
                {
                    '$pull': {'topics': topic},
                    '$unset': {f'topics_timestamps.{topic}': ''}
                }
            )

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit-quiz', methods=['POST'])
@token_required
def submit_quiz():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['session_id', 'topic', 'score', 'total', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate required fields
        required_fields = ['topic', 'score', 'total', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        user_id = request.user_id
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Prepare quiz result
        quiz_result = {
            'topic': data['topic'],
            'score': data['score'],
            'total': data['total'],
            'timestamp': data['timestamp']
        }
        
        # Update or insert quiz result
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {
                '$pull': {'quiz_history': {'topic': data['topic']}}
            }
        )
        
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {
                '$push': {'quiz_history': quiz_result}
            }
        )
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f'Error submitting quiz result: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Check if user already exists
        existing_user = users_collection.find_one({'email': data['email']})
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 409

        # Hash password and create user
        hashed_password = hash_password(data['password'])
        user = {
            'email': data['email'],
            'password': hashed_password,
            'level': '',
            'preference': '',
            'interests': [],
            'onboarded': False,
            'created_at': datetime.utcnow()
        }

        result = users_collection.insert_one(user)
        token = generate_token(str(result.inserted_id))

        return jsonify({
            'token': token,
            'user': {
                'id': str(result.inserted_id),
                'email': user['email'],
                'onboarded': user['onboarded']
            }
        })

    except Exception as e:
        logger.error(f'Error registering user: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Find user by email
        user = mongo_db.users.find_one({'email': data['email']})
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        # Verify password
        if not check_password(data['password'], user['password']):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Generate token
        token = generate_token(str(user['_id']))

        return jsonify({
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'onboarded': user['onboarded']
            }
        })

    except Exception as e:
        logger.error(f'Error logging in: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/me', methods=['GET'])
@token_required
def get_current_user():
    try:
        user = mongo_db.users.find_one({'_id': ObjectId(request.user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({
            'id': str(user['_id']),
            'email': user['email'],
            'level': user['level'],
            'preference': user['preference'],
            'interests': user['interests'],
            'onboarded': user['onboarded']
        })

    except Exception as e:
        logger.error(f'Error getting user: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Verify OpenAI API key is available
        get_openai_client()
        app.run(port=5001, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        exit(1)
