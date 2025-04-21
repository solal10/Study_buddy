#!/bin/bash

# Start the Flask backend
cd server
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt
python app.py &

# Start the React frontend
cd ../client
npm install
npm run dev &

# Wait for any process to exit
wait
