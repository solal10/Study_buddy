# AI Study Buddy

A full-stack AI chatbot application built with React and Flask.

## Project Structure
- `client/`: React frontend with TailwindCSS
- `server/`: Flask backend API
- `run.sh`: Development startup script

## Setup Instructions

1. Install frontend dependencies:
```bash
cd client
npm install
```

2. Install backend dependencies:
```bash
cd server
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure your environment variables

4. Start the development servers:
```bash
./run.sh
```

## Development
- Frontend runs on: http://localhost:5173
- Backend API runs on: http://localhost:5000
