# சாரம் (SAARAM) - Tamil News Summarizer

AI-powered Tamil news summarization using the IndicBART model from AI4Bharat.

🔗 **GitHub Repository**: https://github.com/Ashy-it24/SAARAM

## Features

- **Tamil Text Summarization**: Uses the `ai4bharat/indicbart-ss` model, which is specifically trained for same-script summarization of Indian languages.
- **Real-time Processing**: Fast API backend with optimized model loading.
- **Modern UI**: Clean Next.js frontend with loading states and error handling.
- **Compression Stats**: Shows original vs summary length and compression ratio.
- **Translation**: Translate Tamil text to English using the MyMemory API with toggle option.
- **Text-to-Speech**: Convert Tamil text to speech using gTTS.
- **Question Answering**: Ask questions about the article.
- **MyMemory API Toggle**: Switch between high-quality online translation and basic dictionary translation.

## Tech Stack

- **Frontend**: Next.js, React, Axios
- **Backend**: FastAPI, PyTorch, Transformers (Hugging Face)
- **AI Model**: `ai4bharat/indicbart-ss`

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Git

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

### Quick Start (Windows)

Run the `start.bat` file to start both servers automatically.

## Usage

1. Open http://localhost:3000 in your browser.
2. Paste Tamil news text in the textarea.
3. Click "Summarize" to get an AI-generated summary.
4. View compression statistics.
5. Translate the summary to English.
6. Listen to the Tamil summary.
7. Ask questions about the article.

## API Endpoints

- `GET /`: Health check
- `POST /summarize`: Summarize Tamil text
  - Body: `{"text": "Tamil text here", "max_length": 150, "min_length": 40}`
- `POST /translate`: Translate Tamil text to English
- `POST /tts`: Convert Tamil text to speech
- `POST /chat`: Ask a question about the article

## Model Information

The app uses **`ai4bharat/indicbart-ss`**, a state-of-the-art model for same-script summarization of Indian languages. This model is part of the IndicBART family of models from AI4Bharat, which are designed to be lightweight and efficient for Indian languages.

## Development

- Backend runs on: http://127.0.0.1:8000
- Frontend runs on: http://localhost:3000
- API docs available at: http://127.0.0.1:8000/docs
