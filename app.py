from flask import Flask, request, jsonify, send_file, send_from_directory
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from gtts import gTTS
import pygame
import time
import whisper
import os
import warnings
import subprocess
from collections import deque
from groq import Groq
import logging

app = Flask(__name__)

# Set up logging to write to a file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

groq_key = "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"
client = Groq(api_key=groq_key)

# Initialize the pygame mixer
pygame.mixer.init()

# Initialize a global context variable to store the conversation context
context = deque(maxlen=5)

def record_audio(filename, duration=5, fs=16000):
    """Records audio and saves it to a file."""
    start_time = time.time()
    try:
        logger.info("Recording audio...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        wavfile.write(filename, fs, audio)
        logger.info("Recording complete")
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
    end_time = time.time()
    logger.info(f"Time taken to record audio: {end_time - start_time:.2f} seconds")

def transcribe_audio(filename):
    """Transcribes audio to text using Whisper."""
    start_time = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = whisper.load_model("tiny")
            result = model.transcribe(filename)
        transcription = result['text']
        logger.info("Transcription complete")
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        transcription = ""
    end_time = time.time()
    logger.info(f"Time taken to transcribe audio: {end_time - start_time:.2f} seconds")
    return transcription

def llm_response(context, text):
    """Generates a response from the language model using context."""
    start_time = time.time()
    messages = [{"role": "system", "content": """You are a helpful and kind conversational assistant.
                                                Use short, conversational responses as if you're having a live conversation.
                                                Your response should be under 40 words or less.
                                                Do not respond with any code, only conversation.
                                                Always answer the user's question"""}]
    messages += [{"role": "user", "content": msg} for msg in context]
    messages.append({"role": "user", "content": text})

    try:
        response = client.chat.completions.create(model="llama3-70b-8192", messages=messages)
        llm_response_text = response.choices[0].message.content
        logger.info("LLM response generation complete")
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        llm_response_text = "I'm sorry, I couldn't process that."
    end_time = time.time()
    logger.info(f"Time taken to get LLM response: {end_time - start_time:.2f} seconds")
    return llm_response_text

def text_to_speech(text, output_file="output_audio.mp3"):
    """Converts text to speech and saves it to a file."""
    start_time = time.time()
    try:
        logger.info("Converting text to speech...")
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        logger.info(f"Text to Speech saved as: {output_file}")
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
    end_time = time.time()
    logger.info(f"Time taken to convert text to speech: {end_time - start_time:.2f} seconds")
    return output_file

def play_audio(filename):
    """Plays audio from a file."""
    start_time = time.time()
    try:
        logger.info(f"Playing audio: {filename}")
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        logger.info("Finished playing audio.")
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
    end_time = time.time()
    logger.info(f"Time taken to play audio: {end_time - start_time:.2f} seconds")

# def synthesize_text(text, model, output_file):
def synthesize_text(text, output_file):
    """Synthesizes text using an external model."""
    start_time = time.time()
    try:

        result = subprocess.run(['edge-tts', '--text', text, '--write-media', output_file],
                                input=text, text=True, capture_output=True, check=True)
        # result = subprocess.run(['piper', '--model', model, '--output_file', output_file],
        #                         input=text, text=True, capture_output=True, check=True)
        stdout, stderr = result.stdout, result.stderr
        # print(stdout,stderr)
        logger.info("Text synthesis complete")
    except subprocess.CalledProcessError as e:
        stdout, stderr = e.stdout, e.stderr
        logger.error(f"Error synthesizing text: {stderr}")
    end_time = time.time()
    logger.info(f"Time taken to synthesize text: {end_time - start_time:.2f} seconds")
    return stdout, stderr

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_file.save('input.wav')  # Save the audio file

    global context  # Use the global context variable

    logger.info("Starting transcription...")
    spoken_text = transcribe_audio('input.wav')
    logger.info("Starting LLM response generation...")
    response_from_llm = llm_response(list(context), spoken_text)

    # Add the current user input and AI response to the context
    context.append(spoken_text)
    context.append(response_from_llm)

    logger.info("Starting text-to-speech conversion...")
    # model = "en_US-lessac-medium"
    output_file = "response.wav"
    # synthesize_text(response_from_llm, model, output_file)
    synthesize_text(response_from_llm,output_file)

    return jsonify({
        "request_text": spoken_text,
        "response_text": response_from_llm,
        "audio_url": f"/audio/{output_file}"
    })

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
