import numpy as np
import scipy.io.wavfile as wavfile
from gtts import gTTS
import pygame
import time
import os
import warnings
import sounddevice as sd
from collections import deque
from faster_whisper import WhisperModel
import logging
from pydub import AudioSegment
from groq import Groq

groq_key = "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"
client = Groq(api_key=groq_key)

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

# Initialize a global context variable to store the conversation context
interaction_history = deque(maxlen=5)

# Initialize the Whisper model
model_size = "distil-large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

def transcribe_audio(filename):
    """Transcribes audio to text using faster-whisper."""
    start_time = time.time()
    transcription = ""
    try:
        # Transcribe audio using faster-whisper
        segments, info = model.transcribe(filename, beam_size=5, language="en", condition_on_previous_text=False)
        transcription = " ".join(segment.text for segment in segments)
        logger.info("Transcription complete")
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
    end_time = time.time()
    logger.info(f"Time taken to transcribe audio: {end_time - start_time:.2f} seconds")
    return transcription, end_time - start_time

def text_to_speech(text, output_file="response.mp3"):
    """Converts text to speech and saves it to a file."""
    start_time = time.time()
    if not text:
        logger.error("No text to speak")
        return None, 0.0
    try:
        logger.info("Converting text to speech...")
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        logger.info(f"Text to Speech saved as: {output_file}")
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
    end_time = time.time()
    logger.info(f"Time taken to convert text to speech: {end_time - start_time:.2f} seconds")
    return output_file, end_time - start_time

def play_audio(filename):
    """Plays audio from a file."""
    start_time = time.time()
    try:
        logger.info(f"Playing audio: {filename}")
        pygame.mixer.init()
        audio = AudioSegment.from_file(filename)
        wav_filename = "output_audio.wav"
        audio.export(wav_filename, format="wav")

        pygame.mixer.music.load(wav_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        logger.info("Finished playing audio.")
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
    end_time = time.time()
    logger.info(f"Time taken to play audio: {end_time - start_time:.2f} seconds")
    return end_time - start_time

def llm_response(text):
    """Generates a response from the language model using context."""
    print(f"Sending to LLM: {text}")

    # Construct the message history
    messages = [
        {
            "role": "system",
            "content": """You are a helpful and kind conversational assistant.
                        Use short, conversational responses as if you're having a live conversation.
                        Your response should be under 40 words or less.
                        Do not respond with any code, only conversation"""
        }
    ]

    for interaction in interaction_history:
        messages.append({"role": "user", "content": interaction["request"]})
        messages.append({"role": "assistant", "content": interaction["response"]})

    messages.append({"role": "user", "content": text})

    llm_response_1 = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )

    response_text = llm_response_1.choices[0].message.content
    print(f"Received from LLM: {response_text}")

    # Update interaction history
    interaction_history.append({"request": text, "response": response_text})

    return response_text

def main():
    # Manually upload audio file
    uploaded_file = input("Enter the path to the audio file: ")

    # Check if the file exists
    if not os.path.isfile(uploaded_file):
        logger.error("File does not exist.")
        return

    # Start time
    start_time = time.time()

    # Transcribe uploaded audio
    transcribed_text, transcribe_time = transcribe_audio(uploaded_file)
    print(f"Transcribed Text: {transcribed_text}")
    print(f"Transcribe Time: {transcribe_time:.2f} seconds")

    # Get response from LLM
    llm_text_response = llm_response(transcribed_text)
    print(f"LLM Response: {llm_text_response}")

    # Convert LLM response to speech
    output_audio_file, tts_time = text_to_speech(llm_text_response, "response.mp3")
    print(f"Text-to-Speech Time: {tts_time:.2f} seconds")

    # Play the generated audio
    play_time = play_audio(output_audio_file)
    print(f"Play Audio Time: {play_time:.2f} seconds")

    # End time
    end_time = time.time()

    # Total time
    total_time = end_time - start_time
    print(f"Total Time: {total_time:.2f} seconds")

    logger.info(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
