import librosa
import soundfile as sf
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import json

def split_audio(input_file, output_path, output_basename, min_silence_duration=0.9):
    # Load audio file with Librosa
    audio, sr = librosa.load(input_file, sr=None)

    intervals = librosa.effects.split(audio, top_db=5)

    # Create the output directory if not exists
    os.makedirs(output_path, exist_ok=True)

    # Save each segment as a separate file
    for i, interval in enumerate(intervals):
        start, end = interval
        
        duration = end - start
        if duration >= min_silence_duration:
            segment = audio[start:end]
            output_file = f"{output_path}/{output_basename}{i+1}.wav"

            sf.write(output_file, segment, sr)


def preprocess_audio(audio_path, sr=16000, n_mfcc=40):
    # Load the audio file
    signal, sr = librosa.load(audio_path, sr=sr)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    # Transpose the matrix to have MFCCs as columns
    mfccs = mfccs.T
    return mfccs

input_file = 'data/nt.wav'
output_path = './tempaud/'
output_basename = 'aud'

split_audio(input_file, output_path, output_basename, min_silence_duration=0.5)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load word index from json file
with open('word_index.json', 'r') as f:
    word_index = json.load(f)

# Tokenize the labels and convert to binary matrix
tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_index.keys())

# Define the output file path
output_file = 'predicted_text.txt'

# Initialize a list to store predicted labels
predicted_labels = []

for i in range(len(os.listdir(output_path))):
    audio_file = f"{output_path}/{output_basename}{i+1}.wav"
    try:
        new_mfccs = preprocess_audio(audio_file)
    except FileNotFoundError:
        continue

    
    new_mfccs = tf.ragged.constant([new_mfccs])
    prediction = model.predict(new_mfccs)
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_label_text = tokenizer.sequences_to_texts([[predicted_label]])[0]
    predicted_labels.append(predicted_label_text)


# Save the predicted labels to a text file
with open(output_file, 'w') as f:
    f.write(' '.join(predicted_labels))

print('Done! Predicted labels are saved to', output_file)
