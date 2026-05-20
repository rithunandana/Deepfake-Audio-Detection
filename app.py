import os
import joblib
import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow all domains to access your API

# Set maximum file upload size (16 MB limit)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit for file uploads

# Load model and scaler
model_filename = "svm_model.pkl"
scaler_filename = "scaler.pkl"
svm_classifier = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Ensure the 'uploads' directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Function to extract MFCC features from an audio file
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

# Function to detect unnatural pauses in audio
def detect_unnatural_pauses(audio_path, silence_threshold=0.5, pause_threshold=1.0):
    audio_data, sr = librosa.load(audio_path, sr=None)
    non_silent_intervals = librosa.effects.split(audio_data, top_db=silence_threshold)

    pause_durations = []
    for i in range(1, len(non_silent_intervals)):
        start_time = non_silent_intervals[i-1][1] / sr
        end_time = non_silent_intervals[i][0] / sr
        pause_duration = end_time - start_time
        if pause_duration > pause_threshold:
            pause_durations.append(pause_duration)

    return pause_durations

# Function to calculate jitter and shimmer
def calculate_jitter_and_shimmer(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.stft(y)
    pitches, magnitudes = librosa.core.piptrack(S=D, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    pitch_diffs = np.diff(pitch_values)
    jitter = np.mean(np.abs(pitch_diffs)) / np.mean(pitch_values) if len(pitch_values) > 1 else 0

    amplitude_envelope = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    shimmer_diffs = np.diff(amplitude_envelope)
    shimmer = np.mean(np.abs(shimmer_diffs)) if len(shimmer_diffs) > 1 else 0

    return jitter, shimmer

# Analyze route to handle file and process analysis
@app.route("/analyze", methods=["POST"])
def analyze_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files are supported"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Extract features and make predictions
    mfcc_features = extract_mfcc_features(file_path)
    if mfcc_features is not None:
        # Convert numpy data to native Python data types
        mfcc_features = mfcc_features.tolist()

        # Scale the MFCC features
        mfcc_features_scaled = scaler.transform(np.array(mfcc_features).reshape(1, -1))
        prediction = svm_classifier.predict(mfcc_features_scaled)

        prediction_proba = svm_classifier.predict_proba(mfcc_features_scaled)
        confidence_score = float(prediction_proba[0][prediction[0]] * 100)  # Convert to float for JSON

        # Detect unnatural pauses
        pauses = detect_unnatural_pauses(file_path)

        # Calculate jitter and shimmer
        jitter, shimmer = calculate_jitter_and_shimmer(file_path)

        result = {
            "classification": "genuine" if prediction[0] == 0 else "deepfake",
            "confidence_score": confidence_score,
            "pauses": [round(pause, 2) for pause in pauses],  # Round pauses for better display
            "jitter": float(jitter),  # Convert jitter to float for JSON serialization
            "shimmer": float(shimmer)  # Convert shimmer to float for JSON serialization
        }

        return jsonify(result)

    return jsonify({"error": "Unable to process the audio file"}), 400

if __name__ == "__main__":
    app.run(debug=True)
