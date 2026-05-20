import os
import glob
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to extract MFCC features from an audio file
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

# Function to create a dataset of MFCC features from a directory of audio files
def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {directory}. Please check the directory path.")
    
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")
    
    print(f"Number of samples in {directory}: {len(X)}")
    return X, y

# Function to train the model
def train_model(X, y):
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("At least 2 sets are required to train")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_classifier = SVC(kernel='linear', random_state=42, probability=True)
    svm_classifier.fit(X_train_scaled, y_train)

    y_pred = svm_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_mtx)

    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)

# Function to plot donut chart of confidence score
def plot_donut_chart(confidence_score):
    score = confidence_score  # The confidence score from the model (in percentage)
    remaining = 100 - score  # The remaining part (100% - confidence score)

    # Data for the donut chart
    labels = ['Confidence', 'Remaining']
    sizes = [score, remaining]
    colors = ['#66b3ff', '#f2f2f2']
    explode = (0.1, 0)  # "explode" the first slice (Confidence) for better visibility

    # Plotting the donut chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                      startangle=90, colors=colors, wedgeprops=dict(width=0.3))

    # Draw a circle at the center to create the 'donut' shape
    centre_circle = plt.Circle((0, 0), 0.70, color='white', fc='white', linewidth=1.25)
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    # Add title to the chart
    plt.title("Confidence Score Visualization")

    # Show the plot
    plt.show()

# Function to detect unnatural pauses in audio and visualize the waveform
def detect_and_visualize_unnatural_pauses(audio_path, silence_threshold=0.5, pause_threshold=1.0):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)

        # Split audio into non-silent segments
        non_silent_intervals = librosa.effects.split(audio_data, top_db=silence_threshold)

        # Calculate the pauses between non-silent segments
        pause_durations = []
        pause_intervals = []
        for i in range(1, len(non_silent_intervals)):
            start_time = non_silent_intervals[i-1][1] / sr  # Convert from samples to seconds
            end_time = non_silent_intervals[i][0] / sr  # Convert from samples to seconds
            pause_duration = end_time - start_time

            if pause_duration > pause_threshold:
                pause_durations.append(pause_duration)
                pause_intervals.append((non_silent_intervals[i-1][1], non_silent_intervals[i][0]))

        # Visualize the waveform and silences
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(audio_data, sr=sr, alpha=0.6, color='b')

        # Highlight silent regions (colored in green)
        for interval in non_silent_intervals:
            plt.axvspan(interval[0] / sr, interval[1] / sr, color='g', alpha=0.5)

        # Highlight unnatural pauses (colored in red)
        for pause in pause_intervals:
            plt.axvspan(pause[0] / sr, pause[1] / sr, color='r', alpha=0.5)

        plt.title('Waveform with Silent Segments and Unnatural Pauses', fontsize=16)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True)
        plt.show()

        return pause_durations

    except Exception as e:
        print(f"Error detecting pauses in the audio: {e}")
        return []

# Function to calculate Jitter and Shimmer
def calculate_jitter_and_shimmer(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Estimate the pitch (frequency) using librosa's piptrack
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

# Function to analyze and plot results for audio
def analyze_audio(input_audio_path):
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
        return
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")
        return

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = svm_classifier.predict(mfcc_features_scaled)

        prediction_proba = svm_classifier.predict_proba(mfcc_features_scaled)
        confidence_score = prediction_proba[0][prediction[0]] * 100  # Convert to percentage

        if prediction[0] == 0:
            print(f"The input audio is classified as genuine. Confidence Score: {confidence_score:.2f}%")
        else:
            print(f"The input audio is classified as deepfake. Confidence Score: {confidence_score:.2f}%")

        plot_donut_chart(confidence_score)

        pauses = detect_and_visualize_unnatural_pauses(input_audio_path)
        if pauses:
            print(f"Unnatural pauses detected (greater than 1 second):")
            for idx, pause in enumerate(pauses, 1):
                print(f"Pause {idx}: {pause:.2f} seconds")
        else:
            print("No unnatural pauses detected.")

        jitter, shimmer = calculate_jitter_and_shimmer(input_audio_path)
        print(f"Jitter: {jitter:.4f}")
        print(f"Shimmer: {shimmer:.4f}")
        plot_jitter_shimmer(jitter, shimmer)
    else:
        print("Error: Unable to process the input audio.")

# Function to plot Jitter and Shimmer values
def plot_jitter_shimmer(jitter, shimmer):
    plt.figure(figsize=(6, 4))
    plt.bar(['Jitter', 'Shimmer'], [jitter, shimmer], color=['blue', 'green'])
    plt.title('Jitter and Shimmer Analysis')
    plt.ylabel('Value')
    plt.show()

# Main function to train and analyze audio
def main():
    # Update the paths to the correct directories
    genuine_dir = r"C:\Users\anish\Downloads\CADENCE\backend\DeepFake-Audio-Detection-MFCC\real_audio"
    deepfake_dir = r"C:\Users\anish\Downloads\CADENCE\backend\DeepFake-Audio-Detection-MFCC\deepfake_audio"

    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

    # If there are not enough samples, handle it gracefully
    if len(X_genuine) < 2 or len(X_deepfake) < 2:
        print("Each class should have at least two samples for stratified splitting.")
        print("Combining both classes into one for training.")
        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))
    else:
        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))

    train_model(X, y)

if __name__ == "__main__":
    main()

    user_input_file = input("Enter the path of the .wav file to analyze: ")
    analyze_audio(user_input_file)
