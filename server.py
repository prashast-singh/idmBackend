from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for matplotlib 1
import matplotlib.pyplot as plt

import io
import base64
import tempfile
from scipy.signal import find_peaks

app = Flask(__name__)
CORS(app)

# -----------------------
# 1. Initialize MediaPipe Pose Once
# -----------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# -----------------------
# 2. Helper Functions
# -----------------------
def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three 2D points a, b, c,
    where b is the vertex.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def process_video_mediapipe(video_path):
    """
    Process video using MediaPipe Pose. Return a DataFrame with
    frame-by-frame angles and nose Y position (normalized [0,1]).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Nose Y in [0,1]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_y = nose.y  # already normalized

            # Right side landmarks (x,y) in [0,1]
            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Prepare them as (x, y) for angle calculation
            hip_pt = (hip.x, hip.y)
            knee_pt = (knee.x, knee.y)
            ankle_pt = (ankle.x, ankle.y)
            shoulder_pt = (shoulder.x, shoulder.y)

            # Calculate angles
            hip_angle = calculate_angle(shoulder_pt, hip_pt, knee_pt)
            knee_angle = calculate_angle(hip_pt, knee_pt, ankle_pt)
            trunk_angle = calculate_angle((hip.x, hip.y - 0.01), hip_pt, shoulder_pt)

            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_data.append([current_frame, nose_y, hip_angle, knee_angle, trunk_angle])

    cap.release()

    df = pd.DataFrame(frame_data, columns=["Frame", "Nose_Y", "Hip_Angle", "Knee_Angle", "Trunk_Angle"])
    df["Time"] = df["Frame"] / fps
    return df

def create_base64_plot(df):
    """
    1) Detect peaks in Nose_Y (to count squats).
    2) Create plots for Nose_Y, Hip/Knee/Trunk angles.
    3) Return the number of squats and a dict of base64-encoded plot images.
    """
    # Smooth Nose_Y for less noisy peak detection
    df["Nose_Y_Smooth"] = df["Nose_Y"].rolling(window=5, min_periods=1).mean()

    # Detect minima (squats) by finding peaks in the negative of Nose_Y
    peaks, _ = find_peaks(-df["Nose_Y_Smooth"], prominence=0.1, distance=30)
    num_squats = len(peaks)

    plots_base64 = {}

    # 1) Squat detection plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Time"], df["Nose_Y_Smooth"], label="Smoothed Nose Y")
    ax.plot(df["Time"][peaks], df["Nose_Y_Smooth"][peaks], "x", label="Detected Peaks")
    ax.set_title("Detected Squats (MediaPipe)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Nose Y (normalized)")
    ax.legend()

    plots_base64["squats_detection"] = fig_to_base64(fig)
    plt.close(fig)

    # 2) Hip angle
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Time"], df["Hip_Angle"], label="Hip Angle")
    ax.set_title("Hip Angle Over Time (MediaPipe)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()

    plots_base64["hip_angle"] = fig_to_base64(fig)
    plt.close(fig)

    # 3) Knee angle
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Time"], df["Knee_Angle"], label="Knee Angle")
    ax.set_title("Knee Angle Over Time (MediaPipe)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()

    plots_base64["knee_angle"] = fig_to_base64(fig)
    plt.close(fig)

    # 4) Trunk angle
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Time"], df["Trunk_Angle"], label="Trunk Angle")
    ax.set_title("Trunk Angle Over Time (MediaPipe)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()

    plots_base64["trunk_angle"] = fig_to_base64(fig)
    plt.close(fig)

    return num_squats, plots_base64

def fig_to_base64(fig):
    """
    Convert a Matplotlib figure to a base64-encoded PNG image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# -----------------------
# 3. Flask Route
# -----------------------
@app.route("/analyze", methods=["POST"])
def analyze_video():
    """
    Endpoint that only uses MediaPipe. No permanent file saving.
    Returns JSON with number of squats and base64-encoded plots.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    video_bytes = file.read()

    # We need a temporary file for OpenCV reading
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
        temp.write(video_bytes)
        temp.flush()

        # Process with MediaPipe
        mediapipe_df = process_video_mediapipe(temp.name)
        num_squats, plots_media = create_base64_plot(mediapipe_df)

    # Return results in JSON
    return jsonify({
        "num_squats": num_squats,
        "plots_base64": plots_media
    })

if __name__ == "__main__":
    app.run(debug=True)
