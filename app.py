from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
from datetime import datetime
from utils.process import process_webcam, process_video, process_rtsp, process_image, stop_stream


app = Flask(__name__)

# Directories
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
CSV_FOLDER = "static/csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

csv_file_path = None  # Path to current CSV file

def create_new_csv():
    global csv_file_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = os.path.join(CSV_FOLDER, f"people_data_{timestamp}_{uuid.uuid4().hex}.csv")
    with open(csv_file_path, "w") as f:
        f.write("Frame,Count,Gender,Age\n")

# 🏠 Home route
@app.route("/")
def index():
    return render_template("index.html")

# 📸 Image upload page
@app.route("/image")
def image_page():
    return render_template("image.html")

# 🎞️ Video upload page
@app.route("/video")
def video_page():
    return render_template("video.html")

# 📷 Webcam processing page
@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")

# 🌐 RTSP stream page
@app.route("/rtsp")
def rtsp_page():
    return render_template("rtsp.html")

# 🔄 Upload image for processing
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file:
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        output_path = os.path.join(PROCESSED_FOLDER, filename)
        process_image(filepath, output_path)

        return jsonify({
            "message": "Image uploaded and processed successfully",
            "filepath": output_path
        })

    return jsonify({"error": "No file uploaded"})

# 🎥 Upload video for processing
@app.route("/process_video", methods=["POST"])
def process_video_route():
    file = request.files.get("video")
    if file:
        create_new_csv()
        filename = f"{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        process_video(filepath, csv_file_path)

        return jsonify({
            "message": "Video processed successfully",
            "filepath": filepath,
            "csv": csv_file_path
        })

    return jsonify({"error": "No video uploaded"})

# 🟢 Start webcam
@app.route("/start_webcam", methods=["GET"])
def start_webcam():
    create_new_csv()
    process_webcam(csv_file_path)
    return jsonify({"message": "Webcam processing started"})

# 🟢 Start RTSP stream
@app.route("/start_rtsp", methods=["POST"])
def start_rtsp():
    data = request.get_json()
    rtsp_url = data.get("rtsp_url")
    if not rtsp_url:
        return jsonify({"error": "No RTSP URL provided"})

    create_new_csv()
    process_rtsp(rtsp_url, csv_file_path)
    return jsonify({"message": "RTSP stream processing started"})

# 🔴 Stop any stream (webcam or RTSP)
@app.route("/stop_stream", methods=["GET"])
def stop_stream_route():
    stop_stream()
    return jsonify({"message": "Stream stopped"})

# 📥 Download latest CSV
@app.route("/download_csv", methods=["GET"])
def download_csv():
    if csv_file_path and os.path.exists(csv_file_path):
        return send_file(csv_file_path, as_attachment=True)
    return jsonify({"error": "No CSV file available"})

if __name__ == "__main__":
    app.run(debug=True)
