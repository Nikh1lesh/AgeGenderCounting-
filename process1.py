import cv2
import numpy as np
import dlib
import csv
import os
import face_recognition
import time
import tensorflow as tf

# === Load Models ===
face_detector = dlib.get_frontal_face_detector()

age_model = tf.keras.models.load_model("models/Age-VGG16.keras", compile=True)
gender_model = tf.keras.models.load_model("models/Gender-ResNet152.keras", compile=True)

# CSV File Setup
csv_file = "people_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Age", "Gender", "Timestamp"])

# Helper function to preprocess face image
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = tf.image.resize(face, [224, 224])
    face = tf.cast(face, tf.float32) / 255.0
    face = tf.expand_dims(face, axis=0)
    return face

# Function to predict age and gender
def predict_age_gender(face):
    preprocessed_face = preprocess_face(face)
    pred_age = int(tf.round(tf.squeeze(age_model.predict(preprocessed_face, verbose=0))))
    pred_gender = int(tf.round(tf.squeeze(gender_model.predict(preprocessed_face, verbose=0))))
    gender_label = "Male" if pred_gender == 0 else "Female"
    return pred_age, gender_label

# === Stream Processing Core ===
def process_stream(cap, csv_file_path, stop_check=None):
    frame_count = 0
    unique_faces = {}
    face_id_counter = 0
    last_face_ids = []  # Track the face IDs in the previous frame

    # Open the CSV file and check if it exists
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Face ID", "Gender", "Age"])

    while cap.isOpened():
        if stop_check and stop_check():
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        gender_age_data = []
        current_face_ids = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_crop = frame[top:bottom, left:right]
            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                continue

            # Check if face is already in the tracked faces
            found_id = None
            for face_id, data in unique_faces.items():
                if face_recognition.compare_faces([data['encoding']], face_encoding, tolerance=0.5)[0]:
                    found_id = face_id
                    break

            if found_id is None:
                # New face detected
                age, gender = predict_age_gender(face_crop)
                face_id = f"Face_{face_id_counter}"
                face_id_counter += 1
                unique_faces[face_id] = {
                    "encoding": face_encoding,
                    "gender": gender,
                    "age": age
                }
                # Write the new face information to the CSV
                with open(csv_file_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_count, face_id, gender, age])
                print(f"✅ New face added: {face_id} - {gender}, {age}")
            else:
                # Update the face information if necessary
                old_gender = unique_faces[found_id]["gender"]
                old_age = unique_faces[found_id]["age"]
                age, gender = predict_age_gender(face_crop)
                
                # If the data has changed, update the information
                if old_gender != gender or old_age != age:
                    unique_faces[found_id]["gender"] = gender
                    unique_faces[found_id]["age"] = age
                    
                    # Update the CSV with the new information (overwrite the old row)
                    with open(csv_file_path, "r") as file:
                        rows = list(csv.reader(file))
                    with open(csv_file_path, "w", newline="") as file:
                        writer = csv.writer(file)
                        for row in rows:
                            if row[1] == found_id:
                                row[2] = gender
                                row[3] = age
                            writer.writerow(row)
                    print(f"✅ Updated face: {found_id} - {gender}, {age}")
                else:
                    gender = unique_faces[found_id]["gender"]
                    age = unique_faces[found_id]["age"]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            gender_age_data.append((gender, age))

            # Track the current face ID
            current_face_ids.append(face_id)

        # Now check if unique faces increased but frame faces didn't increase
        if len(unique_faces) > len(last_face_ids) and len(current_face_ids) == len(last_face_ids):
            # Delete the second latest face
            if len(unique_faces) > 1:
                second_latest_face_id = list(unique_faces.keys())[-2]
                del unique_faces[second_latest_face_id]
                print(f"❌ Second latest face removed: {second_latest_face_id}")

        last_face_ids = current_face_ids  # Update the last detected face IDs

        cv2.putText(frame, f"Faces This Frame: {len(gender_age_data)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Unique Faces: {len(unique_faces)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Detection - Age & Gender", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# === Image Processing ===
def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_image, 1)

    for d in detected_faces:
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        face_crop = image[y1:y2, x1:x2]
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        age, gender = predict_age_gender(face_crop)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{gender}, {age}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"✅ Processed image saved to {output_path}")

# === Entry Points ===
def process_webcam(csv_file_path):
    global stop_flag
    stop_flag = False
    cap = cv2.VideoCapture(0)
    process_stream(cap, csv_file_path, stop_check=should_stop)

def process_video(video_path, csv_file_path):
    global stop_flag
    stop_flag = False
    cap = cv2.VideoCapture(video_path)
    process_stream(cap, csv_file_path)

def process_rtsp(rtsp_url, csv_file_path):
    global stop_flag
    stop_flag = False
    cap = cv2.VideoCapture(rtsp_url)
    process_stream(cap, csv_file_path, stop_check=should_stop)
