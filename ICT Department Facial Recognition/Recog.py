import cv2
import pickle
import os
import csv
import time
from datetime import datetime

# --- CONFIGURATION ---
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
DEPARTMENT_NAME = "ICT" 
COOLDOWN_SECONDS = 10  # The 10-second gap you requested

# Folders
PROOF_FOLDER = "Attendance_Proofs"
LOGS_FOLDER = "Attendance_Logs"
if not os.path.exists(PROOF_FOLDER): os.makedirs(PROOF_FOLDER)
if not os.path.exists(LOGS_FOLDER): os.makedirs(LOGS_FOLDER)

# --- STATE VARIABLES ---
REQUIRED_FRAMES = 35  # ~1.5 seconds to verify
patience_timer = 0    
current_person = None 

# Track when a user was last successfully logged
# Format: { "Name_ID": timestamp }
last_logged_timestamps = {}

# 1. Load Brain & Labels
print("[INFO] Loading System Brain...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_FILE)

with open(LABELS_FILE, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# 2. Setup Today's Memory
today_str = datetime.now().strftime("%Y-%m-%d")
csv_filename = os.path.join(LOGS_FOLDER, f"Attendance_{today_str}.csv")
attendance_memory = set()

if os.path.exists(csv_filename):
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) > 2:
                # Key = Name_ID
                key = f"{row[0]}_{row[2]}"
                attendance_memory.add(key)
                # We set the timestamp to 0 so these people go straight to "Info Mode"
                # (Since they logged in hours ago, we don't need the 10s gap)
                last_logged_timestamps[key] = 0 
    print(f"[INFO] Loaded {len(attendance_memory)} records.")
else:
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Department", "Staff_ID", "Time", "Date", "Image_Proof"])

# --- HELPER FUNCTION ---
def log_attendance(full_label, frame):
    # Parse Name/ID
    try:
        split_index = full_label.rfind('_')
        name = full_label[:split_index].replace("_", " ")
        staff_id = full_label[split_index+1:]
    except:
        name = full_label; staff_id = "Unknown"

    unique_key = f"{name}_{staff_id}"
    
    # 1. Save Proof
    daily_proof_path = os.path.join(PROOF_FOLDER, today_str)
    if not os.path.exists(daily_proof_path): os.makedirs(daily_proof_path)
    
    time_str = datetime.now().strftime("%H-%M-%S")
    image_name = f"{name}_{time_str}.jpg"
    cv2.imwrite(os.path.join(daily_proof_path, image_name), frame)

    # 2. Log to CSV
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, DEPARTMENT_NAME, staff_id, time_str, today_str, image_name])

    # 3. Update Memory
    attendance_memory.add(unique_key)
    # Set the timer for the 10-second gap
    last_logged_timestamps[unique_key] = time.time()
    
    return True, name, staff_id

# --- MAIN LOOP ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
cam.set(3, 1280); cam.set(4, 720)

print("[START] Visual Attendance System Running...")

while True:
    ret, frame = cam.read()
    if not ret: break
    frame = cv2.flip(frame, 1) 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        patience_timer = 0
        current_person = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < 75:
            raw_label = labels[id_]
            
            # Create the unique key for lookup
            try:
                s_ind = raw_label.rfind('_')
                clean_name = raw_label[:s_ind].replace('_',' ')
                clean_id = raw_label[s_ind+1:]
                unique_key = f"{clean_name}_{clean_id}"
            except: 
                unique_key = raw_label
                clean_name = raw_label
                clean_id = "Unknown"

            # --- LOGIC BRANCHING ---

            # BRANCH A: Person is ALREADY logged in
            if unique_key in attendance_memory:
                patience_timer = 0 # No need to verify again
                
                # Check how long ago they logged in
                time_since_log = time.time() - last_logged_timestamps.get(unique_key, 0)
                
                if time_since_log < COOLDOWN_SECONDS:
                    # SCENARIO 1: THE 10-SECOND GAP
                    # Just show "Attendance Collected" (Green)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "ATTENDANCE COLLECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                else:
                    # SCENARIO 2: AFTER 10 SECONDS (INFO MODE)
                    # Show full details on SCREEN overlay (Top Left)
                    # Draw a nice dark background for the text
                    cv2.rectangle(frame, (20, 20), (500, 130), (0, 0, 0), -1) 
                    cv2.putText(frame, f"NAME: {clean_name}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"DEPT: {DEPARTMENT_NAME}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"ID:   {clean_id}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Green box around face still
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # BRANCH B: Person is NEW (Need to verify)
            else:
                if current_person == raw_label:
                    patience_timer += 1
                else:
                    current_person = raw_label
                    patience_timer = 0
                
                # Draw Loading Bar
                progress = patience_timer / REQUIRED_FRAMES
                bar_x, bar_y, bar_w = x, y - 30, w
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (100, 100, 100), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10), (0, 255, 255), -1)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "VERIFYING...", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Trigger Log
                if patience_timer >= REQUIRED_FRAMES:
                    success, n, i = log_attendance(raw_label, frame)
                    if success:
                        print(f"Logged: {n}")
                        patience_timer = 0
                        # NOTE: The loop continues immediately. 
                        # Next frame, it will hit 'BRANCH A' -> 'SCENARIO 1' automatically.

        else:
            # UNKNOWN
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            patience_timer = 0
            current_person = None

    cv2.imshow('Final Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv2.destroyAllWindows()