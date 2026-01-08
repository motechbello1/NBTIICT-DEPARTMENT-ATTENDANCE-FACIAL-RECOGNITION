import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
from cvzone.FaceMeshModule import FaceMeshDetector

# --- CONFIGURATION ---
PATH_IMAGES = r'C:\Users\hello\Desktop\NBTI PROJECTS\ICT Department Facial Recognition\ImagesAttendance'
PATH_PROOFS = 'AttendanceProofs'
PATH_LOGS = 'Attendance_Logs'
LATE_CUTOFF = "08:30:00"

# --- PERFORMANCE TUNING ---
FRAME_SKIP = 2                 # Only run heavy recognition every 2 frames
BLINK_FRAMES_REQUIRED = 5      # Reduced from 8 for "Instant" feel
BLINK_RATIO_THRESHOLD = 36     # Sensitivity

# --- DISPLAY SETTINGS ---
SUCCESS_DISPLAY_TIME = 40      

# --- GLOBAL VARIABLES ---
marked_today = set()        
state = "SCANNING"          
blink_counter = 0           
success_timer = 0
current_user_profile = {}   

# Variables for Frame Skipping (Cache)
frame_count = 0
last_face_locations = []
last_face_names = []
last_face_roles = []
last_face_ids = []
last_face_depts = []

# Initialize Detectors
detector_blink = FaceMeshDetector(maxFaces=2) 

# Ensure Directories Exist
if not os.path.exists(PATH_PROOFS): os.makedirs(PATH_PROOFS)
if not os.path.exists(PATH_LOGS): os.makedirs(PATH_LOGS)

# --- 1. QUIET LOADING SYSTEM ---
print("="*50)
print("[INFO] STARTING HIGH-PERFORMANCE SYSTEM...")
print(f"[INFO] Scanning Database: {PATH_IMAGES}")
print("="*50)

known_encodings = []
known_names = []
known_depts = []
known_roles = []
known_ids = []

if not os.path.exists(PATH_IMAGES):
    print(f"[ERROR] Folder not found: {PATH_IMAGES}")
    exit()

for root, dirs, files in os.walk(PATH_IMAGES):
    if root == PATH_IMAGES: continue

    folder_name = os.path.basename(root)
    parts = folder_name.split('_')

    if len(parts) >= 4:
        name = parts[0]
        dept = parts[1]
        role = parts[2]
        staff_id = parts[3]
    else:
        continue

    print(f"[LOADING] Profile found: {name} ({role})")

    valid_images = 0
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            img_file = cv2.imread(full_path)
            if img_file is None: continue
            img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
            try:
                encodes = face_recognition.face_encodings(img_file)
                if len(encodes) > 0:
                    known_encodings.append(encodes[0])
                    known_names.append(name)
                    known_depts.append(dept)
                    known_roles.append(role)
                    known_ids.append(staff_id)
                    valid_images += 1
            except Exception:
                pass

print("-" * 50)
print(f"[READY] System initialized. {len(known_encodings)} face samples loaded.")
print("-" * 50)

# --- 2. LOGGING SETUP ---
def get_today_csv():
    today_str = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(PATH_LOGS, f'Attendance_{today_str}.csv'), today_str

csv_path, today_date = get_today_csv()
if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        f.write('Name,Department,Role,StaffID,Date,Time,Status,ProofPath\n')

# --- 3. RECOVER STATE ---
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: 
            parts = line.split(',')
            if len(parts) >= 4:
                marked_today.add(parts[3].strip())

# --- HELPER: SAVE ATTENDANCE ---
def save_attendance_local(profile, frame):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    status = "LATE" if time_str > LATE_CUTOFF else "ON TIME"
    
    daily_proof_path = os.path.join(PATH_PROOFS, today_date)
    if not os.path.exists(daily_proof_path): os.makedirs(daily_proof_path)
    
    safe_id = "".join([c for c in profile['id'] if c.isalnum() or c in ('-','_')])
    proof_name = f"{safe_id}_{time_str.replace(':','-')}.jpg"
    full_proof_path = os.path.join(daily_proof_path, proof_name)
    cv2.imwrite(full_proof_path, frame)
    
    current_csv, _ = get_today_csv()
    with open(current_csv, 'a') as f:
        f.write(f'\n{profile["name"]},{profile["dept"]},{profile["role"]},{profile["id"]},{today_date},{time_str},{status},{full_proof_path}')
    
    print(f"[LOGGED] {profile['name']} at {time_str}")

# --- 4. OPTIMIZED MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img_display = img.copy()
    frame_count += 1

    # --- A. BLINK DETECTION (Every Frame for Smoothness) ---
    img, facesMesh = detector_blink.findFaceMesh(img, draw=False)
    
    # --- B. RECOGNITION (Throttled for Performance) ---
    # Only run heavy recognition every 'FRAME_SKIP' frames
    if frame_count % FRAME_SKIP == 0 and state != "LOGGED_SUCCESS":
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        last_face_locations = face_recognition.face_locations(imgS)
        encodings = face_recognition.face_encodings(imgS, last_face_locations)

        last_face_names = []
        last_face_roles = []
        last_face_ids = []
        last_face_depts = []

        for encodeFace in encodings:
            matches = face_recognition.compare_faces(known_encodings, encodeFace, tolerance=0.5)
            faceDis = face_recognition.face_distance(known_encodings, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                last_face_names.append(known_names[matchIndex])
                last_face_roles.append(known_roles[matchIndex])
                last_face_ids.append(known_ids[matchIndex])
                last_face_depts.append(known_depts[matchIndex])
            else:
                last_face_names.append("Unknown")
                last_face_roles.append("")
                last_face_ids.append("")
                last_face_depts.append("")

    # --- C. PROCESS RESULTS ---
    # If we are in success mode, just show the success box and countdown
    if state == "LOGGED_SUCCESS":
        success_timer += 1
        # Draw a generic green box in center
        cv2.rectangle(img_display, (440, 100), (840, 500), (0, 255, 0), 2)
        cv2.putText(img_display, "SUCCESS!", (480, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        cv2.putText(img_display, "Attendance Saved", (460, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        if success_timer >= SUCCESS_DISPLAY_TIME:
            state = "SCANNING"
            success_timer = 0
            blink_counter = 0

    # Normal Scanning Mode
    else:
        # Loop through the cached results from the last processed frame
        for i, (top, right, bottom, left) in enumerate(last_face_locations):
            # Scale back up (since we resized by 0.25)
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            
            # Check list bounds safely
            if i < len(last_face_names):
                name = last_face_names[i]
                role = last_face_roles[i]
                staff_id = last_face_ids[i]
                dept = last_face_depts[i]
            else:
                name = "Unknown"

            if name != "Unknown":
                # 1. ALREADY LOGGED
                if staff_id in marked_today:
                    color = (0, 255, 255) # Yellow
                    cv2.rectangle(img_display, (left, top), (right, bottom), color, 2)
                    cv2.putText(img_display, "LOGGED IN", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    
                    # Info Panel
                    info_x, info_y = 50, 50
                    cv2.rectangle(img_display, (info_x, info_y), (info_x + 550, info_y + 200), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img_display, f"NAME: {name}", (info_x+20, info_y+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.putText(img_display, f"ROLE: {role}", (info_x+20, info_y+100), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)

                # 2. NOT LOGGED - CHECK BLINK
                else:
                    color = (0, 0, 255) # Red
                    msg = "HOLD BLINK"
                    
                    # Blink Logic
                    if facesMesh:
                        face = facesMesh[0]
                        # Calculate Blink Ratio
                        leftUp, leftDown = face[159], face[23]
                        leftLeft, leftRight = face[130], face[243]
                        v_dist, _ = detector_blink.findDistance(leftUp, leftDown)
                        h_dist, _ = detector_blink.findDistance(leftLeft, leftRight)
                        ratio = int((v_dist / h_dist) * 100) if h_dist != 0 else 100
                        
                        if ratio < BLINK_RATIO_THRESHOLD:
                            blink_counter += 1
                            color = (0, 165, 255) # Orange
                            msg = "VERIFYING..."
                        else:
                            blink_counter = 0 
                            
                        if blink_counter >= BLINK_FRAMES_REQUIRED:
                            current_user_profile = {"name": name, "dept": dept, "role": role, "id": staff_id}
                            save_attendance_local(current_user_profile, img)
                            marked_today.add(staff_id)
                            state = "LOGGED_SUCCESS"
                            blink_counter = 0

                    cv2.rectangle(img_display, (left, top), (right, bottom), color, 2)
                    cv2.putText(img_display, msg, (left, top - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            else:
                # Unknown Face
                cv2.rectangle(img_display, (left, top), (right, bottom), (100, 100, 100), 2)

    cv2.imshow('NBTI High-Perf Attendance', img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()