import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
from cvzone.FaceMeshModule import FaceMeshDetector

# --- CONFIGURATION ---
PATH_IMAGES = 'ImagesAttendance'
PATH_PROOFS = 'AttendanceProofs'
PATH_LOGS = 'Attendance_Logs'
LATE_CUTOFF = "08:30:00"

# --- BLINK SETTINGS (STRICTER) ---
BLINK_RATIO_THRESHOLD = 36     # Sensitivity (35-36 is a good balance)
BLINK_FRAMES_REQUIRED = 8      # You must HOLD blink for ~0.5 seconds.
                               # (Normal blinks are too fast to trigger this)

# --- DISPLAY SETTINGS ---
SUCCESS_DISPLAY_TIME = 40      # How long the Green "Success" box stays

# --- GLOBAL VARIABLES ---
marked_today = set()        
state = "SCANNING"          
blink_counter = 0           
success_timer = 0
current_user_profile = {}   

# Initialize Detectors
detector_blink = FaceMeshDetector(maxFaces=1)

# Ensure Directories Exist
if not os.path.exists(PATH_PROOFS): os.makedirs(PATH_PROOFS)
if not os.path.exists(PATH_LOGS): os.makedirs(PATH_LOGS)

# --- 1. LOAD DATABASE ---
print("[INFO] Loading Database & Encoding Faces...")
known_encodings = []
known_names = []
known_depts = []
known_ids = []

if not os.path.exists(PATH_IMAGES):
    print(f"[ERROR] '{PATH_IMAGES}' folder missing.")
    exit()

for item in os.listdir(PATH_IMAGES):
    full_path = os.path.join(PATH_IMAGES, item)
    image_to_load = None
    info_source = "" 

    if os.path.isdir(full_path):
        info_source = item
        for subfile in os.listdir(full_path):
            if subfile.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_to_load = os.path.join(full_path, subfile)
                break 
    else:
        if not item.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        info_source = os.path.splitext(item)[0]
        image_to_load = full_path

    if image_to_load is None: continue

    parts = info_source.split('_')
    name = parts[0]
    dept = parts[1] if len(parts) > 1 else "Gen"
    staff_id = parts[2] if len(parts) > 2 else "000"

    img = cv2.imread(image_to_load)
    if img is None: continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        encode = face_recognition.face_encodings(img)[0]
        known_encodings.append(encode)
        known_names.append(name)
        known_depts.append(dept)
        known_ids.append(staff_id)
        print(f" > Loaded: {name} | {dept}")
    except IndexError:
        print(f" [WARNING] No face found for {name}")

print(f"[INFO] System Ready. {len(known_encodings)} users loaded.")

# --- 2. DAILY LOG SETUP ---
def get_today_csv():
    today_str = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(PATH_LOGS, f'Attendance_{today_str}.csv'), today_str

csv_path, today_date = get_today_csv()
if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        f.write('Name,Department,StaffID,Date,Time,Status,ProofPath\n')

# --- 3. RECOVER STATE ---
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 3:
                marked_today.add(parts[2].strip())

# --- HELPER: SAVE DATA ---
def save_attendance(profile, frame):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    status = "LATE" if time_str > LATE_CUTOFF else "ON TIME"
    
    daily_proof_path = os.path.join(PATH_PROOFS, today_date)
    if not os.path.exists(daily_proof_path): os.makedirs(daily_proof_path)
    
    safe_id = "".join([c for c in profile['id'] if c.isalnum() or c in ('-','_')])
    proof_name = f"{safe_id}_{time_str.replace(':','-')}.jpg"
    full_path = os.path.join(daily_proof_path, proof_name)
    cv2.imwrite(full_path, frame)
    
    current_csv, _ = get_today_csv()
    with open(current_csv, 'a') as f:
        f.write(f'\n{profile["name"]},{profile["dept"]},{profile["id"]},{today_date},{time_str},{status},{full_path}')
    return status

# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img_display = img.copy()

    img, facesMesh = detector_blink.findFaceMesh(img, draw=False)
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    face_detected = False

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        face_detected = True
        matches = face_recognition.compare_faces(known_encodings, encodeFace, tolerance=0.5)
        faceDis = face_recognition.face_distance(known_encodings, encodeFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if matches[matchIndex]:
            name = known_names[matchIndex]
            dept = known_depts[matchIndex]
            staff_id = known_ids[matchIndex]

            # --- A. ALREADY LOGGED ---
            if staff_id in marked_today:
                state = "ALREADY_DONE"
                blink_counter = 0 # Reset
                
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img_display, "LOGGED IN", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                
                info_x, info_y = 50, 50
                cv2.rectangle(img_display, (info_x, info_y), (info_x + 500, info_y + 200), (0, 0, 0), cv2.FILLED)
                cv2.putText(img_display, f"NAME: {name}", (info_x+20, info_y+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.putText(img_display, f"DEPT: {dept}", (info_x+20, info_y+100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.putText(img_display, f"ID:   {staff_id}", (info_x+20, info_y+150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # --- B. NOT LOGGED (Scanning) ---
            else:
                current_user_profile = {"name": name, "dept": dept, "id": staff_id}
                
                if state != "LOGGED_SUCCESS":
                    # Default: Red Box
                    color_box = (0, 0, 255)
                    msg_text = "HOLD BLINK TO VERIFY"
                    
                    if facesMesh:
                        face = facesMesh[0]
                        leftUp, leftDown = face[159], face[23]
                        leftLeft, leftRight = face[130], face[243]
                        v_dist, _ = detector_blink.findDistance(leftUp, leftDown)
                        h_dist, _ = detector_blink.findDistance(leftLeft, leftRight)
                        ratio = int((v_dist / h_dist) * 100) if h_dist != 0 else 100
                        
                        # --- STRICT BLINK CHECK ---
                        if ratio < BLINK_RATIO_THRESHOLD:
                            blink_counter += 1
                            color_box = (0, 165, 255) # Orange while holding
                            msg_text = "VERIFYING..."
                        else:
                            blink_counter = 0 

                        # --- TRIGGER ACTION ---
                        if blink_counter >= BLINK_FRAMES_REQUIRED:
                            save_attendance(current_user_profile, img)
                            marked_today.add(current_user_profile['id'])
                            state = "LOGGED_SUCCESS"
                            blink_counter = 0

                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color_box, 2)
                    cv2.putText(img_display, msg_text, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_box, 2)

                # --- C. JUST LOGGED (Show Success) ---
                elif state == "LOGGED_SUCCESS":
                    success_timer += 1
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_display, "ATTENDANCE SAVED!", (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                    
                    if success_timer >= SUCCESS_DISPLAY_TIME:
                        state = "SCANNING"
                        success_timer = 0

        else:
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.putText(img_display, "UNKNOWN", (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 100), 2)
            blink_counter = 0

    if not face_detected:
        blink_counter = 0
        success_timer = 0
        if state == "LOGGED_SUCCESS": state = "SCANNING"

    cv2.imshow('Presentation Mode Attendance', img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()