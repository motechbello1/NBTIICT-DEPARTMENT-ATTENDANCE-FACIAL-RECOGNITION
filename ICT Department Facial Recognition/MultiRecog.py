import cv2
import numpy as np
import face_recognition
import os
import math
from datetime import datetime
from cvzone.FaceMeshModule import FaceMeshDetector

# --- CONFIGURATION ---
PATH_IMAGES = 'ImagesAttendance'
PATH_PROOFS = 'AttendanceProofs'
PATH_LOGS = 'Attendance_Logs'
LATE_CUTOFF = "08:30:00"

# --- BLINK SETTINGS (LIVENESS CHECK) ---
MAX_FACES = 5                 
BLINK_RATIO_THRESHOLD = 36    
BLINK_FRAMES_REQUIRED = 6     # Must hold blink for ~0.3s to prove liveness

# --- DISPLAY SETTINGS ---
SUCCESS_DISPLAY_TIME = 60     # How long the "Welcome" green screen stays

# --- GLOBAL VARIABLES ---
marked_today = set()        
# Dictionary tracks state for EACH face: { 'ID': { 'blink_count': 0, 'success_timer': 0 } }
user_states = {} 

detector_blink = FaceMeshDetector(maxFaces=MAX_FACES)

# Ensure Directories Exist
if not os.path.exists(PATH_PROOFS): os.makedirs(PATH_PROOFS)
if not os.path.exists(PATH_LOGS): os.makedirs(PATH_LOGS)

# --- 1. LOAD DATABASE ---
print("[INFO] Loading Database...")
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

    # Parse: Name_Dept_ID
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
        print(f" > Loaded: {name}")
    except IndexError:
        print(f" [WARNING] No face found for {name}")

print(f"[INFO] System Ready. Loaded {len(known_encodings)} profiles.")

# --- 2. LOG SETUP ---
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
def save_attendance(name, dept, staff_id, frame):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    status = "LATE" if time_str > LATE_CUTOFF else "ON TIME"
    
    daily_proof_path = os.path.join(PATH_PROOFS, today_date)
    if not os.path.exists(daily_proof_path): os.makedirs(daily_proof_path)
    
    safe_id = "".join([c for c in staff_id if c.isalnum() or c in ('-','_')])
    proof_name = f"{safe_id}_{time_str.replace(':','-')}.jpg"
    full_path = os.path.join(daily_proof_path, proof_name)
    cv2.imwrite(full_path, frame)
    
    current_csv, _ = get_today_csv()
    with open(current_csv, 'a') as f:
        f.write(f'\n{name},{dept},{staff_id},{today_date},{time_str},{status},{full_path}')

# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img_display = img.copy()

    # Detect Blinks (Liveness)
    img, facesMesh = detector_blink.findFaceMesh(img, draw=False)
    
    # Detect Identities
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(known_encodings, encodeFace, tolerance=0.5)
        faceDis = face_recognition.face_distance(known_encodings, encodeFace)
        
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        
        # Center of face for mesh matching
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2

        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            # === KNOWN PERSON ===
            name = known_names[matchIndex]
            dept = known_depts[matchIndex]
            staff_id = known_ids[matchIndex]

            # Init State
            if staff_id not in user_states:
                user_states[staff_id] = {'blink_count': 0, 'success_timer': 0}
            state_info = user_states[staff_id]

            # --- CASE 1: ALREADY LOGGED (YELLOW) ---
            if staff_id in marked_today and state_info['success_timer'] == 0:
                color = (0, 255, 255) # Yellow
                
                # Box
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                # Header
                cv2.rectangle(img_display, (x1, y1-40), (x2, y1), color, cv2.FILLED)
                cv2.putText(img_display, "ALREADY LOGGED", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)
                
                # Details Below
                cv2.rectangle(img_display, (x1, y2), (x2, y2+90), color, cv2.FILLED)
                cv2.putText(img_display, f"{name}", (x1+5, y2+25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)
                cv2.putText(img_display, f"Dept: {dept}", (x1+5, y2+50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)
                cv2.putText(img_display, f"ID: {staff_id}", (x1+5, y2+75), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)

            # --- CASE 2: JUST LOGGED / SUCCESS (GREEN) ---
            elif state_info['success_timer'] > 0:
                state_info['success_timer'] -= 1
                color = (0, 255, 0) # Green
                
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                # Welcome Header
                cv2.rectangle(img_display, (x1, y1-40), (x2, y1), color, cv2.FILLED)
                cv2.putText(img_display, f"WELCOME {name}", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
                
                # Details Below
                cv2.rectangle(img_display, (x1, y2), (x2, y2+60), color, cv2.FILLED)
                cv2.putText(img_display, f"Department: {dept}", (x1+5, y2+25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
                cv2.putText(img_display, f"Staff ID: {staff_id}", (x1+5, y2+50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
            
            # --- CASE 3: SCANNING / PROVING LIVENESS (RED) ---
            else:
                # Find matching mesh
                best_mesh = None
                min_dist = 99999
                if facesMesh:
                    for mesh_face in facesMesh:
                        mesh_x, mesh_y = mesh_face[1] 
                        dist = math.hypot(mesh_x - face_center_x, mesh_y - face_center_y)
                        if dist < 150: 
                            if dist < min_dist:
                                min_dist = dist
                                best_mesh = mesh_face

                color = (0, 0, 255) # Red (Unverified / Potential Fake)
                msg = "HOLD BLINK"
                
                if best_mesh:
                    leftUp, leftDown = best_mesh[159], best_mesh[23]
                    leftLeft, leftRight = best_mesh[130], best_mesh[243]
                    v_dist, _ = detector_blink.findDistance(leftUp, leftDown)
                    h_dist, _ = detector_blink.findDistance(leftLeft, leftRight)
                    ratio = int((v_dist / h_dist) * 100) if h_dist != 0 else 100
                    
                    if ratio < BLINK_RATIO_THRESHOLD:
                        state_info['blink_count'] += 1
                        color = (0, 165, 255) # Orange (Verifying Liveness)
                        msg = "VERIFYING..."
                    else:
                        state_info['blink_count'] = 0 
                    
                    if state_info['blink_count'] >= BLINK_FRAMES_REQUIRED:
                        save_attendance(name, dept, staff_id, img)
                        marked_today.add(staff_id)
                        state_info['success_timer'] = SUCCESS_DISPLAY_TIME
                        state_info['blink_count'] = 0
                        # Force refresh next loop to switch to Green instantly
                        continue 

                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img_display, (x1, y1-30), (x2, y1), color, cv2.FILLED)
                cv2.putText(img_display, msg, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

        else:
            # === UNKNOWN FACE (RED) ===
            color = (0, 0, 255)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img_display, (x1, y1-30), (x2, y1), color, cv2.FILLED)
            cv2.putText(img_display, "UNKNOWN", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Robust Attendance System', img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()