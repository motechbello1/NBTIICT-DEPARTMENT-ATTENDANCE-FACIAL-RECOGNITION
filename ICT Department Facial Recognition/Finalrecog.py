import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO

# --- CONFIGURATION ---
PATH_IMAGES = r'C:\Users\hello\Desktop\NBTI PROJECTS\ICT Department Facial Recognition\ImagesAttendance'
PATH_PROOFS = 'AttendanceProofs'
PATH_LOGS = 'Attendance_Logs'
PATH_SUSPICIOUS = 'Suspicious_Behaviour_Proofs'

# --- TIME SETTINGS ---
LATE_CUTOFF = "08:30:00"
SIGNOUT_START_TIME = "10:44:00" 
SUCCESS_DISPLAY_TIME = 10 # Frames to show success message

# --- PERFORMANCE TUNING ---
FRAME_SKIP = 2               
BLINK_FRAMES_REQUIRED = 5    
BLINK_RATIO_THRESHOLD = 36   
RECOGNITION_TOLERANCE = 0.40 # Strict for normal attendance
SUSPICIOUS_TOLERANCE = 0.50  # Looser for security (phones often cast glare/shadows)
SUSPICIOUS_COOLDOWN_FRAMES = 30 # Take 1 photo every ~1 second if phone persists

# --- DISPLAY COLORS ---
COLOR_LOGIN = (0, 255, 0)      
COLOR_SIGNOUT = (0, 140, 255)  
COLOR_LOCKED = (128, 128, 128) 
COLOR_ALREADY = (0, 255, 255)  
COLOR_WARNING = (0, 0, 255)    # RED
COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_TEXT_SUB = (200, 200, 200)

# --- GLOBAL VARIABLES ---
attendance_cache = {}          
state = "SCANNING"          
blink_counter = 0           
success_timer = 0
suspicious_timer = 0         
current_user_profile = {}   
process_mode = "LOGIN" 

# Variables for Frame Skipping
frame_count = 0
last_face_locations = []
last_face_names = []
last_face_roles = []
last_face_ids = []
last_face_depts = []

# --- INITIALIZE DETECTORS ---
detector_blink = FaceMeshDetector(maxFaces=2) 
print("[INFO] Loading Anti-Spoofing Model (YOLOv8n)...")
model_spoof = YOLO("yolov8n.pt") 
CLASS_ID_PHONE = 67 

# Ensure Directories Exist
if not os.path.exists(PATH_PROOFS): os.makedirs(PATH_PROOFS)
if not os.path.exists(PATH_LOGS): os.makedirs(PATH_LOGS)
if not os.path.exists(PATH_SUSPICIOUS): os.makedirs(PATH_SUSPICIOUS)

# --- 1. LOAD DATABASE ---
print("="*50)
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
    
    # Expecting format: Name_Dept_Role_ID
    if len(parts) >= 4:
        name = parts[0].replace("_", " ")
        dept = parts[1]
        role = parts[2]
        staff_id = parts[3]
    else: continue

    print(f"[LOADING] Profile found: {name} ({role})")

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
            except Exception: pass

print("-" * 50)
print(f"[READY] System initialized. {len(known_encodings)} face samples loaded.")
print("-" * 50)

# --- CSV & PATH HELPERS ---
def get_today_csv():
    today_str = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(PATH_LOGS, f'Attendance_{today_str}.csv'), today_str

csv_path, today_date = get_today_csv()

if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        f.write('Name,Department,Role,StaffID,Date,TimeIn,Status,ProofIn,TimeOut,ProofOut\n')

if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: 
            parts = line.strip().split(',')
            if len(parts) >= 4:
                s_id = parts[3].strip()
                if len(parts) > 8 and parts[8].strip() != "":
                    attendance_cache[s_id] = "SIGNED_OUT"
                else:
                    attendance_cache[s_id] = "LOGGED_IN"

# --- HELPER: SAVE SUSPICIOUS PROOF ---
def save_suspicious_proof(raw_clean_frame, detected_name, detected_id):
    """
    Saves the frame to Suspicious_Behaviour_Proofs / Date / Staff_ID_Name
    """
    # 1. Determine Folder Name
    if detected_id in ["N/A", "Unknown", ""]:
        folder_name = "Unknown_Suspects"
    else:
        # Sanitize strings to be folder-safe
        safe_name = detected_name.replace(" ", "_")
        safe_id = "".join([c for c in detected_id if c.isalnum() or c in ('-','_')])
        folder_name = f"{safe_id}_{safe_name}"

    # 2. Build Paths
    today_str = datetime.now().strftime('%Y-%m-%d')
    daily_path = os.path.join(PATH_SUSPICIOUS, today_str)
    suspect_path = os.path.join(daily_path, folder_name)
    
    # 3. Create Directories (Recursively)
    os.makedirs(suspect_path, exist_ok=True)
    
    # 4. Resize if too large (save storage)
    h, w, c = raw_clean_frame.shape
    if w > 1280:
        scale_ratio = 1280 / w
        new_h = int(h * scale_ratio)
        processed_img = cv2.resize(raw_clean_frame, (1280, new_h))
    else:
        processed_img = raw_clean_frame
    
    # 5. Save with timestamp
    now_str = datetime.now().strftime('%H-%M-%S-%f')[:-3]
    filename = f"EVIDENCE_{now_str}.jpg"
    full_path = os.path.join(suspect_path, filename)
    
    cv2.imwrite(full_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    print(f"[SECURITY ALERT] Evidence saved at: {full_path}")

# --- HELPER: SAVE ATTENDANCE PROOF ---
def save_compressed_proof(frame, profile, mode, time_str):
    safe_name = profile['name'].replace(" ", "_")
    safe_id = "".join([c for c in profile['id'] if c.isalnum() or c in ('-','_')])
    subfolder = "IN" if mode == "LOGIN" else "OUT"
    daily_path = os.path.join(PATH_PROOFS, today_date)
    user_path = os.path.join(daily_path, f"{safe_name}_{safe_id}", subfolder)
    
    os.makedirs(user_path, exist_ok=True)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (320, 240))
    filename = f"{mode}_{time_str.replace(':','-')}.jpg"
    full_path = os.path.join(user_path, filename)
    cv2.imwrite(full_path, small_gray, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return full_path

def update_attendance(profile, frame, action_type):
    current_csv, _ = get_today_csv()
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    proof_path = save_compressed_proof(frame, profile, action_type, time_str)
    
    if action_type == "LOGIN":
        status = "LATE" if time_str > LATE_CUTOFF else "ON TIME"
        with open(current_csv, 'a') as f:
            f.write(f'\n{profile["name"]},{profile["dept"]},{profile["role"]},{profile["id"]},{today_date},{time_str},{status},{proof_path},,')
    elif action_type == "SIGNOUT":
        lines = []
        with open(current_csv, 'r') as f: lines = f.readlines()
        with open(current_csv, 'w') as f:
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) > 3 and parts[3] == profile['id']:
                    base_data = ",".join(parts[:8]) 
                    new_line = f"{base_data},{time_str},{proof_path}\n"
                    f.write(new_line)
                else: f.write(line)

# --- UI DRAWING ---
def draw_ui(img, face_box, name, role, dept, staff_id, status_msg, color):
    top, right, bottom, left = face_box
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    cv2.rectangle(img, (left, top - 35), (right, top), color, cv2.FILLED)
    (text_w, text_h), _ = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = left + (right - left - text_w) // 2
    cv2.putText(img, status_msg, (text_x, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    h, w, _ = img.shape
    panel_h = 100
    cv2.rectangle(img, (0, h - panel_h), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.line(img, (0, h - panel_h), (w, h - panel_h), color, 3)
    cv2.putText(img, name.upper(), (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT_MAIN, 2)
    cv2.putText(img, f"{role} | {dept}", (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_SUB, 1)
    cv2.putText(img, f"ID: {staff_id}", (w - 250, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1) # Mirror immediately
    img_display = img.copy() # Copy for UI
    
    frame_count += 1
    current_time = datetime.now().strftime('%H:%M:%S')

    # --- A. PHONE DETECTION ---
    is_phone_detected = False
    results = model_spoof(img, stream=True, verbose=False, conf=0.5) 
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == CLASS_ID_PHONE:
                is_phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_display, (x1, y1), (x2, y2), COLOR_WARNING, 3)
                cv2.putText(img_display, "PHONE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARNING, 2)

    # =========================================================
    # B. SECURITY MODE (Override Attendance)
    # =========================================================
    if is_phone_detected:
        if suspicious_timer > 0: suspicious_timer -= 1
        
        # 1. UI Alert
        cv2.rectangle(img_display, (300, 300), (980, 420), COLOR_WARNING, cv2.FILLED)
        cv2.putText(img_display, "SECURITY ALERT", (480, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img_display, "PHONE DETECTED - LOGGING SUSPECT", (350, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Recognition Logic (Only runs if timer allows, to save FPS)
        if suspicious_timer == 0:
            print("[ALERT] Attempting to identify suspect...")
            
            # Convert for recognition
            img_full_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # UPSAMPLE=2: Better at finding faces next to phones
            suspect_locs = face_recognition.face_locations(img_full_rgb, number_of_times_to_upsample=2)
            suspect_encodes = face_recognition.face_encodings(img_full_rgb, suspect_locs)
            
            found_suspects = False
            
            if not suspect_encodes:
                 # Phone seen, but no face clear enough -> Save as Unknown
                 save_suspicious_proof(img, "Unknown", "N/A")
            
            for i, encode in enumerate(suspect_encodes):
                matches = face_recognition.compare_faces(known_encodings, encode, tolerance=SUSPICIOUS_TOLERANCE)
                faceDis = face_recognition.face_distance(known_encodings, encode)
                
                suspect_name = "Unknown"
                suspect_id = "N/A"
                
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex] and faceDis[matchIndex] < SUSPICIOUS_TOLERANCE:
                        suspect_name = known_names[matchIndex]
                        suspect_id = known_ids[matchIndex]
                
                # === CRITICAL: SAVE TO SUSPICIOUS FOLDER, NOT ATTENDANCE ===
                save_suspicious_proof(img, suspect_name, suspect_id)
                found_suspects = True

            # Reset Cooldown (wait 30 frames before snapping again)
            suspicious_timer = SUSPICIOUS_COOLDOWN_FRAMES

    # =========================================================
    # C. SUCCESS MESSAGE DISPLAY
    # =========================================================
    elif state == "SUCCESS_SHOW":
        success_timer += 1
        if process_mode == "LOGIN":
            box_color = COLOR_LOGIN; msg_top = "LOGIN SUCCESSFUL"; msg_sub = f"Welcome, {current_user_profile.get('name', 'User')}"
        else:
            box_color = COLOR_SIGNOUT; msg_top = "SIGNOUT SUCCESSFUL"; msg_sub = "Have a great evening!"

        h, w, _ = img_display.shape
        cv2.rectangle(img_display, (w//2 - 250, h//2 - 100), (w//2 + 250, h//2 + 100), box_color, 3)
        cv2.rectangle(img_display, (w//2 - 250, h//2 - 100), (w//2 + 250, h//2 + 100), (0,0,0), cv2.FILLED)
        cv2.putText(img_display, msg_top, (w//2 - 180, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        cv2.putText(img_display, msg_sub, (w//2 - 200, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        if success_timer >= SUCCESS_DISPLAY_TIME:
            state = "SCANNING"; success_timer = 0; blink_counter = 0

    # =========================================================
    # D. STANDARD ATTENDANCE (Only runs if NO phone detected)
    # =========================================================
    else:
        img, facesMesh = detector_blink.findFaceMesh(img, draw=False)
        
        if frame_count % FRAME_SKIP == 0:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            last_face_locations = face_recognition.face_locations(imgS)
            encodings = face_recognition.face_encodings(imgS, last_face_locations)

            last_face_names = []
            last_face_roles = []
            last_face_ids = []
            last_face_depts = []

            for encodeFace in encodings:
                matches = face_recognition.compare_faces(known_encodings, encodeFace, tolerance=RECOGNITION_TOLERANCE)
                faceDis = face_recognition.face_distance(known_encodings, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex] and faceDis[matchIndex] < RECOGNITION_TOLERANCE:
                    last_face_names.append(known_names[matchIndex])
                    last_face_roles.append(known_roles[matchIndex])
                    last_face_ids.append(known_ids[matchIndex])
                    last_face_depts.append(known_depts[matchIndex])
                else:
                    last_face_names.append("Unknown"); last_face_roles.append(""); last_face_ids.append(""); last_face_depts.append("")

        for i, (top, right, bottom, left) in enumerate(last_face_locations):
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            name = last_face_names[i] if i < len(last_face_names) else "Unknown"
            
            if name != "Unknown":
                staff_id = last_face_ids[i]
                role = last_face_roles[i]
                dept = last_face_depts[i]
                user_status = attendance_cache.get(staff_id, "NOT_LOGGED")
                
                can_blink = False
                status_msg = ""; status_color = (255, 255, 255)

                if user_status == "NOT_LOGGED":
                    can_blink = True; status_msg = "BLINK TO LOGIN"; status_color = COLOR_LOGIN; process_mode = "LOGIN"
                elif user_status == "LOGGED_IN":
                    if current_time >= SIGNOUT_START_TIME:
                        can_blink = True; status_msg = "BLINK TO SIGN OUT"; status_color = COLOR_SIGNOUT; process_mode = "SIGNOUT"
                    else:
                        can_blink = False; status_msg = "ALREADY LOGGED IN"; status_color = COLOR_ALREADY
                elif user_status == "SIGNED_OUT":
                    can_blink = False; status_msg = "ALREADY SIGNED OUT"; status_color = COLOR_LOCKED

                draw_ui(img_display, (top, right, bottom, left), name, role, dept, staff_id, status_msg, status_color)
                
                if can_blink and facesMesh:
                    face = facesMesh[0]
                    leftUp, leftDown = face[159], face[23]
                    leftLeft, leftRight = face[130], face[243]
                    v_dist, _ = detector_blink.findDistance(leftUp, leftDown)
                    h_dist, _ = detector_blink.findDistance(leftLeft, leftRight)
                    ratio = int((v_dist / h_dist) * 100) if h_dist != 0 else 100
                    
                    if ratio < BLINK_RATIO_THRESHOLD:
                        blink_counter += 1
                        bar_len = int((blink_counter / BLINK_FRAMES_REQUIRED) * (right - left))
                        cv2.rectangle(img_display, (left, bottom + 10), (left + bar_len, bottom + 15), status_color, cv2.FILLED)
                    else: blink_counter = 0 
                    
                    if blink_counter >= BLINK_FRAMES_REQUIRED:
                        current_user_profile = {"name": name, "dept": dept, "role": role, "id": staff_id}
                        if process_mode == "LOGIN":
                            update_attendance(current_user_profile, img, "LOGIN")
                            attendance_cache[staff_id] = "LOGGED_IN"
                        else:
                            update_attendance(current_user_profile, img, "SIGNOUT")
                            attendance_cache[staff_id] = "SIGNED_OUT"
                        state = "SUCCESS_SHOW"; blink_counter = 0
            else:
                cv2.rectangle(img_display, (left, top), (right, bottom), (100, 100, 100), 2)
                cv2.putText(img_display, "UNKNOWN", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    # --- TOP BAR ---
    cv2.rectangle(img_display, (0, 0), (1280, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(img_display, f"TIME: {current_time}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mode_text = "SIGN OUT ACTIVE" if current_time >= SIGNOUT_START_TIME else "LOGIN ACTIVE"
    mode_color = COLOR_SIGNOUT if current_time >= SIGNOUT_START_TIME else COLOR_LOGIN
    cv2.putText(img_display, mode_text, (1050, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    cv2.imshow('NBTI Attendance System', img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()