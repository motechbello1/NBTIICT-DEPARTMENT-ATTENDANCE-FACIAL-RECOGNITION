import cv2
import numpy as np
import face_recognition
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from cvzone.FaceMeshModule import FaceMeshDetector
from datetime import datetime

# --- CONFIGURATION ---
# Note: On GitHub, this folder must be inside the repo.
PATH_IMAGES = 'ImagesAttendance'
LATE_CUTOFF = "08:30:00"

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBTI Smart Attendance", layout="wide")
st.title("NBTI ICT Department: Smart Attendance")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/151/151770.png", width=100)
st.sidebar.markdown("### System Status: 🟢 Online")

# --- 1. LOAD DATABASE (CACHED) ---
# We use @st.cache_resource so it only trains ONCE when the app starts, not every second.
@st.cache_resource
def load_encodings():
    print("Loading Database...")
    known_encodings = []
    known_names = []
    known_roles = []
    known_ids = []
    known_depts = []

    if not os.path.exists(PATH_IMAGES):
        return [], [], [], [], []

    for root, dirs, files in os.walk(PATH_IMAGES):
        if root == PATH_IMAGES: continue

        folder_name = os.path.basename(root)
        parts = folder_name.split('_')

        if len(parts) >= 4:
            name, dept, role, staff_id = parts[0], parts[1], parts[2], parts[3]
        else:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                img = cv2.imread(full_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                try:
                    encode = face_recognition.face_encodings(img)[0]
                    known_encodings.append(encode)
                    known_names.append(name)
                    known_roles.append(role)
                    known_ids.append(staff_id)
                    known_depts.append(dept)
                except:
                    pass
    
    return known_encodings, known_names, known_roles, known_ids, known_depts

# Load data immediately
known_encodings, known_names, known_roles, known_ids, known_depts = load_encodings()

if len(known_encodings) == 0:
    st.error("No faces found! Please check the 'ImagesAttendance' folder in your GitHub repo.")

# --- 2. WEBRTC PROCESSOR ---
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector_blink = FaceMeshDetector(maxFaces=1)
        self.frame_count = 0
        self.blink_counter = 0
        self.state = "SCANNING"
        self.success_timer = 0
        self.matched_profile = None
        
        # Performance: Skip frames to reduce lag on Cloud
        self.process_every_n_frames = 3 
        
        # Cache for last known locations to make video smooth
        self.last_locs = []
        self.last_names = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame_count += 1

        # 1. Blink Detection (Always runs)
        img, facesMesh = self.detector_blink.findFaceMesh(img, draw=False)

        # 2. Face Recognition (Throttled)
        if self.frame_count % self.process_every_n_frames == 0 and self.state != "SUCCESS":
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            self.last_locs = face_recognition.face_locations(imgS)
            encodes = face_recognition.face_encodings(imgS, self.last_locs)
            
            self.last_names = [] # Reset cache
            
            for encodeFace in encodes:
                matches = face_recognition.compare_faces(known_encodings, encodeFace, tolerance=0.5)
                faceDis = face_recognition.face_distance(known_encodings, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    profile = {
                        "name": known_names[matchIndex],
                        "role": known_roles[matchIndex],
                        "id": known_ids[matchIndex]
                    }
                    self.last_names.append(profile)
                else:
                    self.last_names.append(None)

        # 3. Draw Logic
        if self.state == "SUCCESS":
            self.success_timer += 1
            cv2.rectangle(img, (100, 100), (540, 300), (0, 255, 0), -1)
            cv2.putText(img, "ATTENDANCE LOGGED", (120, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
            if self.success_timer > 30: # Reset after ~1 second
                self.state = "SCANNING"
                self.success_timer = 0
                self.blink_counter = 0

        else:
            # Loop through cached results
            for i, (top, right, bottom, left) in enumerate(self.last_locs):
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                
                if i < len(self.last_names) and self.last_names[i] is not None:
                    profile = self.last_names[i]
                    
                    # Blink Logic
                    ratio = 100
                    if facesMesh:
                        face = facesMesh[0]
                        leftUp, leftDown = face[159], face[23]
                        leftLeft, leftRight = face[130], face[243]
                        v_dist, _ = self.detector_blink.findDistance(leftUp, leftDown)
                        h_dist, _ = self.detector_blink.findDistance(leftLeft, leftRight)
                        ratio = int((v_dist / h_dist) * 100) if h_dist != 0 else 100

                    if ratio < 36:
                        self.blink_counter += 1
                        color = (0, 165, 255) # Orange
                        msg = "VERIFYING..."
                    else:
                        color = (0, 0, 255) # Red
                        msg = "HOLD BLINK"
                        # Don't reset counter immediately to be more forgiving on laggy streams

                    if self.blink_counter > 10: # Threshold
                        self.state = "SUCCESS"
                        self.blink_counter = 0
                        # HERE: You would normally save to CSV/Firebase
                    
                    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                    cv2.putText(img, msg, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                    cv2.putText(img, profile['name'], (left, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                
                else:
                    cv2.rectangle(img, (left, top), (right, bottom), (100, 100, 100), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. RUN WEBRTC ---
st.write("Click **START** to open the webcam. Please ensure you allow browser permissions.")
webrtc_streamer(
    key="attendance",
    video_processor_factory=AttendanceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
)

# --- 4. SHOW LOGS (Optional Preview) ---
st.markdown("---")
st.subheader("📋 Today's Logs")
st.info("Note: In this demo, logs reset when the app restarts.")
