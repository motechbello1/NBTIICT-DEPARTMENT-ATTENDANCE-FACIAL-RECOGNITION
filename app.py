import cv2
import numpy as np
import face_recognition
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from cvzone.FaceMeshModule import FaceMeshDetector

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBTI Smart Attendance", layout="wide")
st.title("NBTI ICT Department: Smart Attendance")

# --- 1. LOAD DATABASE ---
# This runs once and saves time
@st.cache_resource
def load_encodings():
    path = 'ImagesAttendance'
    encodings = []
    names = []
    roles = []
    ids = []
    
    if not os.path.exists(path):
        return [], [], [], []
        
    for root, dirs, files in os.walk(path):
        if root == path: continue
        
        # Folder name format: NAME_DEPT_ROLE_ID
        folder_name = os.path.basename(root)
        parts = folder_name.split('_')
        
        if len(parts) >= 4:
            name, role, staff_id = parts[0], parts[2], parts[3]
        else:
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    encode = face_recognition.face_encodings(img)[0]
                    encodings.append(encode)
                    names.append(name)
                    roles.append(role)
                    ids.append(staff_id)
                except:
                    pass
    return encodings, names, roles, ids

known_encodings, known_names, known_roles, known_ids = load_encodings()

if not known_encodings:
    st.warning("Database empty. Please ensure 'ImagesAttendance' folder is uploaded correctly.")

# --- 2. WEBRTC PROCESSOR ---
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FaceMeshDetector(maxFaces=1)
        # Scan every 5th frame to prevent freezing
        self.process_every = 5
        self.count = 0
        self.last_res = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.count += 1
        
        # Detect Face Mesh (Blink)
        img, faces = self.detector.findFaceMesh(img, draw=False)
        
        # Face Recognition Logic (Throttled)
        if self.count % self.process_every == 0:
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            locs = face_recognition.face_locations(imgS)
            encodes = face_recognition.face_encodings(imgS, locs)
            
            self.last_res = []
            for encode, loc in zip(encodes, locs):
                matches = face_recognition.compare_faces(known_encodings, encode, tolerance=0.5)
                dist = face_recognition.face_distance(known_encodings, encode)
                
                name = "Unknown"
                role = ""
                if True in matches:
                    matchIndex = np.argmin(dist)
                    name = known_names[matchIndex]
                    role = known_roles[matchIndex]
                
                # Scale up location
                y1, x2, y2, x1 = loc
                self.last_res.append((name, role, (x1*4, y1*4, x2*4, y2*4)))

        # Draw results from cache
        for name, role, (x1, y1, x2, y2) in self.last_res:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name}", (x1, y2+30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            if role:
                cv2.putText(img, role, (x1, y2+60), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. RUN APP ---
st.write("### 📸 Live Attendance Camera")
webrtc_streamer(key="sample", video_processor_factory=AttendanceProcessor)
