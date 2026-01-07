import cv2
import numpy as np
import face_recognition
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. CONFIGURATION & CSS STYLING ---
st.set_page_config(page_title="NBTI Smart Attendance", layout="wide", page_icon="🛡️")

# Custom CSS for "Beige, Grey, Black, Green" Theme
st.markdown("""
    <style>
    /* Main Background - Warm Beige */
    .stApp {
        background-color: #f4f4f0;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1a1a1a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }

    /* Sidebar - Darker Grey */
    [data-testid="stSidebar"] {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #e0e0e0;
    }

    /* Cards/Metrics - White with Shadow */
    div[data-testid="stMetricValue"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #2e7d32; /* Green Text */
    }
    
    /* Buttons - Forest Green */
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }

    /* Warning/Info Boxes */
    .stAlert {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATABASE (Fixed Path) ---
@st.cache_resource
def load_encodings():
    # Use relative path to ensure it works on Cloud
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, 'ImagesAttendance')
    
    encodings = []
    names = []
    roles = []
    ids = []
    
    # Debug: Check if folder exists
    if not os.path.exists(path):
        st.error(f"⚠️ Error: The folder '{path}' was not found.")
        st.write(f"Current Directory: {os.getcwd()}")
        st.write(f"Files in current dir: {os.listdir(base_dir)}")
        return [], [], [], []
        
    for root, dirs, files in os.walk(path):
        if root == path: continue
        
        # Folder structure: NAME_DEPT_ROLE_ID
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

# --- 3. UI HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("NBTI Smart Attendance")
    st.markdown("##### 📍 ICT Department | Facial Recognition System")
with col2:
    if known_names:
        st.metric(label="Staff Loaded", value=len(set(known_ids)), delta="Active Database")
    else:
        st.metric(label="Status", value="Offline", delta_color="inverse")

st.markdown("---")

# --- 4. VIDEO PROCESSOR ---
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.process_every = 3  # Faster processing (check every 3rd frame)
        self.count = 0
        self.last_res = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. MIRRORING
        img = cv2.flip(img, 1)
        self.count += 1
        
        # 2. DETECTION LOGIC
        if self.count % self.process_every == 0:
            # Increased scale from 0.25 to 0.5 for better detection
            imgS = cv2.resize(img, (0,0), None, 0.5, 0.5) 
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
                    # Double check tolerance to avoid false positives
                    if dist[matchIndex] < 0.50:
                        name = known_names[matchIndex]
                        role = known_roles[matchIndex]
                
                # Scale up location (x2 because we scaled down by 0.5)
                y1, x2, y2, x1 = loc
                self.last_res.append((name, role, (x1*2, y1*2, x2*2, y2*2)))

        # 3. DRAWING (Beautified)
        for name, role, (x1, y1, x2, y2) in self.last_res:
            # Color: Green for Known, Red for Unknown
            color = (46, 125, 50) if name != "Unknown" else (0, 0, 255) # RGB for Green/Red
            
            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label Background
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            
            # Name Text
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Role Text (Above box)
            if role:
                cv2.putText(img, role, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. MAIN LAYOUT ---
c1, c2 = st.columns([2, 1])

with c1:
    st.info("System Ready. Please look directly at the camera.")
    
    # WebRTC Streamer with specific settings to try and force HD
    webrtc_streamer(
        key="attendance",
        video_processor_factory=AttendanceProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with c2:
    st.markdown("### 📋 Quick Stats")
    st.write("Live detection logs will appear here in future updates.")
    
    if not known_encodings:
        st.error("No faces loaded.")
        st.write("Please check the 'ImagesAttendance' folder.")
    else:
        st.success(f"Database Active: {len(known_encodings)} profiles")
        with st.expander("View Staff List"):
            st.write(list(set(known_names)))

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>NBTI ICT Department © 2025 | Powered by Streamlit & OpenCV</div>", unsafe_allow_html=True)
