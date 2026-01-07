import cv2
import numpy as np
import face_recognition
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. CONFIGURATION & BEIGE/GREEN THEME ---
st.set_page_config(page_title="NBTI Smart Attendance", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    /* 1. Main Background - Warm Beige */
    .stApp {
        background-color: #f4f4f0;
    }

    /* 2. Sidebar - Black */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* 3. Fonts & Headers - Dark Grey */
    h1, h2, h3, h4, p, label {
        color: #1a1a1a !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* 4. Metrics Cards - White with Green Text */
    div[data-testid="stMetricValue"] {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        color: #2e7d32 !important; /* NBTI Green */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #666 !important;
    }

    /* 5. Buttons - Forest Green */
    .stButton>button {
        background-color: #2e7d32;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }

    /* 6. Video Container */
    video {
        width: 100% !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST DATABASE LOADER (Fixes "Unsupported Type" Error) ---
@st.cache_resource
def load_encodings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'ImagesAttendance')
    
    encodings = []
    names = []
    roles = []
    ids = []
    
    if not os.path.exists(path):
        return [], [], [], []
        
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                
                # Parse Folder Name: Name_Dept_Role_ID
                # Example: MusaBello_ICT_Analyst_1197
                folder_name = os.path.basename(root)
                parts = folder_name.split('_')
                
                name = "Unknown"
                role = "Staff"
                staff_id = "N/A"

                if len(parts) >= 4:
                    name = parts[0]
                    role = parts[2]
                    staff_id = parts[3]
                elif root != path:
                    name = folder_name # Fallback if underscores are missing

                # Formatting Name
                name = name.replace("_", " ")

                try:
                    # FORCE READ AS RGB (Fixes 'Unsupported image type')
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    # Ensure 8-bit format for dlib
                    img = np.array(img, dtype=np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect Face
                    encs = face_recognition.face_encodings(img)
                    
                    if encs:
                        encodings.append(encs[0])
                        names.append(name)
                        roles.append(role)
                        ids.append(staff_id)
                        print(f"✅ Loaded: {name}")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
                    pass
                    
    return encodings, names, roles, ids

known_encodings, known_names, known_roles, known_ids = load_encodings()

# --- 3. SIDEBAR & HEADER ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Control Panel")
    st.write("---")
    mirror = st.checkbox("Mirror Camera", value=True)
    st.write("---")
    st.markdown("**System Status:**")
    if known_encodings:
        st.success("Online")
    else:
        st.error("Offline")

st.title("NBTI Smart Attendance")
st.markdown("#### 📍 ICT Department Dashboard")

# Stats Grid
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Staff", len(set(known_names)) if known_names else 0)
c2.metric("Department", "ICT")
c3.metric("Camera", "Active")
c4.metric("Date", "Oct 25, 2025")

st.write("---")

# --- 4. VIDEO LOGIC ---
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.mirror = True 

    def update_settings(self, mirror_on):
        self.mirror = mirror_on

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.mirror:
            img = cv2.flip(img, 1)

        self.frame_count += 1
        
        # Process every 3rd frame to improve speed
        if self.frame_count % 3 == 0 and known_encodings:
            # Resize for speed (0.5x)
            small_frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            faces = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, faces)
            
            for encode, loc in zip(encs, faces):
                matches = face_recognition.compare_faces(known_encodings, encode, tolerance=0.5)
                dist = face_recognition.face_distance(known_encodings, encode)
                
                name = "Unknown"
                role = ""
                
                if True in matches:
                    best_match_idx = np.argmin(dist)
                    if dist[best_match_idx] < 0.55:
                        name = known_names[best_match_idx]
                        role = known_roles[best_match_idx]
                
                # Scale coordinates back up (x2)
                top, right, bottom, left = loc
                top, right, bottom, left = top*2, right*2, bottom*2, left*2
                
                # DRAWING UI
                # Green for known, Red for unknown
                color = (46, 125, 50) if name != "Unknown" else (0, 0, 255) 
                
                # 1. Bounding Box
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                
                # 2. Name Tag Background
                cv2.rectangle(img, (left, bottom - 40), (right, bottom), color, cv2.FILLED)
                
                # 3. Text
                cv2.putText(img, name, (left + 10, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
                
                # 4. Role (Floating above)
                if role:
                     cv2.putText(img, role, (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. STREAMER ---
ctx = webrtc_streamer(
    key="attendance",
    video_processor_factory=AttendanceProcessor,
    # GOOGLE STUN SERVER FIXES TIMEOUT ISSUES
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.update_settings(mirror)

# Footer
st.markdown("<br><hr><div style='text-align: center; color: #888;'>NBTI ICT Department System</div>", unsafe_allow_html=True)
