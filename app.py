import cv2
import numpy as np
import face_recognition
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. PAGE CONFIG & CSS FOR FULL WIDTH VIDEO ---
st.set_page_config(page_title="NBTI Smart Attendance", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    /* Force Video to be Wide */
    video {
        width: 100% !important;
        height: auto !important;
        border-radius: 10px;
        border: 2px solid #333;
    }
    
    /* Dark Theme Colors */
    .stApp { background-color: #121212; }
    h1, h2, h3, p, label { color: #e0e0e0 !important; }
    div[data-testid="stMetricValue"] { background-color: #1e1e1e; color: #4CAF50 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE LOADER WITH DEBUGGING ---
@st.cache_resource
def load_encodings():
    # Force absolute path to avoid confusion
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'ImagesAttendance')
    
    encodings = []
    names = []
    debug_log = [] # To store success/failure reasons
    
    if not os.path.exists(path):
        return [], [], ["Folder not found"]
        
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                
                # Extract Name
                # If inside subfolder, use folder name, else filename
                if root != path:
                    folder_name = os.path.basename(root)
                    person_name = folder_name.split('_')[0] # Take first part "DavidAdamu"
                else:
                    person_name = os.path.splitext(file)[0]
                
                person_name = person_name.replace("_", " ")

                try:
                    # Load Image
                    img = cv2.imread(img_path)
                    if img is None:
                        debug_log.append(f"❌ {file}: Failed to load (corrupt?)")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect Faces (Try Upscaling if small)
                    # number_of_times_to_upsample=1 helps find smaller faces
                    encs = face_recognition.face_encodings(img, num_jitters=1, model='small')
                    
                    if not encs:
                         # Fallback: Try finding face locations first with upsizing
                        locs = face_recognition.face_locations(img, number_of_times_to_upsample=2)
                        encs = face_recognition.face_encodings(img, locs)

                    if encs:
                        encodings.append(encs[0])
                        names.append(person_name)
                        debug_log.append(f"✅ {person_name}: Loaded")
                    else:
                        debug_log.append(f"⚠️ {file}: Image loaded, but NO FACE detected.")
                        
                except Exception as e:
                    debug_log.append(f"❌ {file}: Error - {str(e)}")
                    pass
                    
    return encodings, names, debug_log

known_encodings, known_names, debug_info = load_encodings()

# --- 3. HEADER & STATS ---
st.title("NBTI Smart Attendance")
c1, c2, c3 = st.columns(3)
c1.metric("Database Status", "Online" if known_encodings else "Empty")
c2.metric("Profiles Loaded", len(set(known_names)))
c3.metric("System", "Active")

# --- 4. VIDEO PROCESSOR (FIXED MIRROR) ---
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.mirror = True # Default to True

    def update_settings(self, mirror_on):
        self.mirror = mirror_on

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. APPLY MIRROR IMMEDIATELY
        if self.mirror:
            img = cv2.flip(img, 1)

        # 2. FACE RECOGNITION (Every 2nd frame)
        self.frame_count += 1
        if self.frame_count % 2 == 0 and known_encodings:
            
            # Optimization: Resize frame to 0.5x for faster processing
            small_frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            faces = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, faces)
            
            for encode, loc in zip(encs, faces):
                matches = face_recognition.compare_faces(known_encodings, encode, tolerance=0.55)
                dist = face_recognition.face_distance(known_encodings, encode)
                
                name = "Unknown"
                if True in matches:
                    best_match_idx = np.argmin(dist)
                    if dist[best_match_idx] < 0.55:
                        name = known_names[best_match_idx]
                
                # Scale Coords back to 1.0x
                top, right, bottom, left = loc
                top, right, bottom, left = top*2, right*2, bottom*2, left*2
                
                # Draw Box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 3. DEBUG OVERLAY (Shows Mirror Status on Screen)
        status_text = "MIRROR: ON" if self.mirror else "MIRROR: OFF"
        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. MAIN UI ---
st.write("---")
toggle = st.checkbox("Mirror Camera", value=True)

ctx = webrtc_streamer(
    key="attendance",
    video_processor_factory=AttendanceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.update_settings(toggle)

# --- 6. DATABASE DIAGNOSTICS (CRITICAL FOR YOU) ---
st.write("---")
with st.expander("🔍 Troubleshooting & Database Logs", expanded=True):
    if not known_encodings:
        st.error("DATABASE IS EMPTY - See reasons below:")
    else:
        st.success("Database Loaded Successfully")
        
    # Show the log of what happened to each file
    st.write("Processing Log:")
    for log in debug_info:
        if "❌" in log or "⚠️" in log:
            st.warning(log)
        else:
            st.text(log)
