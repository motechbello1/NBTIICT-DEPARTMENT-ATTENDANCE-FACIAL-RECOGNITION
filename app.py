import cv2
import numpy as np
import face_recognition
import os
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. PAGE CONFIGURATION & DARK THEME ---
st.set_page_config(page_title="NBTI Smart Attendance", layout="wide", page_icon="🛡️")

# Custom CSS: Dark Mode (Black, Dark Grey, Green, Beige Text)
st.markdown("""
    <style>
    /* Main Background - Dark Charcoal */
    .stApp {
        background-color: #121212;
    }
    
    /* Text Colors - Off-White/Beige for readability */
    h1, h2, h3, p, div, label, span {
        color: #e0e0e0 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Sidebar - Pure Black */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }

    /* Metrics/Stats Cards - Dark Grey */
    div[data-testid="stMetricValue"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 10px;
        color: #4CAF50 !important; /* Bright Green */
    }
    
    /* Buttons - Green */
    .stButton>button {
        background-color: #2e7d32;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }

    /* Error/Success Messages */
    .stAlert {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST DATABASE LOADER (RECURSIVE) ---
@st.cache_resource
def load_encodings():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, 'ImagesAttendance')
    
    encodings = []
    names = []
    
    if not os.path.exists(path):
        return [], []
        
    # Walk through ALL directories to find images inside sub-folders
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                
                # Try to extract name from folder name first (per your screenshot structure)
                # Structure seems to be: ImagesAttendance/Name_Dept_Role/Image.jpg
                folder_name = os.path.basename(root)
                
                # If image is directly in ImagesAttendance, use filename
                if root == path:
                    person_name = os.path.splitext(file)[0]
                else:
                    # Parse the folder name "BelloMuhammadMustapha_ICT_..."
                    # We take the first part before the first underscore as the name
                    # or just use the whole folder name if you prefer.
                    parts = folder_name.split('_')
                    if len(parts) > 0:
                         # Joins "Bello" "Muhammad" "Mustapha" if they are separated
                         # Modify this logic if your naming convention is different
                         person_name = parts[0] 
                    else:
                        person_name = folder_name

                # Clean up name display
                person_name = person_name.replace("_", " ")

                try:
                    img = cv2.imread(img_path)
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Find faces
                    face_locations = face_recognition.face_locations(img)
                    if face_locations:
                        encode = face_recognition.face_encodings(img, face_locations)[0]
                        encodings.append(encode)
                        names.append(person_name)
                        print(f"Loaded: {person_name}")
                except Exception as e:
                    print(f"Skipping {file}: {e}")
                    pass
                    
    return encodings, names

known_encodings, known_names = load_encodings()

# --- 3. UI HEADER ---
st.title("NBTI Smart Attendance")
st.markdown("#### 📍 ICT Department | Facial Recognition System")

# Stats Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("System Status", "Online" if known_encodings else "Waiting for Database")
with col2:
    st.metric("Staff Profiles", len(known_names))
with col3:
    st.metric("Active Camera", "Ready")

st.markdown("---")

# --- 4. VIDEO PROCESSOR ---
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for speed
        self.last_results = []
        self.mirror_mode = True  # Default

    def update_settings(self, mirror):
        self.mirror_mode = mirror

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror Flip (Controlled by user)
        if self.mirror_mode:
            img = cv2.flip(img, 1)

        self.frame_count += 1

        # Skip frames to save CPU (Crucial for Cloud)
        if self.frame_count % self.process_every_n_frames == 0:
            # Resize small for fast processing (1/2 size)
            small_frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_results = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Scale coordinates back up (x2)
                top, right, bottom, left = face_location
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                current_results.append((name, (left, top, right, bottom)))
            
            self.last_results = current_results

        # Draw results (User sees this every frame)
        for name, (left, top, right, bottom) in self.last_results:
            # Color: Green for Known, Red for Unknown
            color = (46, 125, 50) if name != "Unknown" else (0, 0, 255)
            
            # Box
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            
            # Label
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. MAIN VIDEO SECTION ---
# Mirror Toggle
mirror_check = st.checkbox("Mirror Camera (Flip View)", value=True)

# WebRTC Streamer (Full Width)
ctx = webrtc_streamer(
    key="attendance-app",
    video_processor_factory=AttendanceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Pass the checkbox value to the processor
if ctx.video_processor:
    ctx.video_processor.update_settings(mirror_check)

# --- 6. DEBUGGING HELP ---
if not known_encodings:
    st.error("⚠️ Database Empty!")
    st.write("Ensure your files are named like `John_Doe.jpg` and inside the `ImagesAttendance` folder.")
    # Debug: Show what the app sees
    base_dir = os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, 'ImagesAttendance')
    if os.path.exists(target_dir):
        st.write(f"📂 Scanning folder: {target_dir}")
        found_files = []
        for r, d, f in os.walk(target_dir):
            for file in f:
                found_files.append(os.path.join(r, file))
        st.write(f"Files found: {found_files}")
    else:
        st.write("❌ 'ImagesAttendance' folder not found.")

st.markdown("<br><br><div style='text-align: center; color: #666;'>NBTI ICT Department © 2025</div>", unsafe_allow_html=True)
